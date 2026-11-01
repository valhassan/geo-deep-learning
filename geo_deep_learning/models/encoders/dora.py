"""Low-Rank Adaptation (DoRA) wrapper tuned for Dinov3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as fn
from torch import nn


class DoRAQKVWrapper(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) wrapper ."""

    def __init__(
        self,
        qkv: nn.Module,
        r: int = 16,
        alpha: int = 32,
        dropout_rate: float = 0.05,
    ) -> None:
        """Initialize the DoRAQKVWrapper."""
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

        # Freeze the original weights
        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Trainable components for Query (Q) and Value (V)
        self.lora_a_q = nn.Parameter(torch.empty((r, self.dim)))
        self.lora_b_q = nn.Parameter(torch.empty((self.dim, r)))
        self.mag_q = nn.Parameter(torch.empty(self.dim))

        self.lora_a_v = nn.Parameter(torch.empty((r, self.dim)))
        self.lora_b_v = nn.Parameter(torch.empty((self.dim, r)))
        self.mag_v = nn.Parameter(torch.empty(self.dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the DoRAQKVWrapper."""
        nn.init.kaiming_uniform_(self.lora_a_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q)
        nn.init.kaiming_uniform_(self.lora_a_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v)

        with torch.no_grad():
            self.mag_q.data.copy_(
                torch.linalg.norm(self.qkv.weight[: self.dim, :], dim=1),
            )
            self.mag_v.data.copy_(
                torch.linalg.norm(self.qkv.weight[-self.dim :, :], dim=1),
            )

    def _get_dora_weight(
        self,
        orig_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        mag: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Get the DoRA weight."""
        directional = orig_weight + (lora_b @ lora_a) * self.scaling
        norm = torch.linalg.norm(directional, dim=1, keepdim=True) + eps
        return mag.view(-1, 1) * (directional / norm)

    def _compose_qkv_weight(self) -> torch.Tensor:
        """Build the QKV matrix. K stays at the pretrained weight."""
        w_q = self._get_dora_weight(
            self.qkv.weight[: self.dim, :],
            self.lora_a_q,
            self.lora_b_q,
            self.mag_q,
        )
        w_k = self.qkv.weight[self.dim : 2 * self.dim, :]
        w_v = self._get_dora_weight(
            self.qkv.weight[-self.dim :, :],
            self.lora_a_v,
            self.lora_b_v,
            self.mag_v,
        )
        return torch.cat((w_q, w_k, w_v), dim=0)

    def _masked_bias(self) -> torch.Tensor | None:
        """Apply the DINOv3 key-bias mask when the inner linear has one."""
        if hasattr(self.qkv, "bias_mask") and self.qkv.bias is not None:
            return self.qkv.bias * self.qkv.bias_mask.to(self.qkv.bias.dtype)
        return self.qkv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. The QKV weight is rebuilt from current params every call."""
        return fn.linear(
            self.dropout(x),
            self._compose_qkv_weight(),
            self._masked_bias(),
        )
