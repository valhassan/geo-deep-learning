"""Low-Rank Adaptation (DoRA) wrapper tuned for Dinov3."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import torch
import torch.nn.functional as fn
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


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
        # Full QKV matrix for eval. Rebuilt when weights or device change.
        self._merged_weight: torch.Tensor | None = None

        self.reset_parameters()
        self.register_load_state_dict_post_hook(self._clear_merged_on_load)

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
        self._merged_weight = None

    @override
    def train(self, mode: bool = True) -> DoRAQKVWrapper:
        """Drop the inference weight when training resumes."""
        if mode:
            self._merged_weight = None
        return super().train(mode)

    @override
    def _apply(
        self,
        func: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> DoRAQKVWrapper:
        """Device or dtype changes invalidate the cached matrix."""
        self._merged_weight = None
        return super()._apply(func, recurse=recurse)

    @staticmethod
    def _clear_merged_on_load(module: nn.Module, incompatible_keys: object) -> None:
        del incompatible_keys
        module._merged_weight = None  # noqa: SLF001

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

    def _merged_qkv_weight(self) -> torch.Tensor:
        """Materialize the QKV matrix once per eval session."""
        if self._merged_weight is None:
            device_type = self.qkv.weight.device.type
            with torch.no_grad(), torch.autocast(device_type, enabled=False):
                self._merged_weight = self._compose_qkv_weight().detach()
        return self._merged_weight

    def _masked_bias(self) -> torch.Tensor | None:
        """Apply the DINOv3 key-bias mask when the inner linear has one."""
        if hasattr(self.qkv, "bias_mask") and self.qkv.bias is not None:
            return self.qkv.bias * self.qkv.bias_mask.to(self.qkv.bias.dtype)
        return self.qkv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bias = self._masked_bias()
        if self.training:
            return fn.linear(self.dropout(x), self._compose_qkv_weight(), bias)
        return fn.linear(x, self._merged_qkv_weight(), bias)
