"""SegFormer MLP decoder."""

import torch
import torch.nn.functional as fn
from torch import nn


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        """Initialize the MLP."""
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x.flatten(2).transpose(1, 2).contiguous()
        return self.proj(x)


class Decoder(nn.Module):
    """Decoder for SegFormer."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b2",
        in_channels: list[int] | None = None,
        feature_strides: list[int] | None = None,
        embedding_dim: int | None = None,
        num_classes: int = 1,
        dropout_ratio: float = 0.1,
        skip_channels: int | None = None,
    ) -> None:
        """Initialize the decoder. skip_channels adds a stride-2 map in front."""
        super().__init__()
        if feature_strides is None:
            feature_strides = [4, 8, 16, 32]
        if in_channels is None:
            in_channels = (
                [32, 64, 160, 256] if encoder == "mit_b0" else [64, 128, 320, 512]
            )
        if embedding_dim is None:
            embedding_dim = 256 if encoder in ("mit_b0", "mit_b1") else 768
        if len(feature_strides) != len(in_channels):
            msg = "feature_strides and in_channels must have the same length"
            raise ValueError(msg)
        if min(feature_strides) != feature_strides[0]:
            msg = "The minimum feature stride must be the first element"
            raise ValueError(msg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = (
            self.in_channels
        )

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        n_scales = 4
        self.linear_c0: MLP | None = None
        if skip_channels is not None:
            self.linear_c0 = MLP(input_dim=skip_channels, embed_dim=embedding_dim)
            n_scales = 5

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim * n_scales,
                out_channels=embedding_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def _project(
        self,
        mlp: MLP,
        feat: torch.Tensor,
        target: tuple[int, int],
    ) -> torch.Tensor:
        n = feat.shape[0]
        projected = (
            mlp(feat)
            .permute(0, 2, 1)
            .reshape(n, -1, feat.shape[2], feat.shape[3])
            .contiguous()
        )
        if projected.shape[2:] == target:
            return projected
        return fn.interpolate(
            projected,
            size=target,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Forward pass. x is [c1..c4] or [skip, c1..c4], finest map first."""
        mlps: list[MLP] = [
            self.linear_c1,
            self.linear_c2,
            self.linear_c3,
            self.linear_c4,
        ]
        if self.linear_c0 is not None:
            mlps = [self.linear_c0, *mlps]
        if len(x) != len(mlps):
            msg = f"expected {len(mlps)} feature maps, got {len(x)}"
            raise ValueError(msg)
        target = x[0].shape[2:]
        projs = [
            self._project(mlp, feat, target)
            for mlp, feat in zip(mlps, x, strict=True)
        ]
        fused = self.linear_fuse(torch.cat(projs[::-1], dim=1))
        dropped = self.dropout(fused)
        return self.linear_pred(dropped), None
