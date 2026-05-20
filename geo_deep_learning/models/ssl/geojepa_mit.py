"""GeoJEPA MixTransformer SSL model."""

import torch
from torch import nn

from geo_deep_learning.models.encoders.mix_transformer import (
    DynamicMixTransformer,
    get_encoder,
)

ENCODER_CHANNELS: dict[str, tuple[int, int, int, int]] = {
    "mit_b0": (32, 64, 160, 256),
    "mit_b1": (64, 128, 320, 512),
    "mit_b2": (64, 128, 320, 512),
    "mit_b3": (64, 128, 320, 512),
    "mit_b4": (64, 128, 320, 512),
    "mit_b5": (64, 128, 320, 512),
}


class GeoJEPAMixTransformer(nn.Module):
    """GeoJEPA MixTransformer SSL model."""

    def __init__(
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
        proj_dim: int = 256,
        *,
        use_dynamic_encoder: bool = False,
    ) -> None:
        """Initialize LeJEP-MIT SSL model."""
        super().__init__()
        if encoder not in ENCODER_CHANNELS:
            msg = (
                f"Unknown encoder '{encoder}'. Expected one of {list(ENCODER_CHANNELS)}"
            )
            raise ValueError(msg)

        if use_dynamic_encoder:
            self.encoder = DynamicMixTransformer(
                encoder=encoder,
                weights=weights,
            )
        else:
            self.encoder = get_encoder(
                name=encoder,
                in_channels=in_channels,
                weights=weights,
            )
        stage_channels = ENCODER_CHANNELS[encoder]
        self.projections = nn.ModuleList(
            [nn.Linear(c, proj_dim) for c in stage_channels]
        )
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and project multi-scale spatial tokens.

        Args:
            x: Input image tensor [B, C, H, W].

        Returns:
            Multi-scale spatial tokens [B, N_total, proj_dim]
            where N_total = N1 + N2 + N3 + N4.

        """
        feats = self.encoder(x)  # list of [B, C_i, H_i, W_i]
        tokens = []
        for feat, proj in zip(feats, self.projections, strict=True):
            t = feat.flatten(2).transpose(1, 2)
            tokens.append(proj(t))
        return torch.cat(tokens, dim=1)
