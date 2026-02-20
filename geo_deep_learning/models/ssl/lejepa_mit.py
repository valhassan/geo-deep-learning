"""LeJEPA MixTransformer SSL model."""

import torch
from torch import nn

from geo_deep_learning.models.encoders.mix_transformer import (
    DynamicMixTransformer,
    get_encoder,
)


class LeJEPAMixTransformer(nn.Module):
    """LeJEPA MixTransformer SSL model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
        stages: list[int] | None = None,
        projection_head_dim: int = 16,
        *,
        use_dynamic_encoder: bool = False,
    ) -> None:
        """Initialize LeJEP-MIT SSL model."""
        super().__init__()
        if stages is None:
            stages = (4,)
        stages = tuple(int(s) for s in stages)
        last_stage = 4
        if any(s < 1 or s > last_stage for s in stages):
            msg = f"stages must be in [1..4], got {stages}"
            raise ValueError(msg)
        if len(set(stages)) != len(stages):
            msg = f"stages must be unique, got {stages}"
            raise ValueError(msg)
        self.stages = stages
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
        out_channels = list(self.encoder.out_channels)[2:]
        encoder_dim = sum(out_channels[s - 1] for s in stages)
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, projection_head_dim),
            nn.BatchNorm1d(projection_head_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected embedding: [B, projection_head_dim]."""
        feats = self.encoder(x)  # list of [B, C_i, H_i, W_i]
        pooled = [feats[s - 1].mean(dim=(2, 3)) for s in self.stages]
        h = torch.cat(pooled, dim=1)
        return self.projection_head(h)
