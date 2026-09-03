"""SDF auxiliary head."""

import torch
from torch import nn


class SdfHead(nn.Module):
    """Decoder features to a 1-channel tanh SDF in [-1, 1]."""

    def __init__(self, in_channels: int = 256, mid_channels: int = 64) -> None:
        """Initialize SDF head."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)
