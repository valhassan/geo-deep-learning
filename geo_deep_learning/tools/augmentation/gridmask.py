"""Fast GridMask augmentation."""

import torch
from torch import nn


class FastGridMask(nn.Module):
    """Fully vectorized, GPU-accelerated GridMask."""

    def __init__(
        self, grid_size: int = 64, mask_ratio: float = 0.5, p: float = 0.5,
    ) -> None:
        """Initialize Fast GridMask."""
        super().__init__()
        self.grid_size = grid_size
        self.mask_ratio = mask_ratio
        self.p = p
        self.drop_size = int(grid_size * mask_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self.training or torch.rand(1).item() > self.p:
            return x

        b, _, h, w = x.shape
        device = x.device

        # Generate 1D coordinates
        y_coords = torch.arange(h, device=device)
        x_coords = torch.arange(w, device=device)

        # Generate random offsets for dynamic grid placement
        offset_y = torch.randint(0, self.grid_size, (1,), device=device)
        offset_x = torch.randint(0, self.grid_size, (1,), device=device)

        # Boolean masks: True if coordinate is outside the "drop zone"
        y_keep = ((y_coords + offset_y) % self.grid_size) >= self.drop_size
        x_keep = ((x_coords + offset_x) % self.grid_size) >= self.drop_size

        # Broadcast to 2D mask: Keep pixel if it's safe in Y OR safe in X
        # This forms the structural grid pattern
        mask = y_keep.view(-1, 1) | x_keep.view(1, -1)

        # Expand to match tensor dimensions and multiply
        mask = mask.view(1, 1, h, w).expand(b, -1, -1, -1).to(x.dtype)

        return x * mask
