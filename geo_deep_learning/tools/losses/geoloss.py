"""
GeoAware Loss for semantic segmentation.

Combines Dice loss with Boundary F1 Loss specifically targeting sharp geometries.
"""

import segmentation_models_pytorch as smp
import torch
from torch import nn

from geo_deep_learning.tools.losses.bf1 import BoundaryLoss


class GeoAwareLoss(nn.Module):
    """GeoAware Loss."""

    def __init__(
        self,
        classes: list[int] | None = None,
        alpha: float = 0.2,
        theta0: int = 3,
        theta: int = 5,
        ignore_index: int | None = 255,
    ) -> None:
        """
        Initialize GeoAware Loss.

        Args:
            classes: list of integers representing target classes indices
            alpha: weight multiplier for the boundary loss
            theta0: kernel size for the boundary loss
            theta: kernel size for the boundary loss
            ignore_index: index of the ignored class, None for no ignored class

        """
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

        # Region Loss: Handles the holistic geometry over all classes
        self.region_loss = smp.losses.DiceLoss(
            mode="multiclass",
            smooth=1e-5,
            from_logits=True,
            ignore_index=self.ignore_index,
        )

        # Boundary Loss: Acts as a scalpel for specific geometric classes
        self.boundary_loss = BoundaryLoss(
            classes=classes,
            theta0=theta0,
            theta=theta,
        )

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred: (N, C, H, W) raw logits
            gt: (N, H, W) integer class indices

        Returns:
            Total loss

        """
        l_region = self.region_loss(pred, gt)

        gt_b = gt.clone()
        ignore_mask = None

        if self.ignore_index is not None:
            ignore_mask = (gt != self.ignore_index).float()
            gt_b[gt_b == self.ignore_index] = 0

        l_boundary = self.boundary_loss(pred, gt_b, ignore_mask)

        return l_region + (self.alpha * l_boundary)
