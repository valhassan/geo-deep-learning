"""Boundary Loss for target classes."""

import torch
import torch.nn.functional as fn
from torch import nn


class BoundaryLoss(nn.Module):
    """Boundary Loss for target classes."""

    def __init__(
        self, classes: list[int] | None = None, theta0: int = 3, theta: int = 5,
    ) -> None:
        """Initialize Boundary Loss."""
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.classes = classes

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred: (N, C, H, W) raw logits
            gt: (N, H, W) integer class indices

        """
        n, c, _, _ = pred.shape
        pred = torch.softmax(pred, dim=1)

        one_hot_gt = fn.one_hot(gt, num_classes=c).float()
        one_hot_gt = one_hot_gt.permute(0, 3, 1, 2)

        if self.classes is not None:
            pred = pred[:, self.classes, :, :]
            one_hot_gt = one_hot_gt[:, self.classes, :, :]
            c = len(self.classes)  # Update channel dimension

        gt_b = fn.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        gt_b -= 1 - one_hot_gt

        pred_b = fn.max_pool2d(
            1 - pred,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        pred_b -= 1 - pred

        gt_b_ext = fn.max_pool2d(
            gt_b,
            kernel_size=self.theta,
            stride=1,
            padding=(self.theta - 1) // 2,
        )

        pred_b_ext = fn.max_pool2d(
            pred_b,
            kernel_size=self.theta,
            stride=1,
            padding=(self.theta - 1) // 2,
        )

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall, and Boundary F1 Score
        precision = torch.sum(pred_b * gt_b_ext, dim=2) / (
            torch.sum(pred_b, dim=2) + 1e-7
        )
        recall = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        bf1 = 2 * precision * recall / (precision + recall + 1e-7)

        return torch.mean(1 - bf1)
