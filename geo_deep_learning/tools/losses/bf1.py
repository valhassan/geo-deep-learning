"""Boundary F1 Loss for buildings."""

import torch
import torch.nn.functional as fn
from torch import nn


class BoundaryLoss(nn.Module):
    """Boundary F1 Loss."""

    def __init__(
        self,
        classes: list[int],
        theta0: int = 3,
        theta: int = 5,
    ) -> None:
        """
        Initialise BoundaryLoss.

        Args:
            classes:  Class indices to compute boundary loss over. Required.
            theta0:   Kernel size for boundary extraction (pred and gt).
            theta:    Kernel size for boundary dilation (tolerance window).

        """
        if not classes:
            msg = "classes is required for BF1 (e.g. [4] for buildings)"
            raise ValueError(msg)
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.classes = classes

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred:         (N, C, H, W) raw logits.
            gt:           (N, H, W) integer class indices.
                          Must not contain ignore_index — zero it out before
                          calling (handled in GeoAwareLoss.forward).
            ignore_mask:  (N, H, W) float, 1 = valid pixel, 0 = ignored.

        Returns:
            Scalar boundary loss: mean(1 - BF1) over classes and batch.

        """
        n, c, _, _ = pred.shape
        pred_soft = torch.softmax(pred, dim=1)

        one_hot_gt = fn.one_hot(gt, num_classes=c).float().permute(0, 3, 1, 2)

        pred_soft = pred_soft[:, self.classes]
        one_hot_gt = one_hot_gt[:, self.classes]
        c = len(self.classes)

        pred_b = fn.max_pool2d(
            1 - pred_soft,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        pred_b = pred_b - (1 - pred_soft)

        gt_b = fn.max_pool2d(
            1 - one_hot_gt,
            kernel_size=self.theta0,
            stride=1,
            padding=(self.theta0 - 1) // 2,
        )
        gt_b = gt_b - (1 - one_hot_gt)

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

        if ignore_mask is not None:
            mask = ignore_mask.unsqueeze(1)
            gt_b = gt_b * mask
            pred_b = pred_b * mask
            gt_b_ext = gt_b_ext * mask
            pred_b_ext = pred_b_ext * mask

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        precision = torch.sum(pred_b * gt_b_ext, dim=2) / (
            torch.sum(pred_b, dim=2) + 1e-7
        )
        recall = torch.sum(pred_b_ext * gt_b, dim=2) / (
            torch.sum(gt_b, dim=2) + 1e-7
        )
        bf1 = 2 * precision * recall / (precision + recall + 1e-7)

        return torch.mean(1 - bf1)
