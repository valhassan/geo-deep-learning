"""Segmentation IoU metric."""

import torch
from torch import Tensor
from torchmetrics import Metric


class IoU(Metric):
    """IoU metric."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        **kwargs: object,
    ) -> None:
        """Initialize IoU metric."""
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # A confusion matrix of size (num_classes, num_classes)
        # Row = Ground Truth, Column = Prediction
        self.add_state(
            "conf_matrix",
            default=torch.zeros((num_classes, num_classes), dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update IoU metric."""
        # 1. Flatten and mask ignored pixels
        preds = preds.flatten()
        target = target.flatten()
        mask = (
            (target >= 0) & (target < self.num_classes) & (target != self.ignore_index)
        )

        preds = preds[mask]
        target = target[mask]

        # 2. Compute confusion matrix indices
        # index = target * num_classes + prediction
        indices = target * self.num_classes + preds

        # 3. Update state using bincount
        conf_bin = torch.bincount(indices, minlength=self.num_classes**2)
        self.conf_matrix += conf_bin.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Tensor:
        """Compute IoU metric."""
        # TP are the diagonal elements
        tp = torch.diag(self.conf_matrix)
        # FP is column sum minus TP
        fp = torch.sum(self.conf_matrix, dim=0) - tp
        # FN is row sum minus TP
        fn = torch.sum(self.conf_matrix, dim=1) - tp

        union = tp + fp + fn

        # Avoid division by zero and handle absent classes
        # Use NaN for absent classes so they don't impact mIoU averages
        iou = tp.float() / union.float()
        iou[union == 0] = float("nan")

        return iou
