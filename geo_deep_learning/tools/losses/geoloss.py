"""GeoAware loss: Dice + CE + BF1."""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as fn
from torch import nn

from geo_deep_learning.tools.losses.bf1 import BoundaryLoss


class GeoAwareLoss(nn.Module):
    """GeoAware Loss."""

    def __init__(  # noqa: PLR0913
        self,
        classes: list[int],
        alpha: float = 0.2,
        lambda_ce: float = 0.1,
        theta0: int = 3,
        theta: int = 5,
        ignore_index: int | None = 255,
    ) -> None:
        """Initialize. `classes` is required (BF1 indices, e.g. [4])."""
        if not classes:
            msg = "classes is required for BF1 (e.g. [4] for buildings)"
            raise ValueError(msg)

        super().__init__()
        self.alpha = alpha
        self.lambda_ce = lambda_ce
        self.ignore_index = ignore_index

        self.region_loss = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=self.ignore_index,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )
        self.boundary_loss = BoundaryLoss(
            classes=classes, theta0=theta0, theta=theta,
        )

    def forward(  # noqa: PLR0913
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        roads_centerline_weight: torch.Tensor | None = None,
        edt: torch.Tensor | None = None,
        boundary: torch.Tensor | None = None,
        vertices: torch.Tensor | None = None,
        buildings_geo: torch.Tensor | None = None,
        *,
        geo: bool = True,
    ) -> torch.Tensor:
        """Scalar loss. Optional maps skip if None; geo=False is Dice+CE."""
        if not geo:
            return self.region_loss(pred, gt) + self.lambda_ce * self.ce_loss(
                pred, gt,
            )

        if self.ignore_index is not None:
            ignore_mask = (gt != self.ignore_index).float()
            gt_b = gt.clone()
            gt_b[gt_b == self.ignore_index] = 0
        else:
            ignore_mask = None
            gt_b = gt

        return (
            self.region_loss(pred, gt)
            + self.lambda_ce * self._weighted_ce(
                pred, gt, gt_b, ignore_mask, roads_centerline_weight, edt,
            )
            + self.alpha * self.boundary_loss(
                pred, gt_b, ignore_mask, boundary, vertices, buildings_geo,
            )
        )

    def _weighted_ce(  # noqa: PLR0913
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        gt_b: torch.Tensor,
        ignore_mask: torch.Tensor | None,
        roads_centerline_weight: torch.Tensor | None,
        edt: torch.Tensor | None,
    ) -> torch.Tensor:
        """CE; geometry maps add to pixel weight."""
        if roads_centerline_weight is None and edt is None:
            return self.ce_loss(pred, gt)

        pixel_weight = torch.ones(
            pred.shape[0],
            pred.shape[2],
            pred.shape[3],
            device=pred.device,
            dtype=pred.dtype,
        )
        if roads_centerline_weight is not None:
            pixel_weight = pixel_weight + roads_centerline_weight.squeeze(1)
        if edt is not None:
            pixel_weight = pixel_weight + edt.squeeze(1)
        if ignore_mask is not None:
            pixel_weight = pixel_weight * ignore_mask

        log_probs = fn.log_softmax(pred, dim=1)
        nll = -log_probs.gather(1, gt_b.unsqueeze(1)).squeeze(1)
        nll = nll * pixel_weight
        return nll.sum() / (pixel_weight.sum() + 1e-7)
