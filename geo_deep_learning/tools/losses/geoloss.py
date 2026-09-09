"""GeoAware loss: Dice + CE + BF1 + clDice + SDF."""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as fn
from torch import nn

from geo_deep_learning.tools.losses.bf1 import BoundaryLoss
from geo_deep_learning.tools.losses.cldice import CLDiceLoss


class GeoAwareLoss(nn.Module):
    """GeoAware Loss."""

    def __init__(  # noqa: PLR0913
        self,
        classes: list[int],
        alpha: float = 0.2,
        lambda_ce: float = 0.1,
        ce_smooth: float = 0.1,
        ce_weights: list[float] | None = None,
        theta0: int = 3,
        theta: int = 5,
        ignore_index: int | None = 255,
        road_class: int = 3,
        beta_cldice: float = 0.5,
        cldice_iter: int = 10,
        gamma_sdf: float = 0.1,
    ) -> None:
        """Initialize. `classes` is required (BF1 indices, e.g. [4])."""
        if not classes:
            msg = "classes is required for BF1 (e.g. [4] for buildings)"
            raise ValueError(msg)

        super().__init__()
        self.alpha = alpha
        self.lambda_ce = lambda_ce
        self.beta_cldice = beta_cldice
        self.gamma_sdf = gamma_sdf
        self.ignore_index = ignore_index
        self.ce_smooth = ce_smooth

        self.region_loss = smp.losses.DiceLoss(
            mode="multiclass",
            smooth=1e-5,
            from_logits=True,
            ignore_index=self.ignore_index,
        )
        if ce_weights is not None:
            self.register_buffer(
                "ce_weight",
                torch.tensor(ce_weights, dtype=torch.float),
            )
        else:
            self.ce_weight = None
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.ce_weight,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
            label_smoothing=ce_smooth,
        )
        self.boundary_loss = BoundaryLoss(
            classes=classes, theta0=theta0, theta=theta,
        )
        self.cldice_loss = CLDiceLoss(road_class=road_class, num_iter=cldice_iter)

    def forward(  # noqa: PLR0913
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        roads_centerline_weight: torch.Tensor | None = None,
        edt: torch.Tensor | None = None,
        boundary: torch.Tensor | None = None,
        vertices: torch.Tensor | None = None,
        sdf_pred: torch.Tensor | None = None,
        sdf: torch.Tensor | None = None,
        buildings_geo: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scalar loss. Optional maps match tar stems; None skips that term."""
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
            + self.beta_cldice * self.cldice_loss(pred, gt_b, ignore_mask)
            + self.gamma_sdf * self._sdf_loss(
                sdf_pred, sdf, gt_b, ignore_mask, buildings_geo,
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
        """CE; geometry maps add to pixel weight, ce_weight[gt] still applies."""
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
        if self.ce_weight is not None:
            pixel_weight = pixel_weight * self.ce_weight.to(dtype=pred.dtype)[gt_b]

        log_probs = fn.log_softmax(pred, dim=1)
        nll = -log_probs.gather(1, gt_b.unsqueeze(1)).squeeze(1)
        if self.ce_smooth > 0.0:
            nll = (1.0 - self.ce_smooth) * nll + self.ce_smooth * (
                -log_probs.mean(dim=1)
            )
        nll = nll * pixel_weight
        return nll.sum() / (pixel_weight.sum() + 1e-7)

    def _sdf_loss(
        self,
        sdf_pred: torch.Tensor | None,
        sdf: torch.Tensor | None,
        gt_b: torch.Tensor,
        ignore_mask: torch.Tensor | None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE on |sdf| < 1, ignore-masked."""
        if sdf_pred is None:
            return gt_b.new_zeros(())
        if sdf is None:
            return (sdf_pred * 0).sum()

        sdf = sdf.squeeze(1)
        band = (sdf.abs() < 1.0).float()
        if ignore_mask is not None:
            band = band * ignore_mask
        if valid is not None:
            band = band * valid.view(-1, 1, 1).to(dtype=band.dtype)
        diff = (sdf_pred.squeeze(1) - sdf) ** 2 * band
        return diff.sum() / (band.sum() + 1e-7)
