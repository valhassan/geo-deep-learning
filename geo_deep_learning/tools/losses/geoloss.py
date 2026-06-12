"""
GeoAware Loss for semantic segmentation.

Combines Dice, CE and Boundary F1 (targeting sharp geometries) losses.

Loss stack by class
-------------------
Background (0):  Dice + CE
Forest     (1):  Dice + CE
Hydro      (2):  Dice + CE
Roads      (3):  Dice + CE (centerline-weighted) + clDice
Buildings  (4):  Dice + CE (EDT-weighted) + BF1 (vector-enhanced) + SDF aux
"""

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
        classes: list[int] | None = None,
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
        # buildings
        building_class: int = 4,
        gamma_sdf: float = 0.1,
    ) -> None:
        """
        Initialize GeoAware Loss.

        Args:
            classes:        Class indices for boundary loss. None = all classes.
            alpha:          Weight for boundary (BF1) loss.
            lambda_ce:      Weight for cross-entropy loss.
            ce_smooth:      Label smoothing for cross-entropy.
            ce_weights:     Per-class weights for cross-entropy.
            theta0:         BF1 kernel size for boundary extraction.
            theta:          BF1 kernel size for boundary dilation.
            ignore_index:   Index of ignored pixels. None disables masking.
            road_class:     Class index for roads in the multiclass output.
            beta_cldice:    Weight for clDice loss.
            cldice_iter:    Soft skeleton erosion iterations.
                            10 covers road widths up to ~30px.
                            Increase to 15 for 15cm aerial data.
            building_class: Class index for buildings (SDF mask).
            gamma_sdf:      Weight for SDF auxiliary loss.

        """
        super().__init__()
        self.alpha = alpha
        self.lambda_ce = lambda_ce
        self.beta_cldice = beta_cldice
        self.gamma_sdf = gamma_sdf
        self.building_class = building_class
        self.ignore_index = ignore_index
        self.ce_smooth = ce_smooth

        # Region loss: all classes
        self.region_loss = smp.losses.DiceLoss(
            mode="multiclass",
            smooth=1e-5,
            from_logits=True,
            ignore_index=self.ignore_index,
        )

        # Pixel loss: all classes, optionally class-weighted.
        # ce_weights registered as a buffer so it moves with the model
        # when .to(device) / .cuda() is called — avoids device mismatch.
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

        # Boundary loss: buildings only
        self.boundary_loss = BoundaryLoss(
            classes=classes,
            theta0=theta0,
            theta=theta,
        )

        # Topology loss: roads only
        self.cldice_loss = CLDiceLoss(
            road_class=road_class,
            num_iter=cldice_iter,
        )

    def forward(  # noqa: PLR0913
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        # roads
        centerline_weight: torch.Tensor | None = None,
        # buildings
        edt_weight: torch.Tensor | None = None,
        vector_boundary: torch.Tensor | None = None,
        vertex_heatmap: torch.Tensor | None = None,
        sdf_pred: torch.Tensor | None = None,
        sdf_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred:              (N, C, H, W) raw logits.
            gt:                (N, H, W) integer class indices.

            centerline_weight: (N, 1, H, W) float32 [0, 1].
                               Intra-polygon EDT for roads — maximum at the
                               centerline, zero at polygon boundary.
                               Precomputed from vector geometry at dataset
                               generation time (roads_centerline_weight.npy,
                               divided by 255 at load time).
                               Falls back to uniform CE when None.

            edt_weight:        (N, 1, H, W) float32 [0, 1].
                               Dual-distance inter-instance gap weight for
                               buildings (buildings_edt.npy, divided by 255
                               at load time). Falls back to uniform CE when None.

            vector_boundary:   (N, 1, H, W) float32 [0, 1].
                               Sub-pixel polygon edge map for buildings
                               (buildings_boundary.npy). Passed to BF1.

            vertex_heatmap:    (N, 1, H, W) float32 [0, 1].
                               Corner Gaussian blobs for buildings
                               (buildings_vertices.npy). Passed to BF1.

            sdf_pred:          (N, 1, H, W) float32. Output of SDF auxiliary
                               head. Skipped when None.

            sdf_target:        (N, 1, H, W) float32. Signed metric distance
                               to nearest building boundary (buildings_sdf.npy,
                               loaded as float16, cast to float32 before call).
                               Skipped when None.

        Returns:
            Scalar total loss.

        """
        # Ignore mask and safe gt — only pay the clone cost when needed.
        if self.ignore_index is not None:
            ignore_mask = (gt != self.ignore_index).float()  # (N, H, W)
            gt_b = gt.clone()
            gt_b[gt_b == self.ignore_index] = 0
        else:
            ignore_mask = None
            gt_b = gt

        l_region = self.region_loss(pred, gt)
        l_ce = self._weighted_ce(
            pred, gt, gt_b, ignore_mask, centerline_weight, edt_weight,
        )
        l_boundary = self.boundary_loss(
            pred, gt_b, ignore_mask, vector_boundary, vertex_heatmap,
        )
        l_cldice = self.cldice_loss(pred, gt_b, ignore_mask)
        l_sdf = self._sdf_loss(sdf_pred, sdf_target, gt_b, ignore_mask)

        return (
            l_region
            + self.lambda_ce * l_ce
            + self.alpha * l_boundary
            + self.beta_cldice * l_cldice
            + self.gamma_sdf * l_sdf
        )

    def _weighted_ce(  # noqa: PLR0913
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        gt_b: torch.Tensor,
        ignore_mask: torch.Tensor | None,
        centerline_weight: torch.Tensor | None,
        edt_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        CE loss, optionally pixel-weighted by precomputed geometry targets.

        When both weights are None this is identical to self.ce_loss(pred, gt)
        including label smoothing — no behavioural change for unweighted samples.

        When weights are present:
          - pixel_weight is built by adding geometry weights (each in [0,1])
            to a scalar base of 1.0, shifting weighted pixels to [1, 2] while
            all other pixels retain weight 1.0.
          - Label smoothing is applied manually to match self.ce_loss behaviour.
          - Loss is normalised by sum-of-weights (not pixel count) so the
            scale stays stable regardless of road/building pixel density.
          - A single log_softmax covers both geometry-weighted terms,
            avoiding a second full pass over (N, C, H, W).

        Args:
            pred:             (N, C, H, W) raw logits.
            gt:               Original gt with ignore values intact — used only
                              for the unweighted fallback path via self.ce_loss.
            gt_b:             gt with ignore pixels zeroed — used in weighted path.
            ignore_mask:      (N, H, W) float, 1 = valid, 0 = ignored.
            centerline_weight:(N, 1, H, W) float32 [0, 1] or None.
            edt_weight:       (N, 1, H, W) float32 [0, 1] or None.

        """
        if centerline_weight is None and edt_weight is None:
            return self.ce_loss(pred, gt)

        # Build pixel weight map — add geometry weights to scalar 1.0 baseline.
        # squeeze(1): (N, 1, H, W) -> (N, H, W), no copy.
        pixel_weight = torch.ones(
            pred.shape[0],
            pred.shape[2],
            pred.shape[3],
            device=pred.device,
            dtype=pred.dtype,
        )
        if centerline_weight is not None:
            pixel_weight = pixel_weight + centerline_weight.squeeze(1)
        if edt_weight is not None:
            pixel_weight = pixel_weight + edt_weight.squeeze(1)

        if ignore_mask is not None:
            pixel_weight = pixel_weight * ignore_mask

        # Single log_softmax pass for the weighted NLL.
        log_probs = fn.log_softmax(pred, dim=1)  # (N, C, H, W)
        nll = -log_probs.gather(1, gt_b.unsqueeze(1)).squeeze(1)  # (N, H, W)

        # Label smoothing: match nn.CrossEntropyLoss(label_smoothing=ce_smooth).
        # Smoothed loss = (1 - s) * nll + s * mean(-log_probs over all classes).
        if self.ce_smooth > 0.0:
            smooth_loss = -log_probs.mean(dim=1)  # (N, H, W)
            nll = (1.0 - self.ce_smooth) * nll + self.ce_smooth * smooth_loss

        nll = nll * pixel_weight
        return nll.sum() / (pixel_weight.sum() + 1e-7)

    def _sdf_loss(
        self,
        sdf_pred: torch.Tensor | None,
        sdf_target: torch.Tensor | None,
        gt_b: torch.Tensor,
        ignore_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        MSE loss on SDF auxiliary head, buildings pixels only.

        Masked to building pixels so gradient only flows where SDF
        supervision is meaningful. Returns zero when either tensor
        is absent — no-op when SDF head is not attached.

        Args:
            sdf_pred:    (N, 1, H, W) float32 from auxiliary head.
            sdf_target:  (N, 1, H, W) float32 from buildings_sdf.npy.
            gt_b:        (N, H, W) integer labels, ignore pixels zeroed.
            ignore_mask: (N, H, W) float, 1 = valid, 0 = ignored.

        Returns:
            Scalar mean MSE over building pixels.

        """
        if sdf_pred is None or sdf_target is None:
            return gt_b.new_zeros(())

        building_mask = (gt_b == self.building_class).float()  # (N, H, W)
        if ignore_mask is not None:
            building_mask = building_mask * ignore_mask

        diff = (sdf_pred.squeeze(1) - sdf_target.squeeze(1)) ** 2
        diff = diff * building_mask
        return diff.sum() / (building_mask.sum() + 1e-7)
