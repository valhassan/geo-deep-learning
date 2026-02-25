"""SetCriterion for Mask2Former."""

import torch
import torch.nn.functional as fn
from torch import Tensor, nn

from geo_deep_learning.models.decoders.mask2former.config import CONFIG

from .loss_utils import (
    calculate_uncertainty,
    dice_loss_jit,
    get_src_permutation_idx,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    sigmoid_ce_loss_jit,
)
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """
    Mask2Former loss computation.

    This class computes the loss for Mask2Former, which consists of:
        - Classification loss (cross entropy on matched queries)
        - Mask loss (BCE loss on matched masks)
        - Dice loss (dice loss on matched masks)

    The process:
        1. Use HungarianMatcher for optimal prediction-target assignment
        2. Compute losses only on matched prediction-target pairs
        3. Aggregate losses with specified weights
        4. Include auxiliary losses from intermediate decoder layers

    Args:
        num_classes: Number of object categories (excluding background)
        matcher: HungarianMatcher module for optimal assignment
        weight_dict: Dict with loss names as keys and their weights as values.
                    Expected keys: "loss_ce", "loss_mask", "loss_dice"
        eos_coef: Relative classification weight applied to the no-object category
        losses: List of losses to compute. Can include: "labels", "masks"
        num_points: Number of points to sample for mask loss computation (efficiency)
        oversample_ratio: Oversampling ratio for point sampling
        importance_sample_ratio: Ratio of points sampled via importance sampling

    """

    def __init__(  # noqa: PLR0913
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float,
        losses: list[str],
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        class_weights: list[float] | None = None,
    ) -> None:
        """Initialize SetCriterion."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses or ["labels", "masks"]
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        # Weight for the no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        if class_weights is not None:
            if len(class_weights) != self.num_classes:
                msg = f"Expected {self.num_classes} weights, got {len(class_weights)}"
                raise ValueError(msg)
            empty_weight[: self.num_classes] = torch.tensor(class_weights)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight, persistent=False)

    def __repr__(self) -> str:
        """Representation of the criterion."""
        head = "Criterion " + self.__class__.__name__
        body = [
            f"num_classes: {self.num_classes}",
            f"eos_coef: {self.eos_coef}",
            f"losses: {self.losses}",
            f"weight_dict: {self.weight_dict}",
            f"num_points: {self.num_points}",
        ]
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: float,  # noqa: ARG002
    ) -> dict[str, Tensor]:
        """Cross-entropy over class logits."""
        if "pred_logits" not in outputs:
            msg = "pred_logits not in outputs"
            raise KeyError(msg)
        src_logits = outputs["pred_logits"]  # [B, Q, C+1]
        device = src_logits.device

        # If there are no target labels at all, return a zero scalar
        if sum(len(t["labels"]) for t in targets) == 0:
            return {"loss_ce": torch.zeros((), device=device)}

        # Matched indices (can be empty)
        idx = get_src_permutation_idx(indices)
        if isinstance(idx, tuple):
            if idx[0].numel() == 0 or idx[1].numel() == 0:
                return {"loss_ce": torch.zeros((), device=device)}
        elif idx.numel() == 0:
            return {"loss_ce": torch.zeros((), device=device)}
        # Gather target classes for matched predictions
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices, strict=False)],
        )  # [num_matched]

        # Full target (most queries are "no-object" = num_classes)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=device,
        )
        target_classes[idx] = target_classes_o

        # CE expects [B, C+1, Q]
        loss_ce = fn.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_masks(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: Tensor | float,  # 0-dim tensor preferred
    ) -> dict[str, Tensor]:
        """Mask losses (BCE + dice)."""
        if "pred_masks" not in outputs:
            msg = "pred_masks not in outputs"
            raise KeyError(msg)

        src_masks_all = outputs["pred_masks"]  # [B, Q, H, W]
        device = src_masks_all.device

        # If there are no GT masks at all, return zeros
        if sum(t["masks"].shape[0] for t in targets) == 0:
            zeros = torch.zeros((), device=device)
            return {"loss_mask": zeros, "loss_dice": zeros}

        # Matched indices
        src_idx = get_src_permutation_idx(indices)
        if isinstance(src_idx, tuple):
            if src_idx[0].numel() == 0 or src_idx[1].numel() == 0:
                zeros = torch.zeros((), device=device)
                return {"loss_mask": zeros, "loss_dice": zeros}
        elif src_idx.numel() == 0:
            zeros = torch.zeros((), device=device)
            return {"loss_mask": zeros, "loss_dice": zeros}

        # Select matched predictions
        src_masks = src_masks_all[src_idx]  # [num_matched, H, W]
        if src_masks.numel() == 0:
            zeros = torch.zeros((), device=device)
            return {"loss_mask": zeros, "loss_dice": zeros}

        # Concatenate matched target masks
        target_masks = torch.cat(
            [t["masks"][i] for t, (_, i) in zip(targets, indices, strict=False)],
            dim=0,
        ).to(src_masks)  # [num_matched, Ht, Wt]

        # Align spatial sizes if needed
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            src_masks = fn.interpolate(
                src_masks.unsqueeze(1),
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Add channel dimension: [num_matched, 1, H, W]
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        # Point sampling coords (no grad)
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )  # [num_matched, num_points, 2] in [0, 1] x [0, 1]

        # Sample values at points
        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)  # [num_matched, num_points]
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)  # [num_matched, num_points]

        # Compute BCE loss
        with torch.autocast(device_type="cuda", enabled=False):
            point_logits = point_logits.float()
            point_labels = point_labels.float()
            if not torch.is_tensor(num_masks):
                num_masks_tensor = torch.as_tensor(
                    [num_masks],
                    device=device,
                    dtype=torch.float,
                )
            else:
                num_masks_tensor = num_masks
            loss_mask = sigmoid_ce_loss_jit(
                point_logits,
                point_labels,
                num_masks_tensor,
            )

        # Dice on sampled points
        loss_dice = dice_loss_jit(
            point_logits,
            point_labels,
            num_masks_tensor,
        )

        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def get_loss(
        self,
        loss: str,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: float,
    ) -> dict[str, Tensor]:
        """
        Dispatch to the appropriate loss function.

        Args:
            loss: Loss name ("labels" or "masks")
            outputs: Model outputs
            targets: Ground truth targets
            indices: Matched indices from Hungarian matcher
            num_masks: Number of masks for normalization

        Returns:
            Dict with computed losses

        """
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            msg = f"Loss {loss} not supported"
            raise ValueError(msg)
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """
        Compute the losses.

        Args:
            outputs: Dict with keys:
                - "pred_logits": Tensor of shape [B, Q, C+1] (class predictions)
                - "pred_masks": Tensor of shape [B, Q, H, W] (mask predictions)
                - "aux_outputs": Optional list of intermediate prediction dicts
            targets: List of dicts (one per batch element) with keys:
                - "labels": Tensor of shape [N] (ground truth class indices)
                - "masks": Tensor of shape [N, H, W] (ground truth binary masks)

        Returns:
            Dict of losses with keys like "loss_ce", "loss_mask", "loss_dice"

        """
        # Compute losses for main outputs (exclude auxiliary outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # Retrieve the matching between the outputs of the last layer and the targets
        with torch.no_grad():
            indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target masks for normalization
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_masks, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
            num_masks = torch.clamp(num_masks / max(world_size, 1), min=1.0)
        else:
            num_masks = torch.clamp(num_masks, min=1.0)
        # Compute all requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs_without_aux, targets, indices, num_masks),
            )
        # Compute auxiliary losses (if present)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


def build_criterion(
    num_classes: int,
    class_weights: list[float] | None = None,
) -> SetCriterion:
    """
    Build a SetCriterion with default settings.

    Convenience function for creating a criterion.

    Args:
        num_classes: Number of object categories (excluding background)
        class_weights: Optional list of weights for each class.
        weight_dict: Loss weights. If None, uses defaults.
        eos_coef: Weight for no-object class
        num_points: Number of points to sample for losses

    Returns:
        Configured SetCriterion instance

    Example:
        >>> criterion = build_criterion(
        ...     num_classes=4,
        ...     weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        ... )

    """
    # Build matcher
    matcher = HungarianMatcher(
        cost_class=CONFIG.class_weight,
        cost_mask=CONFIG.mask_weight,
        cost_dice=CONFIG.dice_weight,
        num_points=CONFIG.num_points,
    )
    weight_dict = {
        "loss_ce": CONFIG.class_weight,
        "loss_mask": CONFIG.mask_weight,
        "loss_dice": CONFIG.dice_weight,
    }
    if CONFIG.deep_supervision:
        dec_layers = CONFIG.decoder_layers
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Build criterion
    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=CONFIG.no_object_weight,
        losses=["labels", "masks"],
        num_points=CONFIG.num_points,
        oversample_ratio=CONFIG.oversample_ratio,
        importance_sample_ratio=CONFIG.importance_sample_ratio,
        class_weights=class_weights,
    )
