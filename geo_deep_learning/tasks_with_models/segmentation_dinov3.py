"""
DINOv3 segmentation task.

The batch image stays in 0-1. Sensor mean/std go into the model, which applies
them to the stem and the SAT stats to RGB.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn

from geo_deep_learning.datasets.wds_dataset import GEO_KEYS
from geo_deep_learning.models.segmentation.dinov3 import DINOv3SegmentationModel
from geo_deep_learning.tools.augmentation import RandomD4, RandomPlanckian
from geo_deep_learning.tools.metrics.segmentation_iou import IoU
from geo_deep_learning.tools.utils import load_weights_from_checkpoint, normalization
from geo_deep_learning.tools.visualization import visualize_prediction

logger = logging.getLogger(__name__)


def _adamw_groups(model: nn.Module) -> list[dict[str, Any]]:
    """Decay weights. Skip frozen params, GroupNorm affine, and biases."""
    gn = {
        id(param)
        for module in model.modules()
        if isinstance(module, nn.GroupNorm)
        for param in module.parameters(recurse=False)
    }
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in gn or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, Any]] = [{"params": decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


class SegmentationDINOv3(LightningModule):
    """DINOv3 segmentation model."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        image_size: tuple[int, int],
        num_classes: int,
        max_samples: int,
        loss: Callable,
        use_dora: bool = False,
        pretrained: bool = True,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        load_parts: str | list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.load_parts = load_parts
        self.use_dora = use_dora
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.class_colors = class_colors
        self.max_samples = max_samples
        self.num_classes = num_classes
        self.threshold = 0.5
        self.loss = loss
        metric_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(metric_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou = IoU(num_classes=metric_classes, ignore_index=255)
        self.iou_sensor = nn.ModuleDict()
        self._viz_by_sensor: dict[str, int] = {}
        self.geometric_aug = RandomD4()
        self.radiometric_aug = RandomPlanckian()

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        *,
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Exclude augmentation modules from the checkpoint."""
        state = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        return {
            key: value
            for key, value in state.items()
            if not key.startswith(("geometric_aug.", "radiometric_aug."))
        }

    def configure_model(self) -> None:
        """Build the segmentation model and optionally load a checkpoint."""
        self.model = DINOv3SegmentationModel(
            num_classes=self.num_classes,
            use_dora=self.use_dora,
            pretrained=self.pretrained,
        )
        if self.weights_from_checkpoint_path:
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=self.load_parts,
                map_location=self.device,
            )

    def on_fit_start(self) -> None:
        """Move augmentations onto the training device."""
        self.geometric_aug = self.geometric_aug.to(self.device)
        self.radiometric_aug = self.radiometric_aug.to(self.device)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        """AdamW with decay on weights and none on GroupNorm or bias."""
        optimizer = self.optimizer(_adamw_groups(self.model))
        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(
        self,
        image: Tensor,
        wavelengths: Tensor,
        mean: Tensor,
        std: Tensor,
    ) -> Tensor:
        """0-1 image. Returns logits."""
        return self.model(image, wavelengths, mean, std).out

    def preprocess(
        self,
        x: Tensor,
        mean: list[float] | Tensor,  # noqa: ARG002
        std: list[float] | Tensor,  # noqa: ARG002
        image_min: int = 0,
        image_max: int = 255,
    ) -> Tensor:
        """Scale raw pixels to 0-1. Sensor stats are applied in forward."""
        return normalization(
            x,
            image_min=image_min,
            image_max=image_max,
            norm_min=0.0,
            norm_max=1.0,
        )

    def predict(
        self,
        x: Tensor,
        wavelengths: Tensor,
        mean: Tensor,
        std: Tensor,
        rescale_to: tuple[int, int] | None = None,
    ) -> Tensor:
        """0-1 image in. Returns class probabilities."""
        logits = self(x, wavelengths, mean, std)
        if rescale_to is not None:
            logits = torch.nn.functional.interpolate(
                logits,
                size=rescale_to,
                mode="bilinear",
                align_corners=False,
            )
        if self.num_classes == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def on_after_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Geometric aug, then Planckian on the 0-1 image. No z-score."""
        if self.trainer.training:
            geo = [key for key in GEO_KEYS if key in batch]
            out = self.geometric_aug(
                batch["image"],
                batch["mask"],
                *[batch[key] for key in geo],
            )
            batch["image"], batch["mask"] = out[0], out[1]
            batch.update(dict(zip(geo, out[2:], strict=True)))
            batch["image"] = self.radiometric_aug(
                batch["image"],
                batch["wavelengths"],
            )
        return batch

    @staticmethod
    def _loss_kw(batch: dict[str, Any]) -> dict[str, Tensor | None]:
        return {key: batch.get(key) for key in GEO_KEYS}

    def _logits(self, batch: dict[str, Any]) -> Tensor:
        return self(
            batch["image"],
            batch["wavelengths"],
            batch["mean"],
            batch["std"],
        )

    def _labels(self, logits: Tensor) -> Tensor:
        if self.num_classes == 1:
            return (logits.sigmoid().squeeze(1) > self.threshold).long()
        return logits.softmax(dim=1).argmax(dim=1)

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run one training step."""
        y = batch["mask"].squeeze(1).long()
        logits = self._logits(batch)
        loss = self.loss(logits, y, **self._loss_kw(batch))
        self.log(
            "train_loss",
            loss,
            batch_size=batch["image"].shape[0],
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run one validation step."""
        y = batch["mask"].squeeze(1).long()
        logits = self._logits(batch)
        loss = self.loss(logits, y, **self._loss_kw(batch))
        self.log(
            "val_loss",
            loss,
            batch_size=batch["image"].shape[0],
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )
        return self._labels(logits)

    @staticmethod
    def _platform(batch: dict[str, Any]) -> str | None:
        platform = batch.get("platform")
        if platform is None:
            return None
        if isinstance(platform, (list, tuple)):
            return str(platform[0])
        return str(platform)

    def on_test_start(self) -> None:
        """Build per-sensor IoU from the datamodule."""
        if not self.iou_sensor:
            datasets = getattr(self.trainer.datamodule, "datasets", {}) or {}
            for name, splits in datasets.items():
                if "tst" in splits:
                    self.iou_sensor[name] = IoU(
                        num_classes=self.iou.num_classes,
                        ignore_index=255,
                    )
            self.iou_sensor.to(self.device)
        self._viz_by_sensor = dict.fromkeys(self.iou_sensor, 0)

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run one test step."""
        x = batch["image"]
        y = batch["mask"].squeeze(1).long()
        logits = self._logits(batch)
        loss = self.loss(logits, y, **self._loss_kw(batch))
        y_hat = self._labels(logits)
        self.iou.update(y_hat, y)
        platform = self._platform(batch)
        if platform is not None and platform in self.iou_sensor:
            self.iou_sensor[platform].update(y_hat, y)

        log_kw = {
            "batch_size": x.shape[0],
            "logger": True,
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
        }
        self.log("test_loss", loss, prog_bar=True, rank_zero_only=False, **log_kw)
        if platform is not None:
            self.log(f"test/{platform}/loss", loss, **log_kw)
            used = self._viz_by_sensor.get(platform, 0)
            if used < self.max_samples:
                shown = self._log_visualizations(
                    trainer=self.trainer,
                    batch=batch,
                    outputs=y_hat,
                    max_samples=min(self.max_samples - used, len(x)),
                    artifact_prefix=f"test/{platform}",
                    epoch_suffix=False,
                )
                self._viz_by_sensor[platform] = used + shown

    def _log_iou(self, metric: IoU, prefix: str) -> None:
        per_class = metric.compute()
        metrics = {
            f"{prefix}iou_{label}": iou
            for label, iou in zip(self.labels, per_class, strict=False)
        }
        metrics[f"{prefix}mean_iou"] = torch.nanmean(per_class)
        self.log_dict(metrics, logger=True, sync_dist=True)
        metric.reset()

    def on_test_epoch_end(self) -> None:
        """Log overall and per-sensor IoU."""
        self._log_iou(self.iou, "")
        for name, metric in self.iou_sensor.items():
            self._log_iou(metric, f"test/{name}/")

    def _log_visualizations(  # noqa: PLR0913
        self,
        trainer: Trainer,
        batch: dict[str, Any],
        outputs: Tensor,
        max_samples: int,
        artifact_prefix: str = "val",
        *,
        epoch_suffix: bool = True,
    ) -> int:
        """Log figures. The batch image is 0-1, so scale by 255."""
        if max_samples <= 0:
            return 0
        try:
            image_batch = batch["image"]
            mask_batch = batch["mask"].squeeze(1).long()
            names = batch["image_name"]
            shown = min(max_samples, len(image_batch))
            for i in range(shown):
                image = (image_batch[i].float() * 255).clamp(0, 255).to(torch.uint8)
                fig = visualize_prediction(
                    image=image,
                    mask=mask_batch[i],
                    prediction=outputs[i],
                    sample_name=names[i],
                    num_classes=self.num_classes,
                    class_colors=self.class_colors,
                )
                stem = Path(names[i]).stem
                suffix = f"_epoch_{trainer.current_epoch}" if epoch_suffix else ""
                artifact = f"{artifact_prefix}/{stem}/idx_{i}{suffix}.png"
                trainer.logger.experiment.log_figure(
                    figure=fig,
                    artifact_file=artifact,
                    run_id=trainer.logger.run_id,
                )
        except Exception:
            logger.exception("Error in DINOv3 visualization")
            return 0
        else:
            return shown


class _ExportWrapper(nn.Module):
    """Raw image to logits. Scales to 0-1 and passes sensor stats through."""

    def __init__(self, model: nn.Module) -> None:
        """Store the segmentation model."""
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        mean: Tensor,
        std: Tensor,
        wavelengths: Tensor,
    ) -> Tensor:
        """Export forward."""
        channels = x.shape[1]
        x = normalization(x)
        return self.model(
            x,
            wavelengths,
            mean.view(channels, 1, 1),
            std.view(channels, 1, 1),
        ).out


def export_model(
    checkpoint_path: str,
    output_path: str,
    metadata_path: str | None = None,
) -> None:
    """Export (x, mean, std, wavelengths) -> logits."""
    device = "cuda"
    task = SegmentationDINOv3.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
        weights_from_checkpoint_path=None,
        pretrained=False,
    )
    model = task.model.eval().cuda()
    wrapper = _ExportWrapper(model).cuda()
    batch_dim = torch.export.Dim("batch", min=1, max=32)
    channels_dim = torch.export.Dim("channels", min=3, max=8)
    channels = 4
    height, width = task.image_size
    x = torch.randn(2, channels, height, width, device=device)
    mean = torch.randn(channels, device=device)
    std = torch.randn(channels, device=device).abs() + 1e-5
    wavelengths = torch.randn(channels, device=device)
    extra = None
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        extra = {"metadata.json": json.dumps(metadata)}
    exported = torch.export.export(
        wrapper,
        args=(x, mean, std, wavelengths),
        dynamic_shapes={
            "x": {0: batch_dim, 1: channels_dim},
            "mean": {0: channels_dim},
            "std": {0: channels_dim},
            "wavelengths": {0: channels_dim},
        },
        strict=False,
    )
    torch.export.save(exported, output_path, extra_files=extra)
    logger.info("Exported to %s", output_path)
