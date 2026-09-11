"""Segmentation DOFA model."""

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor, nn

from geo_deep_learning.datasets.wds_dataset import GEO_KEYS
from geo_deep_learning.models.segmentation.dofa import DOFASegmentationModel
from geo_deep_learning.tools.metrics.segmentation_iou import IoU
from geo_deep_learning.tools.utils import (
    denormalization,
    load_weights_from_checkpoint,
    normalization,
    standardization,
)
from geo_deep_learning.tools.visualization import visualize_prediction

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class SegmentationDOFA(LightningModule):
    """Segmentation DOFA model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str,
        *,
        pretrained: bool,
        image_size: tuple[int, int],
        num_classes: int,
        max_samples: int,
        loss: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        freeze_layers: list[str] | None = None,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        load_parts: str | list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.pretrained = pretrained
        self.image_size = image_size
        self.freeze_layers = freeze_layers
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.load_parts = load_parts
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.class_colors = class_colors
        self.max_samples = max_samples
        self.num_classes = num_classes
        self.threshold = 0.5
        self.loss = loss
        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(num_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou = IoU(num_classes=num_classes, ignore_index=255)
        self.iou_sensor = nn.ModuleDict()
        self._viz_by_sensor: dict[str, int] = {}

        self.geometric_aug = self._geometric_aug()

    def _geometric_aug(self) -> AugmentationSequential:
        return AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=False,
                keepdim=True,
            ),
            data_keys=["image", "mask"],
            random_apply=1,
        )

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        *,
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Exclude augmentation modules from checkpoint."""
        state = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        return {
            k: v
            for k, v in state.items()
            if not k.startswith(("geometric_aug.", "radiometric_aug."))
        }

    def configure_model(self) -> None:
        """Configure model."""
        self.model = DOFASegmentationModel(
            encoder=self.encoder,
            image_size=self.image_size,
            freeze_layers=self.freeze_layers,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
        )
        if self.weights_from_checkpoint_path:
            map_location = self.device
            load_parts = self.load_parts
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=load_parts,
                map_location=map_location,
            )

    def on_fit_start(self) -> None:
        """On fit start."""
        self.geometric_aug = self.geometric_aug.to(self.device)

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image: Tensor, wavelengths: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image, wavelengths)

    def preprocess(
        self,
        x: torch.Tensor,
        mean: list[float] | torch.Tensor,
        std: list[float] | torch.Tensor,
        image_min: int = 0,
        image_max: int = 255,
    ) -> torch.Tensor:
        """
        Apply normalization and standardization for inference.

        Args:
            x: Raw input tensor (B, C, H, W), values in [0, 255] range
            mean: Mean values for standardization (per channel)
            std: Std values for standardization (per channel)
            image_min: Minimum value for normalization
            image_max: Maximum value for normalization
        Returns:
            Preprocessed tensor ready for model forward pass

        """
        x = normalization(
            x,
            image_min=image_min,
            image_max=image_max,
            norm_min=0.0,
            norm_max=1.0,
        )
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32, device=x.device)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32, device=x.device)
        if mean.dim() == 1:
            mean = mean.view(-1, 1, 1)
        if std.dim() == 1:
            std = std.view(-1, 1, 1)
        return standardization(x, mean, std)

    def predict(
        self,
        x: torch.Tensor,
        wavelengths: Tensor,
        rescale_to: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Inference forward pass (expects preprocessed input).

        Args:
            x: Preprocessed input tensor (B, C, H, W)
            wavelengths: Wavelength tensor (B, C) or (C,) for band wavelengths
            rescale_to: Optional output size to rescale predictions to (H, W)

        Returns:
            Predictions (B, C, H, W) - probabilities for each class

        """
        outputs = self(x, wavelengths)
        logits = outputs.out

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
        """On after batch transfer."""
        if self.trainer.training:
            geo = [k for k in GEO_KEYS if k in batch]
            out = self.geometric_aug(
                batch["image"],
                batch["mask"],
                *[batch[k] for k in geo],
                data_keys=["image", "mask", *["image"] * len(geo)],
            )
            batch["image"], batch["mask"] = out[0], out[1]
            batch.update(dict(zip(geo, out[2:], strict=True)))
        batch["image"] = standardization(batch["image"], batch["mean"], batch["std"])
        return batch

    @staticmethod
    def _loss_kw(batch: dict[str, Any]) -> dict[str, Tensor | None]:
        return {k: batch.get(k) for k in (*GEO_KEYS, "buildings_geo")}

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x = batch["image"]
        y = batch["mask"]
        wv = batch["wavelengths"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x, wv)
        loss_kw = self._loss_kw(batch)

        loss_main = self.loss(outputs.out, y, **loss_kw)
        loss_aux = self.loss(outputs.aux["aux"], y, geo=False)
        total_loss = loss_main + 0.4 * loss_aux

        self.log(
            "train_loss",
            total_loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )

        return total_loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run validation step."""
        x = batch["image"]
        y = batch["mask"]
        wv = batch["wavelengths"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x, wv)
        loss = self.loss(outputs.out, y, **self._loss_kw(batch))
        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )
        if self.num_classes == 1:
            y_hat = (outputs.out.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = outputs.out.softmax(dim=1).argmax(dim=1)

        return y_hat

    @staticmethod
    def _platform(batch: dict[str, Any]) -> str | None:
        p = batch.get("platform")
        if p is None:
            return None
        if isinstance(p, (list, tuple)):
            return str(p[0])
        return str(p)

    def on_test_start(self) -> None:
        """Build per-sensor IoU from the datamodule (stable DDP keys)."""
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
        """Run test step."""
        x = batch["image"]
        y = batch["mask"]
        wv = batch["wavelengths"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x, wv)
        loss = self.loss(outputs.out, y, **self._loss_kw(batch))

        if self.num_classes == 1:
            y_hat = (outputs.out.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = outputs.out.softmax(dim=1).argmax(dim=1)

        self.iou.update(y_hat, y)
        platform = self._platform(batch)
        if platform is not None and platform in self.iou_sensor:
            self.iou_sensor[platform].update(y_hat, y)

        log_kw = {
            "batch_size": batch_size,
            "logger": True,
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
        }
        self.log("test_loss", loss, prog_bar=True, rank_zero_only=False, **log_kw)
        if platform is not None:
            self.log(f"test/{platform}/loss", loss, **log_kw)

        if platform is not None:
            used = self._viz_by_sensor.get(platform, 0)
            if used < self.max_samples:
                n = self._log_visualizations(
                    trainer=self.trainer,
                    batch=batch,
                    outputs=y_hat,
                    max_samples=min(self.max_samples - used, len(x)),
                    artifact_prefix=f"test/{platform}",
                    epoch_suffix=False,
                )
                self._viz_by_sensor[platform] = used + (n or 0)

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
        """Compute and log IoU metrics at end of test epoch."""
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
    ) -> None:
        """
        DOFA-specific log visualizations.

        Args:
            trainer: Lightning trainer
            batch: Batch data containing image, mask, image_name, mean, std
            outputs: Model predictions
            max_samples: Maximum number of samples to visualize
            artifact_prefix: Prefix for artifact path ("test" or "val")
            epoch_suffix: Whether to add epoch info to artifact filename

        Returns:
            Number of samples actually visualized

        """
        if batch is None or outputs is None:
            return 0

        try:
            logger.info("Logging visualizations")
            image_batch = batch["image"]
            mask_batch = batch["mask"].squeeze(1).long()
            batch_image_name = batch["image_name"]
            mean_batch = batch["mean"]
            std_batch = batch["std"]
            num_samples = min(max_samples, len(image_batch))
            for i in range(num_samples):
                image = image_batch[i]
                image_name = batch_image_name[i]
                mean = mean_batch[i]
                std = std_batch[i]
                image = denormalization(image, mean=mean, std=std)

                fig = visualize_prediction(
                    image=image,
                    mask=mask_batch[i],
                    prediction=outputs[i],
                    sample_name=image_name,
                    num_classes=self.num_classes,
                    class_colors=self.class_colors,
                )
                base_path = f"{artifact_prefix}/{Path(image_name).stem}"
                if epoch_suffix and trainer is not None:
                    artifact_file = (
                        f"{base_path}/idx_{i}_epoch_{trainer.current_epoch}.png"
                    )
                else:
                    artifact_file = f"{base_path}/idx_{i}.png"
                trainer.logger.experiment.log_figure(
                    figure=fig,
                    artifact_file=artifact_file,
                    run_id=trainer.logger.run_id,
                )
        except Exception:
            logger.exception("Error in DOFA visualization")
        else:
            return num_samples


class _ExportWrapper(nn.Module):
    """
    DOFA export: single forward, norm + standardize then model.

    Inputs: x (B,C,H,W), mean (C,), std (C,), wavelengths (C,).
    Returns logits (no TTA).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        wavelengths: Tensor,
    ) -> Tensor:
        c = x.shape[1]
        mean = mean.view(c, 1, 1)
        std = std.view(c, 1, 1)
        x = standardization(normalization(x), mean, std)
        return self.model(x, wavelengths).out


def export_model(checkpoint_path: str, output_path: str) -> None:
    """Export DOFA: (x, mean, std, wavelengths) -> logits, single forward."""
    device = "cuda"
    model_class = SegmentationDOFA.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
        weights_from_checkpoint_path=None,
    )
    model = model_class.model
    model.eval().cuda()
    wrapper = _ExportWrapper(model).cuda()
    batch_size = int(404.5432096881631)  # quirk of torch export for dynamic batch size
    batch_dim = torch.export.Dim("batch", min=1, max=batch_size)
    channels_dim = torch.export.Dim("channels", min=1, max=8)
    c = 4
    x = torch.randn(batch_size, c, 512, 512, device=device)
    mean = torch.randn(c, device=device)
    std = torch.randn(c, device=device).abs() + 1e-5
    wavelengths = torch.randn(c, device=device, dtype=torch.float32)

    dynamic_shapes = {
        "x": {0: batch_dim, 1: channels_dim},
        "mean": {0: channels_dim},
        "std": {0: channels_dim},
        "wavelengths": {0: channels_dim},
    }
    exported = torch.export.export(
        wrapper,
        args=(x, mean, std, wavelengths),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    torch.export.save(exported, output_path)
    logger.info("Exported to %s", output_path)
