"""Segmentation SegFormer model."""

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
from geo_deep_learning.models.segmentation.segformer import SegFormerSegmentationModel
from geo_deep_learning.tools.augmentation import RandomD4, RandomPlanckian
from geo_deep_learning.tools.metrics.segmentation_iou import IoU
from geo_deep_learning.tools.utils import (
    denormalization,
    load_weights_from_checkpoint,
    normalization,
    standardization,
)
from geo_deep_learning.tools.visualization import visualize_prediction

logger = logging.getLogger(__name__)


class SegmentationSegformer(LightningModule):
    """Segmentation SegFormer model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str,
        *,
        image_size: tuple[int, int],
        in_channels: int,
        num_classes: int,
        max_samples: int,
        embedding_dim: int | None = None,
        loss: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        use_dynamic_encoder: bool = False,
        freeze_layers: list[str] | None = None,
        weights: str | None = None,
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
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_samples = max_samples
        self.embedding_dim = embedding_dim
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.weights = weights
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.load_parts = load_parts
        self.use_dynamic_encoder = use_dynamic_encoder
        self.freeze_layers = freeze_layers
        self.class_colors = class_colors
        self.threshold = 0.5
        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(num_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou = IoU(num_classes=num_classes, ignore_index=255)
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
        self.model = SegFormerSegmentationModel(
            encoder=self.encoder,
            in_channels=self.in_channels,
            weights=self.weights,
            freeze_layers=self.freeze_layers,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            use_dynamic_encoder=self.use_dynamic_encoder,
        )
        if self.weights_from_checkpoint_path:
            map_location = self.device
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                self.model,
                self.weights_from_checkpoint_path,
                load_parts=self.load_parts,
                map_location=map_location,
            )

    def on_fit_start(self) -> None:
        """On fit start."""
        self.geometric_aug = self.geometric_aug.to(self.device)
        self.radiometric_aug = self.radiometric_aug.to(self.device)

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(
        self,
        image: Tensor,
        wavelengths: Tensor | None = None,
    ) -> Tensor:
        """Forward pass. Wavelengths are required when the dynamic stem is on."""
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
            )
            batch["image"], batch["mask"] = out[0], out[1]
            batch.update(dict(zip(geo, out[2:], strict=True)))
            batch["image"] = self.radiometric_aug(
                batch["image"],
                batch["wavelengths"],
            )
        batch["image"] = standardization(batch["image"], batch["mean"], batch["std"])
        return batch

    @staticmethod
    def _loss_kw(batch: dict[str, Any]) -> dict[str, Tensor | None]:
        return {k: batch.get(k) for k in GEO_KEYS}

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
        loss = self.loss(outputs.out, y, **self._loss_kw(batch))

        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
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
    ) -> int:
        """
        Log prediction figures.

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
            logger.exception("Error in SegFormer visualization")
            return 0
        else:
            return num_samples


class _ExportWrapper(nn.Module):
    """
    SegFormer export: single forward, norm + standardize then model.

    Inputs: x (B,C,H,W), mean (C,), std (C,), wavelengths (C,).
    Returns logits (no TTA). Wavelengths are required by the dynamic stem.
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


def export_model(
    checkpoint_path: str,
    output_path: str,
    metadata_path: str | None = None,
) -> None:
    """Export SegFormer: (x, mean, std, wavelengths) -> logits, single forward."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = SegmentationSegformer.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
        weights_from_checkpoint_path=None,
    )
    model = model_class.model
    model.eval().to(device)
    wrapper = _ExportWrapper(model).to(device)
    example_batch = 2
    batch_dim = torch.export.Dim("batch", min=1, max=32)
    h, w = model_class.image_size
    dynamic = model_class.use_dynamic_encoder
    c = 4 if dynamic else model_class.in_channels
    x = torch.randn(example_batch, c, h, w, device=device)
    mean = torch.randn(c, device=device)
    std = torch.randn(c, device=device).abs() + 1e-5
    wavelengths = torch.linspace(0.45, 2.2, c, device=device)
    dynamic_shapes: dict[str, dict[int, torch.export.Dim]] = {
        "x": {0: batch_dim},
        "mean": {},
        "std": {},
        "wavelengths": {},
    }
    if dynamic:
        channels_dim = torch.export.Dim("channels", min=1, max=32)
        dynamic_shapes["x"][1] = channels_dim
        dynamic_shapes["mean"][0] = channels_dim
        dynamic_shapes["std"][0] = channels_dim
        dynamic_shapes["wavelengths"][0] = channels_dim
    extra = None
    if metadata_path:
        with Path(metadata_path).open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        extra = {"metadata.json": json.dumps(metadata)}
    exported = torch.export.export(
        wrapper,
        args=(x, mean, std, wavelengths),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    torch.export.save(exported, output_path, extra_files=extra)
    logger.info("Exported to %s", output_path)
