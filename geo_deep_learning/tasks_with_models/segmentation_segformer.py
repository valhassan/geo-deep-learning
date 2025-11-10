"""Segmentation SegFormer model."""

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
from segmentation_models_pytorch.losses import SoftCrossEntropyLoss
from torch import Tensor
from torchmetrics.segmentation import MeanIoU
from torchmetrics.wrappers import ClasswiseWrapper

from geo_deep_learning.models.segmentation.segformer import SegFormerSegmentationModel
from geo_deep_learning.tools.utils import denormalization, load_weights_from_checkpoint
from geo_deep_learning.tools.visualization import visualize_prediction

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

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
        loss: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        use_dynamic_encoder: bool = False,
        freeze_layers: list[str] | None = None,
        weights: str | None = None,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
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

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}

        self.weights = weights
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.use_dynamic_encoder = use_dynamic_encoder
        self.freeze_layers = freeze_layers

        self.class_colors = class_colors
        self.threshold = 0.5
        self.ce_loss = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=255)
        self.aux_weight = {"s4": 0.4, "s3": 0.3, "s2": 0.2}

        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.iou_metric = MeanIoU(
            num_classes=num_classes,
            per_class=True,
            input_format="index",
            include_background=True,
        )
        self.labels = (
            [str(i) for i in range(num_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou_classwise_metric = ClasswiseWrapper(
            self.iou_metric,
            labels=self.labels,
        )
        self._total_samples_visualized = 0

    def _apply_aug(self) -> AugmentationSequential:
        """Augmentation pipeline."""
        random_resized_crop_zoom_in = krn.augmentation.RandomResizedCrop(
            size=self.image_size,
            scale=(1.0, 2.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )
        random_resized_crop_zoom_out = krn.augmentation.RandomResizedCrop(
            size=self.image_size,
            scale=(0.5, 1.0),
            p=0.5,
            align_corners=False,
            keepdim=True,
        )

        return AugmentationSequential(
            krn.augmentation.RandomHorizontalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomVerticalFlip(p=0.5, keepdim=True),
            krn.augmentation.RandomRotation90(
                times=(1, 3),
                p=0.5,
                align_corners=True,
                keepdim=True,
            ),
            random_resized_crop_zoom_in,
            random_resized_crop_zoom_out,
            data_keys=None,
            random_apply=1,
        )

    def configure_model(self) -> None:
        """Configure model."""
        self.model = SegFormerSegmentationModel(
            encoder=self.encoder,
            in_channels=self.in_channels,
            weights=self.weights,
            freeze_layers=self.freeze_layers,
            num_classes=self.num_classes,
            use_dynamic_encoder=self.use_dynamic_encoder,
        )
        if self.weights_from_checkpoint_path:
            map_location = self.device
            load_parts = self.hparams.get("load_parts")
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

    def configure_optimizers(self) -> list[list[dict[str, Any]]]:
        """Configure optimizers."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image)

    def on_after_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On after batch transfer."""
        if not self.trainer.training:
            return batch
        device = batch["image"].device
        aug = self._apply_aug()
        batch_aug = aug({"image": batch["image"], "mask": batch["mask"]})
        for key in ["image", "mask"]:
            if key in batch_aug and batch_aug[key].device != device:
                batch[key] = batch_aug[key].to(device, non_blocking=True)
            elif key in batch_aug:
                batch[key] = batch_aug[key]
        return batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x)
        main_loss = self.loss(outputs.out, y) + self.ce_loss(outputs.out, y)
        aux_loss = torch.zeros((), device=y.device, dtype=main_loss.dtype)
        aux = outputs.aux or {}
        for key, weight in self.aux_weight.items():
            if weight and key in aux:
                logits = aux[key]
                aux_loss = aux_loss + weight * (
                    self.loss(logits, y) + self.ce_loss(logits, y)
                )
        loss = main_loss + aux_loss

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
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x)
        loss = self.loss(outputs.out, y) + self.ce_loss(outputs.out, y)
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

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x)
        loss = self.loss(outputs.out, y) + self.ce_loss(outputs.out, y)

        if self.num_classes == 1:
            y_hat = (outputs.out.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = outputs.out.softmax(dim=1).argmax(dim=1)

        metrics = self.iou_classwise_metric(y_hat, y)
        metrics["test_loss"] = loss

        if self._total_samples_visualized < self.max_samples:
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=y_hat,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )
            self._total_samples_visualized += samples_visualized
        self.log_dict(
            metrics,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )

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
        SegFormer-specific log visualizations.

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
        else:
            return num_samples
