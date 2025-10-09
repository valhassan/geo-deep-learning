"""Pixel-wise Regression SegFormer model."""

import logging
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from models.segmentation.segformer import SegFormerSegmentationModel
from tools.utils import denormalization, load_weights_from_checkpoint
from tools.visualization import visualize_regression
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

warnings.filterwarnings(
    "ignore",
    message="Default grid_sample and affine_grid behavior has changed",
)

logger = logging.getLogger(__name__)


class RegressionSegformer(LightningModule):
    """Pixel-wise Regression SegFormer model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str,
        *,
        image_size: tuple[int, int],
        in_channels: int,
        num_outputs: int = 1,
        max_samples: int,
        loss: Callable,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        use_dynamic_encoder: bool = False,
        freeze_layers: list[str] | None = None,
        weights: str | None = None,
        weights_from_checkpoint_path: str | None = None,
        output_min: float | None = None,
        output_max: float | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.in_channels = in_channels
        self.num_outputs = num_outputs
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

        self.output_min = output_min
        self.output_max = output_max
        self.aux_weight = {"s4": 0.4, "s3": 0.3, "s2": 0.2}

        # Regression metrics
        self.train_mae = MeanAbsoluteError()
        self.train_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()

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
            num_classes=self.num_outputs,
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
        if (
            self.hparams["scheduler"]["class_path"]
            == "torch.optim.lr_scheduler.OneCycleLR"
        ):
            max_lr = (
                self.hparams.get("scheduler", {}).get("init_args", {}).get("max_lr")
            )
            stepping_batches = self.trainer.estimated_stepping_batches
            if stepping_batches > -1:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                )
            elif (
                stepping_batches == -1
                and getattr(self.trainer.datamodule, "epoch_size", None) is not None
            ):
                batch_size = self.trainer.datamodule.batch_size
                epoch_size = self.trainer.datamodule.epoch_size
                accumulate_grad_batches = self.trainer.accumulate_grad_batches
                max_epochs = self.trainer.max_epochs
                steps_per_epoch = math.ceil(
                    epoch_size / (batch_size * accumulate_grad_batches),
                )
                buffer_steps = int(steps_per_epoch * accumulate_grad_batches)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    steps_per_epoch=steps_per_epoch + buffer_steps,
                    epochs=max_epochs,
                )
            else:
                stepping_batches = (
                    self.hparams.get("scheduler", {})
                    .get("init_args", {})
                    .get("total_steps")
                )
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    total_steps=stepping_batches,
                )
        else:
            scheduler = self.scheduler(optimizer)

        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(image)

    def _clamp_output(self, output: Tensor) -> Tensor:
        """Optionally clamp output to valid range."""
        if self.output_min is not None and self.output_max is not None:
            return torch.clamp(output, self.output_min, self.output_max)
        return output

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

    def transfer_batch_to_device(
        self,
        batch: dict[str, Any],
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Override to handle TorchGeo frozen dataclasses."""
        # Only transfer tensors to device, keep metadata on CPU
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(device)
            else:
                device_batch[key] = value  # Keep on CPU (BoundingBox, CRS, etc.)
        return device_batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Run training step."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        target_dim = 3
        if y.dim() == target_dim:  # (B, H, W)
            y = y.unsqueeze(1)  # (B, 1, H, W)
        y = y.float()
        outputs = self(x)
        main_loss = self.loss(outputs.out, y)

        # Auxiliary loss
        aux_loss = torch.zeros((), device=y.device, dtype=main_loss.dtype)
        aux = outputs.aux or {}
        for key, weight in self.aux_weight.items():
            if weight and key in aux:
                logits = aux[key]
                aux_loss = aux_loss + weight * self.loss(logits, y)

        loss = main_loss + aux_loss

        # Compute predictions for metrics
        y_hat = self._clamp_output(outputs.out)
        self.train_mae(y_hat, y)
        self.train_mse(y_hat, y)

        self.log(
            "train_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log(
            "train_mae",
            self.train_mae,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log(
            "train_mse",
            self.train_mse,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
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

        # Ensure target is float and has correct shape
        target_dim = 3
        if y.dim() == target_dim:  # (B, H, W)
            y = y.unsqueeze(1)  # (B, 1, H, W)
        y = y.float()

        outputs = self(x)
        loss = self.loss(outputs.out, y)

        # Compute predictions
        y_hat = self._clamp_output(outputs.out)
        self.val_mae(y_hat, y)
        self.val_mse(y_hat, y)
        self.val_r2(y_hat.flatten(), y.flatten())

        self.log(
            "val_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log(
            "val_mae",
            self.val_mae,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log(
            "val_mse",
            self.val_mse,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        self.log(
            "val_r2",
            self.val_r2,
            batch_size=batch_size,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )

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

        # Ensure target is float and has correct shape
        target_dim = 3
        if y.dim() == target_dim:  # (B, H, W)
            y = y.unsqueeze(1)  # (B, 1, H, W)
        y = y.float()

        outputs = self(x)
        loss = self.loss(outputs.out, y)

        # Compute predictions
        y_hat = self._clamp_output(outputs.out)
        mae = self.test_mae(y_hat, y)
        mse = self.test_mse(y_hat, y)
        r2 = self.test_r2(y_hat.flatten(), y.flatten())

        metrics = {
            "test_loss": loss,
            "test_mae": mae,
            "test_mse": mse,
            "test_r2": r2,
        }

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
            rank_zero_only=True,
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
    ) -> int:
        """
        Log visualizations for regression.

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
        target_dim = 3
        try:
            logger.info("Logging visualizations")
            image_batch = batch["image"]
            mask_batch = batch["mask"]
            if mask_batch.dim() == target_dim:
                mask_batch = mask_batch.unsqueeze(1)
            batch_image_name = batch.get(
                "image_name",
                [f"sample_{i}" for i in range(len(image_batch))],
            )
            mean_batch = batch.get(
                "mean",
                [torch.zeros(self.in_channels) for _ in range(len(image_batch))],
            )
            std_batch = batch.get(
                "std",
                [torch.ones(self.in_channels) for _ in range(len(image_batch))],
            )

            num_samples = min(max_samples, len(image_batch))
            for i in range(num_samples):
                image = image_batch[i].cpu()
                image_name = batch_image_name[i]
                mean = mean_batch[i] if isinstance(mean_batch, list) else mean_batch
                std = std_batch[i] if isinstance(std_batch, list) else std_batch

                # Move mean/std to CPU for visualization
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu()
                if isinstance(std, torch.Tensor):
                    std = std.cpu()

                image = denormalization(image, mean=mean, std=std)

                fig = visualize_regression(
                    image=image,
                    target=mask_batch[i].cpu(),
                    prediction=outputs[i].cpu(),
                    sample_name=image_name,
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
            logger.exception("Error in regression visualization")
            return 0
        else:
            return num_samples
