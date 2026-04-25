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

from geo_deep_learning.models.segmentation.dofa import DOFASegmentationModel
from geo_deep_learning.tools.augmentation.gridmask import FastGridMask
from geo_deep_learning.tools.losses.lejepa import SIGReg
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
        use_sigreg: bool = False,
        lambda_sig: float = 0.05,
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
        self.use_sigreg = use_sigreg
        self.lambda_sig = lambda_sig
        num_classes = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(num_classes)]
            if class_labels is None
            else class_labels
        )
        self.iou = IoU(num_classes=num_classes, ignore_index=255)
        self._total_samples_visualized = 0

        self.geometric_aug = self._geometric_aug()
        self.gridmask = FastGridMask(grid_size=64, mask_ratio=0.5, p=0.5)

        if self.use_sigreg:
            self.sigreg = SIGReg(num_slices=256)

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
            use_sigreg=self.use_sigreg,
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

    def on_train_epoch_start(self) -> None:
        """On train epoch start."""
        # Dynamically scale GridMask probability once per epoch.
        max_p = 0.5
        ramp_up_epochs = 15
        current_epoch = self.trainer.current_epoch
        if current_epoch >= ramp_up_epochs:
            new_p = max_p
        else:
            new_p = max_p * (current_epoch / ramp_up_epochs)
        self.gridmask.p = new_p
        self.log("gridmask_p", new_p, on_step=False, on_epoch=True, sync_dist=True)

    def on_after_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On after batch transfer."""
        if self.trainer.training:
            x, y = self.geometric_aug(batch["image"], batch["mask"])
            batch["image"] = x
            batch["mask"] = y
        batch["image"] = standardization(batch["image"], batch["mean"], batch["std"])
        if self.trainer.training:
            batch["image"] = self.gridmask(batch["image"])
        return batch

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
        loss_main = self.loss(outputs.out, y)
        loss_aux = self.loss(outputs.aux["aux"], y)
        seg_loss = loss_main + 0.4 * loss_aux
        total_loss = seg_loss
        metrics = {
            "seg_loss": seg_loss,
        }

        if self.use_sigreg and "sigreg_embedding" in outputs.aux:
            sigreg_loss = self.sigreg(outputs.aux["sigreg_embedding"])
            total_loss += self.lambda_sig * sigreg_loss
            metrics["sigreg_loss"] = sigreg_loss

        metrics["train_loss"] = total_loss
        self.log_dict(
            metrics,
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
        loss = self.loss(outputs.out, y)
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
        wv = batch["wavelengths"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        outputs = self(x, wv)
        loss = self.loss(outputs.out, y)

        if self.num_classes == 1:
            y_hat = (outputs.out.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = outputs.out.softmax(dim=1).argmax(dim=1)

        self.iou.update(y_hat, y)

        self.log(
            "test_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )

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

    def on_test_epoch_end(self) -> None:
        """Compute and log IoU metrics at end of test epoch."""
        per_class_iou = self.iou.compute()
        metrics = {
            f"iou_{label}": iou
            for label, iou in zip(self.labels, per_class_iou, strict=False)
        }
        metrics["mean_iou"] = torch.nanmean(per_class_iou).item()
        self.log_dict(metrics, logger=True, sync_dist=True)
        self.iou.reset()

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
