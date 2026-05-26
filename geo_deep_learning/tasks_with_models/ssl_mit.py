"""SSL MixTransformer model."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
import torch.nn.functional as fn
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities import rank_zero_only

from geo_deep_learning.models.decoders.segformer_mlp import Decoder
from geo_deep_learning.models.ssl.geojepa_mit import GeoJEPAMixTransformer
from geo_deep_learning.tools.losses.geojepa import GeoJEPALoss
from geo_deep_learning.tools.metrics.segmentation_iou import IoU
from geo_deep_learning.tools.utils import (
    denormalization,
    load_weights_from_checkpoint,
    standardization,
)
from geo_deep_learning.tools.visualization import visualize_prediction

logger = logging.getLogger(__name__)


class SSLMixTransformer(LightningModule):
    """GeoJEPA SSL MixTransformer model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        num_classes: int = 1,
        weights: str | None = None,
        proj_dim: int = 256,
        lambda_sig: float = 0.05,
        embedding_dim: int | None = None,
        probe_loss: Callable | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 5e-2,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 1e-7,
        *,
        use_dynamic_encoder: bool = False,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        max_samples: int = 0,
        load_parts: str | list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize GeoJEPA SSL MixTransformer model."""
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.weights = weights
        self.proj_dim = proj_dim
        self.lambda_sig = lambda_sig
        self.embedding_dim = embedding_dim
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {"interval": "epoch"}
        self.lr = lr
        self.weight_decay = weight_decay
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay

        self.use_dynamic_encoder = use_dynamic_encoder
        self.load_parts = load_parts
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.probe_loss = probe_loss or torch.nn.CrossEntropyLoss()
        self.geojepa_loss = GeoJEPALoss(lambda_sig=self.lambda_sig)
        self.geometric_aug = self._geometric_aug()

        self.class_colors = class_colors
        num_classes_for_iou = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(num_classes_for_iou)]
            if class_labels is None
            else class_labels
        )
        self.iou = IoU(num_classes=num_classes_for_iou, ignore_index=255)
        self.threshold = 0.5
        self.max_samples = max_samples
        self._total_samples_visualized = 0

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
            data_keys=["input", "input"],
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
        return {k: v for k, v in state.items() if not k.startswith("geometric_aug.")}

    def configure_model(self) -> None:
        """Configure model."""
        self.model = GeoJEPAMixTransformer(
            encoder=self.encoder,
            in_channels=self.in_channels,
            weights=self.weights,
            proj_dim=self.proj_dim,
            use_dynamic_encoder=self.use_dynamic_encoder,
        )
        self.decoder = Decoder(
            encoder=self.encoder,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
        )
        # Only loading decoder weights from checkpoint for now.
        if self.weights_from_checkpoint_path:
            map_location = self.device
            logger.info(
                "Loading decoder weights from checkpoint %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                model=self.decoder,
                checkpoint_path=self.weights_from_checkpoint_path,
                load_parts=["decoder"],
                map_location=map_location,
            )

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers."""
        g1 = {
            "params": self.model.parameters(),
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        g2 = {
            "params": self.decoder.parameters(),
            "lr": self.probe_lr,
            "weight_decay": self.probe_weight_decay,
        }
        optimizer = self.optimizer([g1, g2])
        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def on_fit_start(self) -> None:
        """On fit start."""
        self.geometric_aug = self.geometric_aug.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def _seg_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segmentation forward through frozen encoder."""
        with torch.no_grad():
            feats = self.model.encoder(x)
        seg_out, _ = self.decoder(feats)
        return fn.interpolate(
            seg_out,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    def on_after_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On after batch transfer."""
        if self.trainer.training:
            x, y = self.geometric_aug(batch["image_low"], batch["image_high"])
            batch["image_low"] = x
            batch["image_high"] = y
            batch["image_low"] = standardization(
                batch["image_low"],
                batch["mean"],
                batch["std"],
            )
            batch["image_high"] = standardization(
                batch["image_high"],
                batch["mean"],
                batch["std"],
            )
        elif isinstance(batch.get("image"), torch.Tensor):
            batch["image"] = standardization(
                batch["image"],
                batch["mean"],
                batch["std"],
            )
        return batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run training step."""
        z_low = self(batch["image_low"])
        z_high = self(batch["image_high"])
        zs = torch.stack([z_low, z_high], dim=0)
        loss_dict = self.geojepa_loss(zs)

        with torch.no_grad():
            z_flat = zs.reshape(-1, zs.shape[-1])
            z_std = z_flat.std(dim=0).mean()
            z_norm = z_flat.norm(dim=-1).mean()

        batch_size = batch["image_low"].shape[0]
        loss_dict.update({
            "z_std": z_std,
            "z_norm": z_norm,
        })
        self.log_dict({f"{k}": v for k, v in loss_dict.items()},
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )
        return loss_dict["ssl_loss"]

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run validation step (Segmentation Probe)."""
        x = batch["image"]
        y = batch["mask"]
        batch_size = x.shape[0]
        y = y.squeeze(1).long()
        seg_out = self._seg_forward(x)
        seg_loss = self.probe_loss(seg_out, y)
        if self.num_classes == 1:
            y_hat = (seg_out.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = seg_out.softmax(dim=1).argmax(dim=1)

        self.iou.update(y_hat, y)
        if (
            self.trainer.is_global_zero
            and self._total_samples_visualized < self.max_samples
        ):
            remaining_samples = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining_samples, len(x))
            samples_visualized = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=y_hat,
                max_samples=samples_to_visualize,
                artifact_prefix="val",
                epoch_suffix=True,
            )
            self._total_samples_visualized += samples_visualized

        self.log(
            "val_loss",
            seg_loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )

    def on_validation_epoch_end(self) -> None:
        """Compute and log IoU metrics at end of validation epoch."""
        per_class_iou = self.iou.compute()
        metrics = {
            f"iou_{label}": iou
            for label, iou in zip(self.labels, per_class_iou, strict=False)
        }
        metrics["mean_iou"] = torch.nanmean(per_class_iou).item()
        self.log_dict(metrics, logger=True, sync_dist=True)
        self.iou.reset()

    @rank_zero_only
    def _log_visualizations(  # noqa: PLR0913
        self,
        trainer: Trainer,
        batch: dict[str, Any],
        outputs: torch.Tensor,
        max_samples: int,
        artifact_prefix: str = "val",
        *,
        epoch_suffix: bool = True,
    ) -> int:
        """
        Segmentation Probe visualizations.

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
            logger.exception("Error in Segmentation Probe visualization")
        else:
            return num_samples

    # def test_step(
    #     self,
    #     batch: dict[str, Any],
    #     batch_idx: int,
    # ) -> torch.Tensor:
    #     """Run test step (Segmentation Probe)."""
    #     x = batch["image"]
    #     y = batch["mask"]
    #     batch_size = x.shape[0]
    #     y = y.squeeze(1).long()
    #     seg_out, _ = self._seg_forward(x)
    #     seg_loss = self.probe_loss(seg_out, y)
    #     self.log(
    #         "test_loss",
    #         seg_loss,
    #         batch_size=batch_size,
    #         prog_bar=True,
    #         logger=True,
    #         on_step=False,
    #         on_epoch=True,
    #         sync_dist=True,
    #         rank_zero_only=False,
    #     )
