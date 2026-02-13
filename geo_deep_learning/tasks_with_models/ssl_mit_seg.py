"""SSL MixTransformer with segmentation probe (LeJEPA + SegFormer decoder)."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import kornia as krn
import torch
import torch.nn.functional as f
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import Tensor

from geo_deep_learning.models.decoders.segformer_mlp import Decoder
from geo_deep_learning.models.ssl.lejepa_mit import LeJEPAMixTransformer
from geo_deep_learning.tools.losses.lejepa import LeJEPALoss
from geo_deep_learning.tools.metrics.segmentation_iou import IoU
from geo_deep_learning.tools.utils import (
    denormalization,
    load_weights_from_checkpoint,
    normalization,
    standardization,
)
from geo_deep_learning.tools.visualization import visualize_prediction

logger = logging.getLogger(__name__)

NUM_TRAIN_VIEWS = 8  # 2 global + 6 local


class SSLMixTransformerSeg(LightningModule):
    """SSL MixTransformer with segmentation probe."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
        stages: list[int] | None = None,
        projection_head_dim: int = 128,
        num_classes: int = 5,
        embedding_dim: int | None = None,
        ignore_index: int = 255,
        lambda_seg: float = 1.0,
        seg_loss: Callable[[Tensor, Tensor], Tensor] | None = None,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        scheduler_config: dict[str, Any] | None = None,
        lr: float = 1e-4,
        weight_decay: float = 5e-2,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 1e-7,
        *,
        use_dynamic_encoder: bool = False,
        val_lejepa: bool = True,
        max_samples: int = 0,
        class_labels: list[str] | None = None,
        class_colors: list[str] | None = None,
        load_parts: str | list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize SSL + segmentation probe."""
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.val_lejepa = val_lejepa
        self.in_channels = in_channels
        self.weights = weights
        self.stages = stages
        self.projection_head_dim = projection_head_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.ignore_index = ignore_index
        self.lambda_seg = lambda_seg
        self.seg_loss = seg_loss
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

        self.max_samples = max_samples
        self.class_colors = class_colors

        self.lejepa_loss = LeJEPALoss(lambda_sig=0.02)
        num_classes_for_iou = num_classes + 1 if num_classes == 1 else num_classes
        self.labels = (
            [str(i) for i in range(num_classes_for_iou)]
            if class_labels is None
            else class_labels
        )
        self.iou = IoU(num_classes=num_classes_for_iou, ignore_index=ignore_index)
        self.threshold = 0.5
        self._total_samples_visualized = 0
        self._apply_aug()

    def _apply_aug(self) -> None:
        """Augmentation pipeline (crops, flip, radiometric)."""
        self.global_crop = krn.augmentation.RandomResizedCrop(
            size=(224, 224),
            scale=(0.3, 1.0),
        )
        self.local_crop = krn.augmentation.RandomResizedCrop(
            size=(98, 98),
            scale=(0.05, 0.3),
        )
        self.flip = krn.augmentation.RandomHorizontalFlip(p=0.5)
        self.radiometric_aug = self._radiometric_aug()
        self._aug_module_names = (
            "global_crop",
            "local_crop",
            "flip",
            "radiometric_aug",
        )

    def _radiometric_aug(self) -> AugmentationSequential:
        """Radiometric augmentations."""
        return AugmentationSequential(
            krn.augmentation.RandomBrightness(
                brightness=(0.0, 0.45),
                p=0.7,
                keepdim=True,
            ),
            krn.augmentation.RandomContrast(
                contrast=(0.6, 2.0),
                p=0.65,
                keepdim=True,
            ),
            krn.augmentation.RandomGamma(
                gamma=(0.6, 1.7),
                p=0.4,
                keepdim=True,
            ),
            krn.augmentation.RandomGaussianNoise(
                mean=0.0,
                std=0.01,
                p=0.25,
                keepdim=True,
            ),
            data_keys=["image"],
            random_apply=False,
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
        aug_prefixes = tuple(f"{n}." for n in self._aug_module_names)
        return {k: v for k, v in state.items() if not k.startswith(aug_prefixes)}

    def _augment(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        crop: krn.augmentation.RandomResizedCrop,
    ) -> torch.Tensor:
        """Apply augmentations (crop, flip, radiometric)."""
        x = crop(x)
        x = self.flip(x)
        x = self.radiometric_aug(x)
        x = torch.clamp(x, 0.0, 1.0)
        return standardization(x, mean, std)

    def configure_model(self) -> None:
        """Configure model: LeJEPA net + SegFormer decoder as seg probe."""
        self.model = LeJEPAMixTransformer(
            encoder=self.encoder,
            in_channels=self.in_channels,
            weights=self.weights,
            stages=self.stages,
            projection_head_dim=self.projection_head_dim,
            use_dynamic_encoder=self.use_dynamic_encoder,
        )
        self.decoder = Decoder(
            encoder=self.encoder,
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
        )
        if self.weights_from_checkpoint_path:
            map_location = self.device
            logger.info(
                "Loading weights from checkpoint: %s",
                self.weights_from_checkpoint_path,
            )
            load_weights_from_checkpoint(
                model=self.model,
                checkpoint_path=self.weights_from_checkpoint_path,
                load_parts=self.load_parts,
                map_location=map_location,
            )

    def on_fit_start(self) -> None:
        """On fit start."""
        for name in self._aug_module_names:
            setattr(self, name, getattr(self, name).to(self.device))

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers."""
        opt_init = self.hparams.get("optimizer", {}).get("init_args", {})
        other_kwargs = {
            k: v
            for k, v in opt_init.items()
            if k not in ("lr", "weight_decay")
        }
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
        optimizer = self.optimizer([g1, g2], **other_kwargs)
        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, **self.scheduler_config}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (projections for LeJEPA)."""
        return self.model(x)

    def preprocess(
        self,
        x: torch.Tensor,
        mean: list[float] | torch.Tensor,
        std: list[float] | torch.Tensor,
    ) -> torch.Tensor:
        """Apply normalization and standardization for inference."""
        x = normalization(
            x, image_min=0, image_max=255, norm_min=0.0, norm_max=1.0,
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
        rescale_to: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Segmentation inference (preprocessed input). Returns (B, C, H, W) probs."""
        logits = self._seg_forward(x)
        if rescale_to is not None:
            logits = f.interpolate(
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
        """Build views for LeJEPA and keep 512x512 image + mask for seg."""
        images = batch["image"]
        if images.dtype == torch.uint8:
            images = images.float().div_(255.0)
        mean = batch["mean"]
        std = batch["std"]

        batch["seg_image"] = standardization(images, mean, std)
        batch["seg_mask"] = batch["mask"]

        if not self.trainer.training:
            global_view = f.interpolate(
                images,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            local_view = f.interpolate(
                images,
                size=(98, 98),
                mode="bilinear",
                align_corners=False,
            )
            batch["views"] = [
                standardization(global_view, mean, std),
                standardization(local_view, mean, std),
            ]
            return batch

        views = [
            self._augment(images, mean, std, self.global_crop) for _ in range(2)
        ] + [self._augment(images, mean, std, self.local_crop) for _ in range(6)]
        batch["views"] = views
        return batch

    def _seg_forward(self, seg_image: Tensor) -> Tensor:
        """Segmentation probe forward."""
        feats = self.model.encoder(seg_image)
        seg_out, _ = self.decoder([f.detach() for f in feats])
        return seg_out

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """LeJEPA loss on views + seg loss on 512x512."""
        views = batch["views"]
        batch_size = views[0].shape[0]

        # LeJEPA
        if len(views) == NUM_TRAIN_VIEWS:
            z_global = self(torch.cat(views[:2], dim=0)).view(2, batch_size, -1)
            z_local = self(torch.cat(views[2:], dim=0)).view(6, batch_size, -1)
            zs = torch.cat([z_global, z_local], dim=0)
        else:
            zs = torch.stack([self(views[0]), self(views[1])], dim=0)
        lejepa_loss = self.lejepa_loss(zs)

        # Segmentation probe
        seg_image = batch["seg_image"]
        y = batch["seg_mask"].squeeze(1).long()
        seg_logits = self._seg_forward(seg_image)
        seg_logits = f.interpolate(
            seg_logits,
            size=y.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        seg_loss = self.seg_loss(seg_logits, y)

        loss = lejepa_loss + self.lambda_seg * seg_loss

        self.log_dict(
            {
                "train_lejepa_loss": lejepa_loss,
                "train_seg_loss": seg_loss,
                "train_loss": loss,
            },
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
        batch_size = batch["views"][0].shape[0]

        if self.val_lejepa:
            views = batch["views"]
            if len(views) == NUM_TRAIN_VIEWS:
                z_global = self(torch.cat(views[:2], dim=0)).view(2, batch_size, -1)
                z_local = self(torch.cat(views[2:], dim=0)).view(6, batch_size, -1)
                zs = torch.cat([z_global, z_local], dim=0)
            else:
                zs = torch.stack([self(views[0]), self(views[1])], dim=0)
            lejepa_loss = self.lejepa_loss(zs)
        else:
            lejepa_loss = None

        seg_image = batch["seg_image"]
        y = batch["seg_mask"].squeeze(1).long()
        seg_logits = self._seg_forward(seg_image)
        seg_logits = f.interpolate(
            seg_logits,
            size=y.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        seg_loss = self.seg_loss(seg_logits, y)
        if lejepa_loss is None:
            lejepa_loss = torch.zeros((), device=seg_loss.device, dtype=seg_loss.dtype)

        if self.num_classes == 1:
            y_hat = (seg_logits.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = seg_logits.softmax(dim=1).argmax(dim=1)
        self.iou.update(y_hat, y)

        val_loss = lejepa_loss + self.lambda_seg * seg_loss
        self.log(
            "val_loss",
            val_loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=False,
        )
        return val_loss

    def on_validation_epoch_end(self) -> None:
        """Log mean IoU."""
        per_class_iou = self.iou.compute()
        mean_iou = torch.nanmean(per_class_iou).item()
        self.log(
            "val_mean_iou",
            mean_iou,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.iou.reset()

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run test step (segmentation only)."""
        seg_image = batch["seg_image"]
        y = batch["seg_mask"].squeeze(1).long()
        batch_size = seg_image.shape[0]

        seg_logits = self._seg_forward(seg_image)
        seg_logits = f.interpolate(
            seg_logits,
            size=y.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        loss = self.seg_loss(seg_logits, y)

        if self.num_classes == 1:
            y_hat = (seg_logits.sigmoid().squeeze(1) > self.threshold).long()
        else:
            y_hat = seg_logits.softmax(dim=1).argmax(dim=1)

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

        if self._total_samples_visualized < self.max_samples and "image_name" in batch:
            remaining = self.max_samples - self._total_samples_visualized
            samples_to_visualize = min(remaining, len(seg_image))
            n = self._log_visualizations(
                trainer=self.trainer,
                batch=batch,
                outputs=y_hat,
                max_samples=samples_to_visualize,
                artifact_prefix="test",
                epoch_suffix=False,
            )
            self._total_samples_visualized += n

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
        artifact_prefix: str = "test",
        *,
        epoch_suffix: bool = False,
    ) -> int:
        """Log segmentation visualizations to logger (e.g. MLflow)."""
        if batch is None or outputs is None or max_samples <= 0:
            return 0
        try:
            logger.info("Logging visualizations")
            image_batch = batch["seg_image"]
            mask_batch = batch["seg_mask"].squeeze(1).long()
            batch_image_name = batch["image_name"]
            mean_batch = batch["mean"]
            std_batch = batch["std"]
            num_samples = min(max_samples, len(image_batch))
            for i in range(num_samples):
                image = denormalization(
                    image_batch[i],
                    mean=mean_batch[i],
                    std=std_batch[i],
                )
                fig = visualize_prediction(
                    image=image,
                    mask=mask_batch[i],
                    prediction=outputs[i],
                    sample_name=batch_image_name[i],
                    num_classes=self.num_classes,
                    class_colors=self.class_colors,
                )
                base_path = f"{artifact_prefix}/{Path(batch_image_name[i]).stem}"
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
            logger.exception("Error in SSL+seg visualization")
            return 0
        return num_samples
