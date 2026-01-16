"""SSL MixTransformer model."""

import logging
from typing import Any

import kornia as krn
import torch
from kornia.augmentation import AugmentationSequential
from lightning.pytorch import LightningModule

from geo_deep_learning.models.ssl.lejepa_mit import LeJEPAMixTransformer
from geo_deep_learning.tools.losses.lejepa import LeJEPALoss
from geo_deep_learning.tools.utils import load_weights_from_checkpoint, standardization

logger = logging.getLogger(__name__)


class SSLMixTransformer(LightningModule):
    """SSL MixTransformer model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
        stages: list[int] | None = None,
        projection_head_dim: int = 128,
        *,
        use_dynamic_encoder: bool = False,
        load_parts: str | list[str] | None = None,
        weights_from_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize SSL MixTransformer model."""
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.in_channels = in_channels
        self.weights = weights
        self.stages = stages
        self.projection_head_dim = projection_head_dim
        self.use_dynamic_encoder = use_dynamic_encoder
        self.load_parts = load_parts
        self.weights_from_checkpoint_path = weights_from_checkpoint_path
        self.loss = LeJEPALoss(lambda_sig=0.02)
        self._apply_aug()

    def _apply_aug(self) -> None:
        """Strict Augmentation pipeline for LeJEPA (official order)."""
        # 1. Crops
        self.global_crop = krn.augmentation.RandomResizedCrop(
            size=(224, 224),
            scale=(0.3, 1.0),
        )
        self.local_crop = krn.augmentation.RandomResizedCrop(
            size=(98, 98),
            scale=(0.05, 0.3),
        )

        # 2. Flip
        self.flip = krn.augmentation.RandomHorizontalFlip(p=0.5)

        # 3-4. Color + Grayscale (RGB only)
        self.color_aug = AugmentationSequential(
            krn.augmentation.ColorJiggle(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
                p=0.8,
            ),
            krn.augmentation.RandomGrayscale(p=0.2),
            data_keys=["image"],
        )

        # 5. Blur
        self.blur = krn.augmentation.RandomGaussianBlur(
            kernel_size=(23, 23),
            sigma=(0.1, 2.0),
            p=0.5,
        )

        # 6. Solarize (RGB only)
        self.solarize = krn.augmentation.RandomSolarize(thresholds=0.5, p=0.2)

    def _augment(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        crop: krn.augmentation.RandomResizedCrop,
    ) -> torch.Tensor:
        """Apply augmentations in official LeJEPA order."""
        # 1. Crop
        x = crop(x)

        # 2. Flip (all channels)
        x = self.flip(x)

        # 3-4. ColorJitter + Grayscale (RGB only)
        rgb, rest = x[:, :3], x[:, 3:]
        rgb = self.color_aug(rgb)
        x = torch.cat([rgb, rest], dim=1)

        # 5. Blur (all channels)
        x = self.blur(x)

        # 6. Solarize (RGB only)
        rgb, rest = x[:, :3], x[:, 3:]
        rgb = self.solarize(rgb)
        x = torch.cat([rgb, rest], dim=1)
        return standardization(x, mean, std)

    def configure_model(self) -> None:
        """Configure model."""
        self.model = LeJEPAMixTransformer(
            encoder=self.encoder,
            in_channels=self.in_channels,
            weights=self.weights,
            stages=self.stages,
            projection_head_dim=self.projection_head_dim,
            use_dynamic_encoder=self.use_dynamic_encoder,
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

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            weight_decay=5e-2,
        )
        warmup_epochs = 5
        max_epochs = self.trainer.max_epochs
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=5e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def on_after_batch_transfer(
        self,
        batch: dict[str, Any],
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        """On after batch transfer."""
        images = batch["image"].to(self.device, non_blocking=True)
        mean = batch["mean"].to(self.device, non_blocking=True)
        std = batch["std"].to(self.device, non_blocking=True)

        if not self.trainer.training:
            batch["views"] = [
                self._augment(images, mean, std, self.global_crop),
                self._augment(images, mean, std, self.global_crop),
            ]
            return batch

        views = []
        # 2 global views to capture global context
        views.extend(
            [self._augment(images, mean, std, self.global_crop) for _ in range(2)],
        )
        # 6 local views to capture local context
        views.extend(
            [self._augment(images, mean, std, self.local_crop) for _ in range(6)],
        )

        batch["views"] = views
        return batch

    def training_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run training step."""
        views = batch["views"]
        zs = [self(view) for view in views]
        zs = torch.stack(zs, dim=0)
        loss = self.loss(zs)
        batch_size = zs.shape[1]

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
        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run validation step."""
        views = batch["views"]
        zs = [self(view) for view in views]
        zs = torch.stack(zs, dim=0)
        batch_size = zs.shape[1]
        loss = self.loss(zs)

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
        return loss

    def test_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Run test step."""
        views = batch["views"]
        zs = [self(view) for view in views]
        zs = torch.stack(zs, dim=0)
        batch_size = zs.shape[1]
        loss = self.loss(zs)

        self.log(
            "test_loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
