"""DINOv3 segmentation model."""

import torch
import torch.nn.functional as fn
from torch import Tensor, nn

from geo_deep_learning.models.decoders.upernet import UperNetDecoder
from geo_deep_learning.models.encoders.dino_v3 import DINOv3Adapter, vit_large
from geo_deep_learning.models.heads.segmentation_head import (
    SegmentationHead,
    SegmentationOutput,
)

# SAT-493M. https://github.com/facebookresearch/dinov3
_SAT_MEAN = (0.430, 0.411, 0.296)
_SAT_STD = (0.213, 0.156, 0.143)


class DINOv3SegmentationModel(nn.Module):
    """DINOv3 segmentation model."""

    def __init__(self, num_classes: int = 1, *, use_dora: bool = False) -> None:
        """Initialize DINOv3 segmentation model."""
        super().__init__()
        backbone = vit_large()
        if use_dora:
            backbone.inject_dora()
        self.encoder = DINOv3Adapter(backbone=backbone)
        embed_dim = self.encoder.backbone.embed_dim
        self.decoder = UperNetDecoder(
            embed_dim=[embed_dim] * 4,
            channels=256,
            align_corners=False,
            scale_modules=False,
            norm="gn",
        )
        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)
        self.register_buffer("sat_mean", torch.tensor(_SAT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("sat_std", torch.tensor(_SAT_STD).view(1, 3, 1, 1))
        self._manage_gradients(use_dora=use_dora)

    def _manage_gradients(self, *, use_dora: bool = False) -> None:
        """Freeze the backbone. Train the adapter, decoder, head, and DoRA."""
        for name, param in self.named_parameters():
            on_backbone = name.startswith("encoder.backbone.")
            dora = "lora_" in name or "mag_" in name
            param.requires_grad = (not on_backbone) or (use_dora and dora)

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
        sensor_mean: Tensor,
        sensor_std: Tensor,
    ) -> SegmentationOutput:
        """X is 0-1. Stem uses sensor stats; ViT RGB uses the SAT buffers."""
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            msg = f"H and W must be multiples of 32, got {tuple(x.shape[-2:])}"
            raise ValueError(msg)
        mean = sensor_mean.to(dtype=torch.float32, device=x.device)
        std = sensor_std.to(dtype=torch.float32, device=x.device)
        if mean.dim() == 1:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        x = x.float()
        stem = (x - mean) / std
        rgb = (x[:, :3] - self.sat_mean) / self.sat_std
        feats = self.encoder(stem, rgb, wavelengths)
        logits = self.head(self.decoder([feats[k] for k in "1234"]))
        logits = fn.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return SegmentationOutput(out=logits)


if __name__ == "__main__":
    model = DINOv3SegmentationModel(num_classes=150)
    x = torch.rand(1, 4, 320, 320)
    wavelengths = torch.tensor([0.665, 0.560, 0.485, 0.842])
    sensor_mean = torch.tensor([0.29, 0.27, 0.22, 0.35]).view(4, 1, 1)
    sensor_std = torch.tensor([0.15, 0.14, 0.13, 0.16]).view(4, 1, 1)
    output = model(x, wavelengths, sensor_mean, sensor_std)
