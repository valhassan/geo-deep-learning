"""DINOv3 segmentation model."""

import torch
from torch import Tensor, nn

from geo_deep_learning.models.decoders.mask2former.mask2former import Mask2FormerHead
from geo_deep_learning.models.encoders.dino_v3 import DINOv3Adapter, vit_large

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
        self.encoder = DINOv3Adapter(
            backbone=backbone,
        )
        embed_dim = self.encoder.backbone.embed_dim
        patch_size = self.encoder.backbone.patch_size
        self.decoder = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim, int(patch_size / 2), int(patch_size / 2), 4],
            },
            num_classes=num_classes,
        )
        self.register_buffer("sat_mean", torch.tensor(_SAT_MEAN).view(1, 3, 1, 1))
        self.register_buffer("sat_std", torch.tensor(_SAT_STD).view(1, 3, 1, 1))
        self._manage_gradients(use_dora=use_dora)

    def _manage_gradients(self, *, use_dora: bool = False) -> None:
        """Freeze the backbone. Train the adapter, decoder, and DoRA."""
        for param in self.parameters():
            param.requires_grad = False

        for name, module in self.encoder.named_children():
            if name == "backbone":
                continue
            for param in module.parameters():
                param.requires_grad = True
        self.encoder.level_embed.requires_grad = True

        if use_dora:
            for name, param in self.encoder.backbone.named_parameters():
                if "lora_" in name or "mag_" in name:
                    param.requires_grad = True

        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
        sensor_mean: Tensor,
        sensor_std: Tensor,
    ) -> Tensor:
        """X is 0-1. Stem uses sensor stats; ViT RGB uses the SAT buffers."""
        mean = sensor_mean.to(dtype=torch.float32, device=x.device)
        std = sensor_std.to(dtype=torch.float32, device=x.device)
        if mean.dim() == 1:
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        x = x.float()
        stem = (x - mean) / std
        rgb = (x[:, :3] - self.sat_mean) / self.sat_std
        return self.decoder(self.encoder(stem, rgb, wavelengths))


if __name__ == "__main__":
    model = DINOv3SegmentationModel(num_classes=150)
    x = torch.rand(1, 4, 320, 320)
    wavelengths = torch.tensor([0.665, 0.560, 0.485, 0.842])
    sensor_mean = torch.tensor([0.29, 0.27, 0.22, 0.35]).view(4, 1, 1)
    sensor_std = torch.tensor([0.15, 0.14, 0.13, 0.16]).view(4, 1, 1)
    output = model(x, wavelengths, sensor_mean, sensor_std)
    # print(output.keys())
    # print(f"pred_logits shape: {output['pred_logits'].shape}")
    # print(f"pred_masks shape: {output['pred_masks'].shape}")
