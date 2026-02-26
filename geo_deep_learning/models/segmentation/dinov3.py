"""DINOv3 segmentation model."""

import torch
from torch import nn

from geo_deep_learning.models.decoders.mask2former.mask2former import Mask2FormerHead
from geo_deep_learning.models.encoders.dino_v3 import DINOv3Adapter, vit_large


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
        self._manage_gradients(use_dora=use_dora)

    def _manage_gradients(self, *, use_dora: bool = False) -> None:
        """Manage gradients."""
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze specific modules
        for param in self.encoder.spm.parameters():
            param.requires_grad = True
        for param in self.encoder.interactions.parameters():
            param.requires_grad = True
        for param in self.encoder.up.parameters():
            param.requires_grad = True
        self.encoder.level_embed.requires_grad = True

        # Unfreeze DoRA specific parameters
        if use_dora:
            for name, param in self.encoder.backbone.named_parameters():
                if "lora_" in name or "mag_" in name:
                    param.requires_grad = True

        # Unfreeze Decoder
        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # noqa: RET504


if __name__ == "__main__":
    model = DINOv3SegmentationModel(num_classes=150)
    x = torch.randn(1, 3, 320, 320)
    output = model(x)
    # print(output.keys())
    # print(f"pred_logits shape: {output['pred_logits'].shape}")
    # print(f"pred_masks shape: {output['pred_masks'].shape}")
