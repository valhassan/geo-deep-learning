"""DOFA segmentation model."""

import torch
import torch.nn.functional as fn

from geo_deep_learning.models.decoders.upernet import UperNetDecoder
from geo_deep_learning.models.encoders.dofa_v2 import (
    DOFAv2,
    create_dofa_base,
    create_dofa_large,
)
from geo_deep_learning.models.heads.fcn_head import FCNHead
from geo_deep_learning.models.heads.segmentation_head import (
    SegmentationHead,
    SegmentationOutput,
)

from .base import BaseSegmentationModel


class DOFASegmentationModel(BaseSegmentationModel):
    """DOFA segmentation model."""

    _BATCHED_WAVELENGTHS_NDIM = 2

    def __init__(
        self,
        encoder: str = "dofa_base",
        image_size: tuple[int, int] = (512, 512),
        freeze_layers: list[str] | None = None,
        num_classes: int = 1,
        *,
        pretrained: bool = True,
    ) -> None:
        """Initialize DOFA segmentation model."""
        super().__init__(
            DOFAv2,
            None,
            UperNetDecoder,
            SegmentationHead,
            SegmentationOutput,
        )
        if encoder == "dofa_base":
            self.embed_dim = 768
            self.encoder = create_dofa_base(img_size=image_size, pretrained=pretrained)

        elif encoder == "dofa_large":
            self.embed_dim = 1024
            self.encoder = create_dofa_large(img_size=image_size, pretrained=pretrained)
        else:
            msg = f"Invalid encoder: {encoder}"
            raise ValueError(msg)

        self.decoder = UperNetDecoder(
            embed_dim=[self.embed_dim] * 4,
            pool_scales=(1, 2, 3, 6),
            channels=256,
            align_corners=False,
            scale_modules=True,
        )
        self.aux_head = FCNHead(
            in_channels=self.embed_dim,
            channels=256,
            num_convs=1,
            num_classes=num_classes,
        )

        self.head = SegmentationHead(in_channels=256, num_classes=num_classes)

        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)

    @staticmethod
    def _normalize_wavelengths(wavelengths: torch.Tensor) -> torch.Tensor:
        if wavelengths.dim() == DOFASegmentationModel._BATCHED_WAVELENGTHS_NDIM:
            return wavelengths[0]  # DOFA expects a single wavelength
        return wavelengths

    def forward_encoder(
        self,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encoder-only forward. Returns multi-scale feature maps."""
        wavelengths = self._normalize_wavelengths(wavelengths)
        return self.encoder(x, wavelengths)

    def _to_image(
        self,
        x: torch.Tensor,
        image_size: tuple[int, int],
        padded_size: tuple[int, int],
    ) -> torch.Tensor:
        x = fn.interpolate(x, size=padded_size, mode="bilinear", align_corners=False)
        h, w = image_size
        top = (x.shape[-2] - h) // 2
        left = (x.shape[-1] - w) // 2
        return x[..., top : top + h, left : left + w]

    def forward_decoder(
        self,
        feats: list[torch.Tensor],
        *,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run decoder + heads. Returns logits, aux logits, sdf."""
        stride = self.encoder.patch_stride
        padded_size = (feats[0].shape[-2] * stride, feats[0].shape[-1] * stride)
        dec = self.decoder(feats)
        logits = self._to_image(self.head(dec), image_size, padded_size)
        sdf = self._to_image(self.auxilary_head(dec), image_size, padded_size)
        aux_logits = self._to_image(self.aux_head(feats[2]), image_size, padded_size)
        return logits, aux_logits, sdf

    def forward(
        self,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        *,
        return_feat: bool = True,
    ) -> SegmentationOutput:
        """
        Run full segmentation forward.

        Args:
            x: Input image tensor (B, C, H, W).
            wavelengths: Wavelength tensor (B, C) or (C,).
            return_feat: If True, include last encoder feature map in aux as "feat".

        """
        image_size = x.shape[2:]
        feats = self.forward_encoder(x, wavelengths)
        logits, aux_logits = self.forward_decoder(feats, image_size=image_size)

        aux_dict: dict[str, torch.Tensor] = {"aux": aux_logits}
        if return_feat:
            aux_dict["feat"] = feats[-1]

        return SegmentationOutput(out=logits, aux=aux_dict)


if __name__ == "__main__":
    model = DOFASegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    wavelengths = torch.tensor([0.665, 0.549, 0.481])
    outputs = model(x, wavelengths)
    # print(f"outputs.shape: {outputs.out.shape}")
    # print(f"aux_outputs.shape: {outputs.aux.shape}")
