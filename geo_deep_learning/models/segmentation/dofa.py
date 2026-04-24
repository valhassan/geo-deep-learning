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

    def __init__(
        self,
        encoder: str = "dofa_base",
        image_size: tuple[int, int] = (512, 512),
        freeze_layers: list[str] | None = None,
        num_classes: int = 1,
        *,
        pretrained: bool = True,
        use_sigreg: bool = False,
    ) -> None:
        """Initialize DOFA segmentation model."""
        super().__init__(
            DOFAv2,
            None,
            UperNetDecoder,
            SegmentationHead,
            SegmentationOutput,
        )
        self.use_sigreg = use_sigreg
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

        if self.use_sigreg:
            self.projection_head = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.embed_dim, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 16),
            )

        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)

    def forward(self, x: torch.Tensor, wavelengths: torch.Tensor) -> SegmentationOutput:
        """Forward pass."""
        expected_ndim = 2
        if wavelengths.dim() == expected_ndim:
            wavelengths = wavelengths[0]  # DOFA expects a single wavelength
        image_size = x.shape[2:]
        feats = self.encoder(x, wavelengths)
        x = self.decoder(feats)
        x = self.head(x)
        x = fn.interpolate(
            input=x,
            size=image_size,
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )

        aux_x = self.aux_head(feats[2])
        aux_x = fn.interpolate(
            input=aux_x,
            size=image_size,
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )
        aux_dict = {"aux": aux_x}
        if self.use_sigreg:
            aux_dict["sigreg_embedding"] = self.projection_head(feats[-1])

        return SegmentationOutput(out=x, aux=aux_dict)


if __name__ == "__main__":
    model = DOFASegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    wavelengths = torch.tensor([0.665, 0.549, 0.481])
    outputs = model(x, wavelengths)
    # print(f"outputs.shape: {outputs.out.shape}")
    # print(f"aux_outputs.shape: {outputs.aux.shape}")
