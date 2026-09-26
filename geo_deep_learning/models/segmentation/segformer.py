"""SegFormer segmentation model."""

import torch
import torch.nn.functional as fn

from geo_deep_learning.models.decoders.segformer_mlp import Decoder
from geo_deep_learning.models.encoders.mix_transformer import get_encoder
from geo_deep_learning.models.encoders.wavelength_stem import WavelengthMixTransformer
from geo_deep_learning.models.heads.segmentation_head import SegmentationOutput

from .base import BaseSegmentationModel


class SegFormerSegmentationModel(BaseSegmentationModel):
    """SegFormer segmentation model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
        freeze_layers: list[str] | None = None,
        num_classes: int = 1,
        embedding_dim: int | None = None,
        *,
        use_dynamic_encoder: bool = False,
    ) -> None:
        """Initialize SegFormer segmentation model."""
        super().__init__()
        self.use_dynamic_encoder = use_dynamic_encoder
        skip_channels = None
        if use_dynamic_encoder:
            self.encoder = WavelengthMixTransformer(
                encoder=encoder,
                weights=weights,
            )
            skip_channels = self.encoder.stem.embed_dim
        else:
            self.encoder = get_encoder(
                name=encoder,
                in_channels=in_channels,
                depth=5,
                weights=weights,
            )
        if freeze_layers:
            self._freeze_layers(layers=freeze_layers)

        self.decoder = Decoder(
            encoder=encoder,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            skip_channels=skip_channels,
        )
        self.output_struct = SegmentationOutput

    def forward(
        self,
        img: torch.Tensor,
        wavelengths: torch.Tensor | None = None,
    ) -> SegmentationOutput:
        """Forward pass. Wavelengths are required for the dynamic stem."""
        if self.use_dynamic_encoder:
            if wavelengths is None:
                msg = "wavelengths are required for the wavelength stem"
                raise ValueError(msg)
            feats, skip = self.encoder(img, wavelengths)
            encoded = [skip, *feats]
        else:
            encoded = self.encoder(img)
        out, aux = self.decoder(encoded)
        out = fn.interpolate(
            input=out,
            size=img.shape[2:],
            scale_factor=None,
            mode="bilinear",
            align_corners=False,
        )
        if aux is not None:
            aux = {
                k: fn.interpolate(
                    v,
                    size=img.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for k, v in aux.items()
            }
        return self.output_struct(out=out, aux=aux)


if __name__ == "__main__":
    model = SegFormerSegmentationModel()
    x = torch.randn(5, 3, 512, 512)
    outputs = model(x)
    # print(f"outputs.shape: {outputs.shape}")
