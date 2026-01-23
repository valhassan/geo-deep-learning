"""SegFormer segmentation model."""

import torch
import torch.nn.functional as fn

from geo_deep_learning.models.decoders.segformer_mlp import Decoder
from geo_deep_learning.models.encoders.mix_transformer import (
    DynamicMixTransformer,
    get_encoder,
)
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
        if use_dynamic_encoder:
            self.encoder = DynamicMixTransformer(
                encoder=encoder,
                weights=weights,
            )
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
        )
        self.output_struct = SegmentationOutput

    def forward(self, img: torch.Tensor) -> SegmentationOutput:
        """Forward pass."""
        x = self.encoder(img)
        out, aux = self.decoder(x)
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
