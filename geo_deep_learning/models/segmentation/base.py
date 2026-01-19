"""Base segmentation model."""

from torch import Tensor, nn


class BaseSegmentationModel(nn.Module):
    """Base segmentation model."""

    def __init__(  # noqa: PLR0913
        self,
        encoder: nn.Module | None = None,
        neck: nn.Module | None = None,
        decoder: nn.Module | None = None,
        head: nn.Module | None = None,
        output_struct: nn.Module | None = None,
        auxilary_head: nn.Module | None = None,
    ) -> None:
        """Initialize base segmentation model."""
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.auxilary_head = auxilary_head
        self.head = head
        self.output_struct = output_struct

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        aux = None
        if self.auxilary_head:
            aux = self.auxilary_head(x)
        x = self.head(x)
        return self.output_struct(out=x, aux=aux)

    def _freeze_layers(self, layers: list[str]) -> None:
        """Freeze layers."""
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                param.requires_grad = False
