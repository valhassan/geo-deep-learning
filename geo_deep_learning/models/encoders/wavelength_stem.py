"""Wavelength-conditioned MiT stem."""

import torch
import torch.nn.functional as fn
from torch import Tensor, nn

from geo_deep_learning.models.encoders.dofa_v2 import (
    FCResLayer,
    TransformerWeightGenerator,
    position_embedding,
)
from geo_deep_learning.models.encoders.mix_transformer import get_encoder

_WAVE_DIM = 128
_PATCH = 7
_STRIDE = 4
_SKIP_KERNEL = 3
_SKIP_STRIDE = 2
_SCALE = 0.01


def _sensor_wavelengths(wavelengths: Tensor, channels: int) -> Tensor:
    """Return one wavelength vector (C,). A batch uses the first row, as DOFA does."""
    lam = wavelengths[0] if wavelengths.ndim == 2 else wavelengths
    if lam.ndim != 1 or lam.shape[0] != channels:
        msg = (
            f"wavelengths shape {tuple(wavelengths.shape)} "
            f"does not match channels {channels}"
        )
        raise ValueError(msg)
    return lam


class WavelengthStem(nn.Module):
    """DOFA kernel generator at MiT stage-1 stride, plus a stride-2 skip."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize the stem. embed_dim is MiT stage-1 width (32 or 64)."""
        super().__init__()
        self.embed_dim = embed_dim
        self.wave_dim = _WAVE_DIM
        self.fclayer = FCResLayer(_WAVE_DIM)
        self.stage_gen = TransformerWeightGenerator(
            input_dim=_WAVE_DIM,
            output_dim=_PATCH * _PATCH * embed_dim,
            embed_dim=embed_dim,
        )
        self.skip_gen = TransformerWeightGenerator(
            input_dim=_WAVE_DIM,
            output_dim=_SKIP_KERNEL * _SKIP_KERNEL * embed_dim,
            embed_dim=embed_dim,
        )
        self.token_norm = nn.LayerNorm(embed_dim)
        self.skip_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Match DOFA: Xavier kernels, then a 0.01 scale applied at conv time."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def _encode(self, wavelengths: Tensor, channels: int) -> Tensor:
        lam = _sensor_wavelengths(wavelengths, channels).to(dtype=torch.float32)
        waves = position_embedding(self.wave_dim, lam * 1000)
        return self.fclayer(waves)

    def _conv(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        kernel: int,
        stride: int,
    ) -> Tensor:
        """Dense conv. weight is (C, kernel*kernel*E) from the generator."""
        filters = weight.view(x.shape[1], kernel, kernel, self.embed_dim)
        filters = filters.permute(3, 0, 1, 2).mul(_SCALE).to(dtype=x.dtype)
        bias = bias.view(self.embed_dim).mul(_SCALE).to(dtype=x.dtype)
        return fn.conv2d(
            x,
            filters,
            bias=bias,
            stride=stride,
            padding=kernel // 2,
        )

    @staticmethod
    def _norm(feat: Tensor, norm: nn.LayerNorm) -> Tensor:
        return norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
    ) -> tuple[Tensor, int, int, Tensor]:
        """Return stage-1 tokens (B, L, E), height, width, and the stride-2 skip."""
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            waves = self._encode(wavelengths, x.shape[1])
            stage_w, stage_b = self.stage_gen(waves)
            skip_w, skip_b = self.skip_gen(waves)

        feat = self._norm(
            self._conv(x, stage_w, stage_b, _PATCH, _STRIDE),
            self.token_norm,
        )
        skip = self._norm(
            self._conv(x, skip_w, skip_b, _SKIP_KERNEL, _SKIP_STRIDE),
            self.skip_norm,
        )
        height, width = feat.shape[-2:]
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, height, width, skip


class WavelengthMixTransformer(nn.Module):
    """MiT with a DOFA wavelength stem. Stages 2-4 load from ImageNet when asked."""

    def __init__(
        self,
        encoder: str = "mit_b5",
        weights: str | None = None,
    ) -> None:
        """Initialize from an ImageNet MiT variant. Stage-1 stem is new."""
        super().__init__()
        base = get_encoder(name=encoder, in_channels=3, weights=weights)
        embed_dim = base.patch_embed1.proj.out_channels
        self.stem = WavelengthStem(embed_dim=embed_dim)
        self.block1 = base.block1
        self.block2 = base.block2
        self.block3 = base.block3
        self.block4 = base.block4
        self.patch_embed2 = base.patch_embed2
        self.patch_embed3 = base.patch_embed3
        self.patch_embed4 = base.patch_embed4
        self.norm1 = base.norm1
        self.norm2 = base.norm2
        self.norm3 = base.norm3
        self.norm4 = base.norm4

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        """Return the four MiT stages and the stride-2 skip."""
        batch = x.shape[0]
        outs: list[Tensor] = []
        tokens, height, width, skip = self.stem(x, wavelengths)
        for blk in self.block1:
            tokens = blk(tokens, height, width)
        tokens = self.norm1(tokens)
        feat = tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        outs.append(feat.contiguous())

        x, height, width = self.patch_embed2(feat)
        for blk in self.block2:
            x = blk(x, height, width)
        x = self.norm2(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, height, width = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, height, width)
        x = self.norm3(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, height, width = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, height, width)
        x = self.norm4(x)
        x = x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs, skip
