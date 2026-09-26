"""Wavelength-conditioned MiT stem."""

import torch
import torch.nn.functional as fn
from torch import Tensor, nn

from geo_deep_learning.models.encoders.dofa_v2 import position_embedding
from geo_deep_learning.models.encoders.mix_transformer import get_encoder

_WAVE_DIM = 128
_PATCH = 7
_STRIDE = 4
_SKIP_KERNEL = 3
_SKIP_STRIDE = 2
_NUM_QUERIES = 4


def _batch_wavelengths(
    wavelengths: Tensor,
    batch: int,
    channels: int,
    device: torch.device,
) -> Tensor:
    """Return wavelengths as (B, C) in µm."""
    lam = wavelengths.to(device=device, dtype=torch.float32)
    if lam.ndim == 1:
        if lam.shape[0] != channels:
            msg = f"wavelengths length {lam.shape[0]} != channels {channels}"
            raise ValueError(msg)
        return lam.expand(batch, channels)
    if lam.ndim == 2 and lam.shape == (batch, channels):
        return lam
    if lam.ndim == 2 and lam.shape == (1, channels):
        return lam.expand(batch, channels)
    msg = f"wavelengths shape {tuple(lam.shape)} does not match {(batch, channels)}"
    raise ValueError(msg)


def _depthwise(
    x: Tensor,
    weight: Tensor,
    kernel: int,
    stride: int,
    padding: int,
) -> Tensor:
    """Per-sample depthwise conv. weight is (B, C, kernel*kernel)."""
    patches = fn.unfold(x, kernel_size=kernel, padding=padding, stride=stride)
    batch, _, length = patches.shape
    channels = x.shape[1]
    patches = patches.view(batch, channels, kernel * kernel, length)
    spatial = torch.einsum("bckl,bck->bcl", patches, weight)
    height = (x.shape[-2] + 2 * padding - kernel) // stride + 1
    width = (x.shape[-1] + 2 * padding - kernel) // stride + 1
    return spatial.view(batch, channels, height, width)


class WavelengthAxialStem(nn.Module):
    """Stage-1 stem: wavelength kernels, axial channel collapse, stride-2 skip."""

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = _NUM_QUERIES,
    ) -> None:
        """Initialize the stem. embed_dim is MiT stage-1 width (32 or 64)."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.wave_dim = _WAVE_DIM

        self.wave_mlp = nn.Sequential(
            nn.Linear(_WAVE_DIM, _WAVE_DIM),
            nn.GELU(),
            nn.Linear(_WAVE_DIM, _WAVE_DIM),
            nn.GELU(),
        )
        self.kernel_head = nn.Linear(_WAVE_DIM, _PATCH * _PATCH)
        self.skip_kernel_head = nn.Linear(_WAVE_DIM, _SKIP_KERNEL * _SKIP_KERNEL)
        self.wave_to_embed = nn.Linear(_WAVE_DIM, embed_dim)
        self.queries = nn.Parameter(torch.empty(num_queries, embed_dim))
        self.direction = nn.Parameter(torch.empty(embed_dim))
        self.out_proj = nn.Conv2d(num_queries * embed_dim, embed_dim, kernel_size=1)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.skip_norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Small kernels so the spectral mix starts near uniform."""
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.direction, std=0.02)
        nn.init.normal_(self.kernel_head.weight, std=0.02)
        nn.init.zeros_(self.kernel_head.bias)
        nn.init.normal_(self.skip_kernel_head.weight, std=0.02)
        nn.init.zeros_(self.skip_kernel_head.bias)

    def _encode_waves(self, lam: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = (lam * 1000).reshape(-1)
        feat = position_embedding(self.wave_dim, flat)
        feat = feat.view(*lam.shape, self.wave_dim)
        hidden = self.wave_mlp(feat)
        wave = self.wave_to_embed(hidden)
        return self.kernel_head(hidden), self.skip_kernel_head(hidden), wave

    def _collapse(self, spatial: Tensor, wave: Tensor) -> Tensor:
        """Mix bands. spatial (B, C, H, W), wave (B, C, E) -> (B, E, H, W)."""
        batch, _, height, width = spatial.shape
        queries = self.queries.to(dtype=spatial.dtype)
        direction = self.direction.to(dtype=spatial.dtype)
        scale = self.embed_dim**-0.5
        qw = torch.einsum("qe,bce->bqc", queries, wave) * scale
        qd = torch.einsum("qe,e->q", queries, direction) * scale
        score = qw[:, :, :, None, None] + spatial[:, None] * qd.view(1, -1, 1, 1, 1)
        attn = score.softmax(dim=2)
        mixed = torch.einsum("bqchw,bce->bqehw", attn, wave)
        mixed_s = torch.einsum("bqchw,bchw->bqhw", attn, spatial)
        mixed = mixed + mixed_s[:, :, None] * direction.view(1, 1, -1, 1, 1)
        mixed = mixed.reshape(batch, self.num_queries * self.embed_dim, height, width)
        return self.out_proj(mixed)

    def forward(
        self,
        x: Tensor,
        wavelengths: Tensor,
    ) -> tuple[Tensor, int, int, Tensor]:
        """Return stage-1 tokens (B, L, E), height, width, and stride-2 skip."""
        batch, channels, _, _ = x.shape
        lam = _batch_wavelengths(wavelengths, batch, channels, x.device)
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            kernel, skip_kernel, wave = self._encode_waves(lam)
        kernel = kernel.to(dtype=x.dtype)
        skip_kernel = skip_kernel.to(dtype=x.dtype)
        wave = wave.to(dtype=x.dtype)

        spatial = _depthwise(x, kernel, _PATCH, _STRIDE, padding=_PATCH // 2)
        skip_spatial = _depthwise(
            x,
            skip_kernel,
            _SKIP_KERNEL,
            _SKIP_STRIDE,
            padding=_SKIP_KERNEL // 2,
        )
        feat = self._collapse(spatial, wave)
        skip = self._collapse(skip_spatial, wave)
        feat = self.token_norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skip = self.skip_norm(skip.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        height, width = feat.shape[-2:]
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, height, width, skip


class WavelengthMixTransformer(nn.Module):
    """MiT with a wavelength stem. Stages 2-4 load from ImageNet when asked."""

    def __init__(
        self,
        encoder: str = "mit_b5",
        weights: str | None = None,
    ) -> None:
        """Initialize from an ImageNet MiT variant. Stage-1 stem is new."""
        super().__init__()
        base = get_encoder(name=encoder, in_channels=3, weights=weights)
        embed_dim = base.patch_embed1.proj.out_channels
        self.stem = WavelengthAxialStem(embed_dim=embed_dim)
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
