"""Mix-Transformer encoder."""

import math
import warnings
from functools import partial
from typing import Any

import torch
import torch.nn.functional as fn
from timm.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor, nn
from torch.utils import model_zoo

from geo_deep_learning.models.utils import patch_first_conv


class EncoderMixin:
    """Encoder mixin."""

    _output_stride = 32

    @property
    def out_channels(self) -> list[int]:
        """Return channels dimensions for each tensor of forward output of encoder."""
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self) -> int:
        """Return output stride."""
        return min(self._output_stride, 2**self._depth)

    def set_in_channels(self, in_channels: int, *, pretrained: bool = True) -> None:
        """Change first convolution channels."""
        expected_in_channels = 3
        if in_channels == expected_in_channels:
            return

        self._in_channels = in_channels
        expected_out_channels = 3
        if self._out_channels[0] == expected_out_channels:
            self._out_channels = (in_channels, *self._out_channels[1:])

        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)


class Mlp(nn.Module):
    """MLP module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize MLP module."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    """Attention module."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int = 8,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        *,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize Attention module."""
        super().__init__()
        if dim % num_heads != 0:
            msg = f"dim {dim} should be divided by num_heads {num_heads}."
            raise ValueError(msg)

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass."""
        batch_size, num_patches, channels = x.shape
        q = (
            self.q(x)
            .reshape(
                batch_size,
                num_patches,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch_size, channels, h, w)
            x_ = self.sr(x_).reshape(batch_size, channels, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Block module."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        sr_ratio: int = 1,
        *,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize Block module."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        return x + self.drop_path(self.mlp(self.norm2(x), h, w))


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """Initialize OverlapPatchEmbed."""
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.h, self.w = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.h * self.w
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class MixVisionTransformer(nn.Module):
    """Mix-Transformer encoder."""

    def __init__(  # noqa: PLR0913
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dims: list[int] | None = None,
        num_heads: list[int] | None = None,
        mlp_ratios: list[float] | None = None,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        depths: list[int] | None = None,
        sr_ratios: list[int] | None = None,
        *,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize MixVisionTransformer."""
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        embed_dims = embed_dims or [64, 128, 256, 512]
        num_heads = num_heads or [1, 2, 4, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        depths = depths or [3, 4, 6, 3]
        sr_ratios = sr_ratios or [8, 4, 2, 1]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ],
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ],
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ],
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ],
        )
        self.norm4 = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained: str | None = None) -> None:
        """Initialize weights."""

    def reset_drop_path(self, drop_path_rate: float) -> None:
        """Reset drop path."""
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self) -> None:
        """Freeze patch embedding."""
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self) -> dict[str, bool]:
        """No weight decay."""
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self) -> nn.Module:
        """Get classifier."""
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        """Reset classifier."""
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Forward features."""
        batch_size = x.shape[0]
        outs = []

        # stage 1
        x, h, w = self.patch_embed1(x)
        for _, blk in enumerate(self.block1):
            x = blk(x, h, w)
        x = self.norm1(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, h, w = self.patch_embed2(x)
        for _, blk in enumerate(self.block2):
            x = blk(x, h, w)
        x = self.norm2(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, h, w = self.patch_embed3(x)
        for _, blk in enumerate(self.block3):
            x = blk(x, h, w)
        x = self.norm3(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, h, w = self.patch_embed4(x)
        for _, blk in enumerate(self.block4):
            x = blk(x, h, w)
        x = self.norm4(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass."""
        return self.forward_features(x)


class DWConv(nn.Module):
    """Depthwise Convolution."""

    def __init__(self, dim: int = 768) -> None:
        """Initialize DWConv."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass."""
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).view(batch_size, channels, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


# Taken from segmentation models pytorch, adapted to support arbitrary input channels
class MixVisionTransformerEncoder(MixVisionTransformer, EncoderMixin):
    """MixVisionTransformer encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 5,
        **kwargs: object,
    ) -> None:
        """Initialize MixVisionTransformer encoder."""
        super().__init__(in_chans=in_channels, **kwargs)
        self._out_channels = out_channels
        self._depth = depth

    def make_dilated(self) -> None:
        """Make dilated."""
        msg = "MixVisionTransformer encoder does not support dilated mode"
        raise ValueError(msg)

    def set_in_channels(self, in_channels: int) -> None:
        """Set in channels."""
        expected_channels = 3
        if in_channels != expected_channels:
            pass

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass."""
        return self.forward_features(x)[: self._depth - 1]

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load state dict."""
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        return super().load_state_dict(state_dict)


def get_pretrained_cfg(name: str) -> dict[str, Any]:
    """Get pretrained config."""
    return {
        "url": f"https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/{name}.pth",
        "input_space": "RGB",
        "input_size": [3, 224, 224],
        "input_range": [0, 1],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


mix_transformer_encoders = {
    "mit_b0": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b0"),
        },
        "params": {
            "out_channels": (3, 0, 32, 64, 160, 256),
            "embed_dims": [32, 64, 160, 256],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b1": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b1"),
        },
        "params": {
            "out_channels": (3, 0, 64, 128, 320, 512),
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b2": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b2"),
        },
        "params": {
            "out_channels": (3, 0, 64, 128, 320, 512),
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 4, 6, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b3": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b3"),
        },
        "params": {
            "out_channels": (3, 0, 64, 128, 320, 512),
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 4, 18, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b4": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b4"),
        },
        "params": {
            "out_channels": (3, 0, 64, 128, 320, 512),
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 8, 27, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b5": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b5"),
        },
        "params": {
            "out_channels": (3, 0, 64, 128, 320, 512),
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "depths": [3, 6, 40, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
}


def get_encoder(
    name: str,
    in_channels: int = 3,
    depth: int = 5,
    weights: str | None = None,
    output_stride: int = 32,
) -> MixVisionTransformerEncoder:
    """Get encoder."""
    try:
        encoder_class = mix_transformer_encoders[name]["encoder"]
    except KeyError as err:
        msg = (
            f"Wrong encoder name `{name}`, supported encoders: "
            f"{list(mix_transformer_encoders.keys())}"
        )
        raise KeyError(msg) from err

    params = mix_transformer_encoders[name]["params"]
    params.update(in_channels=in_channels, depth=depth)
    encoder = encoder_class(**params)

    if weights is not None:
        expected_channels = 3
        if in_channels == expected_channels:
            try:
                settings = mix_transformer_encoders[name]["pretrained_settings"][
                    weights
                ]
            except KeyError as err:
                msg = (
                    f"Wrong pretrained weights `{weights}` for encoder `{name}`. "
                    f"Available options are: "
                    f"{list(mix_transformer_encoders[name]['pretrained_settings'].keys())}"
                )
                raise KeyError(msg) from err
            encoder.load_state_dict(model_zoo.load_url(settings["url"]))
        else:
            msg = (
                "MixVisionTransformer encoder does not support pretrained weights "
                "for non-RGB input channels"
            )
            warnings.warn(msg, stacklevel=2)

    encoder.set_in_channels(in_channels)
    expected_stride = 32
    if output_stride != expected_stride:
        encoder.make_dilated(output_stride)

    return encoder


class DynamicChannelEmbed(nn.Module):
    """Dynamic channel patch embedding."""

    def __init__(  # noqa: PLR0913
        self,
        patch_size: int = 7,
        stride: int = 4,
        embed_dim: int = 64,  # b0=32, b1:b5=64
        num_heads: int = 4,  # b0=2, b1:b5=4
        drop: float = 0.1,
        max_channels: int = 256,
        gate_floor: float = 0.5,
        mod_alpha: float = 0.5,
        bottleneck_channels: int = 0,
        attn_drop: float | None = None,
    ) -> None:
        """Initialize DynamicChannelEmbed."""
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = "embed_dim must be divisible by num_heads"
            raise ValueError(msg)

        hidden_dim = embed_dim * 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gate_floor = gate_floor
        self.mod_alpha = mod_alpha
        self.max_channels = max_channels

        if bottleneck_channels > 0:
            self.bottleneck = nn.LazyConv2d(
                out_channels=bottleneck_channels,
                kernel_size=1,
                bias=True,
            )
        else:
            self.bottleneck = nn.Identity()

        self.spatial_pw = nn.Conv2d(1, embed_dim, kernel_size=1, bias=True)
        self.spatial_dw = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            groups=embed_dim,
            bias=True,
        )

        self.channel_embed = nn.Embedding(max_channels, hidden_dim)
        self.weight_gen = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh(),
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.pre_gate_norm = nn.LayerNorm(embed_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )

        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(drop)
        self.attn_drop = nn.Dropout(drop if attn_drop is None else attn_drop)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """
        Forward pass.

        x: [B, C, H, W]
        returns: tokens [B, H'*W', D], h_out, w_out
        """
        x = self.bottleneck(x)
        batch_size, channels, height, width = x.shape
        device = x.device

        if self.max_channels < channels:
            msg = (
                f"Input channels {channels} > max_channels {self.max_channels}. "
                "Increase max_channels or use bottleneck_channels to reduce C."
            )
            raise ValueError(msg)

        # Vectorized per-channel spatial features
        xc = x.reshape(batch_size * channels, 1, height, width)
        feat = self.spatial_pw(xc)
        feat = self.spatial_dw(feat)  # [B*C, D, H', W']
        _, emb_dim, h_out, w_out = feat.shape
        feat = feat.view(
            batch_size,
            channels,
            emb_dim,
            h_out,
            w_out,
        )  # [B, C, D, H', W']

        # Channel modulation (learnable, index-based)
        ch_idx = torch.arange(channels, device=device)
        ch_emb = self.channel_embed(ch_idx)  # [C, hidden_dim]
        ch_mod = self.weight_gen(ch_emb)  # [C, D]
        feat = feat * (1.0 + self.mod_alpha * ch_mod.view(1, channels, emb_dim, 1, 1))

        # Channel tokens
        ch_tok = feat.mean(dim=(3, 4))  # [B, C, D]

        # Channel self-attention
        q, k, v = self.qkv(ch_tok).chunk(3, dim=-1)
        d_head = emb_dim // self.num_heads
        q = q.view(
            batch_size,
            channels,
            self.num_heads,
            d_head,
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            channels,
            self.num_heads,
            d_head,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            channels,
            self.num_heads,
            d_head,
        ).transpose(1, 2)
        attn = fn.scaled_dot_product_attention(q, k, v)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                channels,
                emb_dim,
            )
        )
        attn = self.attn_drop(attn)

        # Bounded gating (prevents channel collapse)
        gated = self.pre_gate_norm(attn)
        logits = self.gate_mlp(gated).squeeze(-1)  # [B, C]
        gate = torch.sigmoid(logits)
        gate = self.gate_floor + (1.0 - self.gate_floor) * gate

        # Aggregate spatial maps
        out_map = torch.einsum("bcdhw,bc->bdhw", feat, gate)

        # Tokens for transformer
        tokens = out_map.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        tokens = self.drop(tokens)
        tokens = self.norm(tokens)

        return tokens, h_out, w_out


class DynamicMixTransformer(nn.Module, EncoderMixin):
    """Dynamic MixVisionTransformer, handles arbitrary channel counts."""

    def __init__(
        self,
        encoder: str = "mit_b0",
        in_channels: int = 3,
        weights: str | None = None,
    ) -> None:
        """Initialize DynamicMixTransformer."""
        super().__init__()
        base_encoder = get_encoder(
            name=encoder,
            in_channels=in_channels,
            weights=weights,
        )
        self._out_channels = base_encoder._out_channels  # noqa: SLF001
        self._depth = base_encoder._depth  # noqa: SLF001

        self.dynamic_patch_embed1 = DynamicChannelEmbed(
            patch_size=7,
            stride=4,
            embed_dim=base_encoder.patch_embed1.proj.out_channels,
        )
        self.patch_embed2 = base_encoder.patch_embed2
        self.patch_embed3 = base_encoder.patch_embed3
        self.patch_embed4 = base_encoder.patch_embed4

        self.block1 = base_encoder.block1
        self.block2 = base_encoder.block2
        self.block3 = base_encoder.block3
        self.block4 = base_encoder.block4

        self.norm1 = base_encoder.norm1
        self.norm2 = base_encoder.norm2
        self.norm3 = base_encoder.norm3
        self.norm4 = base_encoder.norm4

    def set_in_channels(self, in_channels: int) -> None:
        """Set in channels."""

    def forward_features(self, x: Tensor) -> list[Tensor]:
        """Forward features pass."""
        batch_size = x.shape[0]
        outs = []
        x, h, w = self.dynamic_patch_embed1(x)
        for blk in self.block1:
            x = blk(x, h, w)
        x = self.norm1(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, h, w = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, h, w)
        x = self.norm2(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, h, w = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, h, w)
        x = self.norm3(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, h, w = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, h, w)
        x = self.norm4(x)
        x = x.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward pass."""
        return self.forward_features(x)
