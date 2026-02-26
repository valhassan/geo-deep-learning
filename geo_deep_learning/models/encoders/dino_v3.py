"""DINOv3 Backbone."""

# Adapted from https://github.com/facebookresearch/dinov3/

import logging
import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as fn
import torch.nn.init
import torch.utils.checkpoint as cp
from torch import Tensor, nn

from geo_deep_learning.models.utils import MSDeformAttn

from .dinov3_layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RMSNorm,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
    named_apply,
)
from .dora import DoRAQKVWrapper

logger = logging.getLogger("dinov3")


ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = "") -> None:  # noqa: ARG001
    """Initialize weights of a module."""
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    """DINO Vision Transformer."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: str | torch.device | None = None,
        **ignored_kwargs: object,
    ) -> None:
        """Initialize DINO Vision Transformer."""
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning("Ign`ored kwargs: %s", ignored_kwargs)
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device),
            )
        logger.info("using base=%s for rope new", pos_embed_rope_base)
        logger.info("using min_period=%s for rope new", pos_embed_rope_min_period)
        logger.info("using max_period=%s for rope new", pos_embed_rope_max_period)
        logger.info(
            "using normalize_coords=%s for rope new",
            pos_embed_rope_normalize_coords,
        )
        logger.info("using shift_coords=%s for rope new", pos_embed_rope_shift_coords)
        logger.info(
            "using rescale_coords=%s for rope new",
            pos_embed_rope_rescale_coords,
        )
        logger.info("using jitter_coords=%s for rope new", pos_embed_rope_jitter_coords)
        logger.info("using dtype=%s for rope new", pos_embed_rope_dtype)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info("using %s layer as FFN", ffn_layer)
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self) -> None:
        """Initialize weights."""
        self.rope_embed._init_weights()  # noqa: SLF001
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def inject_dora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout_rate: float = 0.05,
    ) -> None:
        """Inject DoRA wrappers into the QKV layers."""
        for block in self.blocks:
            # Wrap the existing qkv layer
            block.attn.qkv = DoRAQKVWrapper(
                qkv=block.attn.qkv,
                r=r,
                alpha=alpha,
                dropout_rate=dropout_rate,
            )

    def prepare_tokens_with_masks(
        self,
        x: Tensor,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, tuple[int]]:
        """Prepare tokens with masks."""
        x = self.patch_embed(x)
        b, h, w, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(x.dtype).unsqueeze(0),
                x,
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(b, -1, -1),
                storage_tokens.expand(b, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (h, w)

    def forward_features_list(
        self,
        x_list: list[Tensor],
        masks_list: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        """Forward features list."""
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list, strict=False):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(height=H, width=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list, strict=False)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(
                        x[:, : self.n_storage_tokens + 1],
                    )
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                },
            )
        return output

    def forward_features(
        self,
        x: Tensor | list[Tensor],
        masks: Tensor | None = None,
    ) -> list[dict[str, Tensor]]:
        """Forward features."""
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(
        self,
        x: Tensor,
        n: int = 1,
    ) -> list[Tensor]:
        """Get intermediate layers not chunked."""
        x, (h, w) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(height=h, width=w)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), (  # noqa: S101
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output

    def get_intermediate_layers(  # noqa: PLR0913
        self,
        x: torch.Tensor,
        *,
        n: int | Sequence = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, ...]]:
        """Get intermediate layers."""
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(
                        torch.cat((x_norm_cls_reg, x_norm_patch), dim=1),
                    )
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            b, _, h, w = x.shape
            outputs = [
                out.reshape(b, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens, strict=False))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens, strict=False))
        # Both return_class_token and return_extra_tokens are True
        return tuple(zip(outputs, class_tokens, extra_tokens, strict=False))

    def forward(
        self,
        *args: object,
        is_training: bool = False,
        **kwargs: object,
    ) -> list[dict[str, Tensor]] | Tensor:
        """Forward pass."""
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        return self.head(ret["x_norm_clstoken"])


def vit_large(
    patch_size: int = 16,
    pretrained: bool = True,  # noqa: FBT001, FBT002
    **kwargs: object,
) -> DinoVisionTransformer:
    """Vit large. supports weight "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"."""
    model = DinoVisionTransformer(
        patch_size=patch_size,
        pos_embed_rope_rescale_coords=2.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=True,
        **kwargs,
    )
    model.init_weights()
    if pretrained:
        model_url = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        hub_dir = torch.hub.get_dir()
        ckpt_path = Path(hub_dir) / "checkpoints" / model_url
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))
            logger.info("Loaded pretrained weights %s", model_url)
        else:
            logger.warning("Pretrained weights %s not found", model_url)

    return model


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:  # noqa: FBT001, FBT002
    """Drop path."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialize DropPath."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return drop_path(x, self.drop_prob, self.training)


def get_reference_points(spatial_shapes: Tensor, device: str | torch.device) -> Tensor:
    """Get reference points."""
    reference_points_list = []
    for _, (h_, w_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, h_ - 0.5, h_, dtype=torch.float32, device=device),
            torch.linspace(0.5, w_ - 0.5, w_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / h_
        ref_x = ref_x.reshape(-1)[None] / w_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None]


def deform_inputs(x: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Deform inputs."""
    _, _, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]),
    )
    reference_points = get_reference_points(
        [(h // patch_size, w // patch_size)],
        x.device,
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor(
        [(h // patch_size, w // patch_size)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]),
    )
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        x.device,
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    """ConvFFN."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        """Initialize ConvFFN."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class DWConv(nn.Module):
    """DWConv."""

    def __init__(self, dim: int = 768) -> None:
        """Initialize DWConv."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Forward pass."""
        b, n, c = x.shape
        n = n // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(b, c, h * 2, w * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(b, c, h, w).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(b, c, h // 2, w // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        return torch.cat([x1, x2, x3], dim=1)


class Extractor(nn.Module):
    """Extractor."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        with_cffn: bool = True,  # noqa: FBT001, FBT002
        cffn_ratio: float = 0.25,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # noqa: B008
        with_cp: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize Extractor."""
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=dim,
                hidden_features=int(dim * cffn_ratio),
                drop=drop,
            )
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(  # noqa: PLR0913
        self,
        query: Tensor,
        reference_points: Tensor,
        feat: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        h: int,
        w: int,
    ) -> Tensor:
        """Forward pass."""

        def _inner_forward(query: Tensor, feat: Tensor) -> Tensor:
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), h, w))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlockWithCls(nn.Module):
    """InteractionBlockWithCls."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),  # noqa: B008
        drop: float = 0.0,
        drop_path: float = 0.0,
        with_cffn: bool = True,  # noqa: FBT001, FBT002
        cffn_ratio: float = 0.25,
        init_values: float = 0.0,  # noqa: ARG002
        deform_ratio: float = 1.0,
        extra_extractor: bool = False,  # noqa: FBT001, FBT002
        with_cp: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize InteractionBlockWithCls."""
        super().__init__()
        self.extractor = Extractor(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
        )
        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        with_cp=with_cp,
                    )
                    for _ in range(2)
                ],
            )
        else:
            self.extra_extractors = None

    def forward(  # noqa: PLR0913
        self,
        x: Tensor,
        c: Tensor,
        cls: Tensor,
        deform_inputs1: Tensor,  # noqa: ARG002
        deform_inputs2: Tensor,
        h_c: int,
        w_c: int,
        h_toks: int,  # noqa: ARG002
        w_toks: int,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            h=h_c,
            w=w_c,
        )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    h=h_c,
                    w=w_c,
                )
        return x, c, cls


class SpatialPriorModule(nn.Module):
    """SpatialPriorModule."""

    def __init__(
        self,
        inplanes: int = 64,
        embed_dim: int = 384,
        with_cp: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize SpatialPriorModule."""
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes,
                    inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes,
                    inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ],
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ],
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ],
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ],
        )
        self.fc1 = nn.Conv2d(
            inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            2 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fc3 = nn.Conv2d(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fc4 = nn.Conv2d(
            4 * inplanes,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass."""

        def _inner_forward(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class DINOv3Adapter(nn.Module):
    """DINOv3 Multi-Scale Adapter."""

    def __init__(  # noqa: PLR0913
        self,
        backbone: nn.Module,
        interaction_indexes: list[int] = [4, 11, 17, 23],  # noqa: B006
        pretrain_size: int = 512,
        conv_inplane: int = 64,
        n_points: int = 4,
        deform_num_heads: int = 16,
        drop_path_rate: float = 0.3,
        init_values: float = 0.0,
        with_cffn: bool = True,  # noqa: FBT001, FBT002
        cffn_ratio: float = 0.25,
        deform_ratio: float = 0.5,
        add_vit_feature: bool = True,  # noqa: FBT001, FBT002
        use_extra_extractor: bool = True,  # noqa: FBT001, FBT002
        with_cp: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize DINOv3 Multi-Scale Adapter."""
        super().__init__()
        self.backbone = backbone
        # Important: we freeze the backbone
        self.backbone.requires_grad_(requires_grad=False)

        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        # print("embed dim", embed_dim)
        # print("interaction_indexes", self.interaction_indexes)
        # print("patch_size", self.patch_size)

        block_fn = InteractionBlockWithCls
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane,
            embed_dim=embed_dim,
            with_cp=False,
        )
        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (i == len(self.interaction_indexes) - 1) and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(self.interaction_indexes))
            ],
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed: Tensor, h: int, w: int) -> Tensor:
        """Get positional embedding."""
        pos_embed = pos_embed.reshape(
            1,
            self.pretrain_size[0] // self.patch_size,
            self.pretrain_size[1] // self.patch_size,
            -1,
        ).permute(0, 3, 1, 2)
        return (
            fn.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            .reshape(1, -1, h * w)
            .permute(0, 2, 1)
        )

    def _init_deform_weights(self, m: nn.Module) -> None:
        """Initialize deform weights."""
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()  # noqa: SLF001

    def _add_level_embed(
        self,
        c2: Tensor,
        c3: Tensor,
        c4: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Add level embedding."""
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass."""
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c = torch.cat([c2, c3, c4], dim=1)
        h_c, w_c = x.shape[2] // 16, x.shape[3] // 16
        h_toks, w_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, _, _, _ = x.shape

        with torch.autocast("cuda", torch.bfloat16), torch.no_grad():
            all_layers = self.backbone.get_intermediate_layers(
                x,
                n=self.interaction_indexes,
                return_class_token=True,
            )

        x_for_shape, _ = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        cls, x = (
            x[
                :,
                :1,
            ],
            x[
                :,
                5:,
            ],
        )

        outs: list[Tensor] = []
        for i, layer in enumerate(self.interactions):
            x, cls = all_layers[i]
            _, c, _ = layer(
                x,
                c,
                cls,
                deform_inputs1,
                deform_inputs2,
                h_c,
                w_c,
                h_toks,
                w_toks,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, h_toks, w_toks).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, h_c * 2, w_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, h_c, w_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, h_c // 2, w_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = fn.interpolate(
                x1,
                size=(4 * h_c, 4 * w_c),
                mode="bilinear",
                align_corners=False,
            )
            x2 = fn.interpolate(
                x2,
                size=(2 * h_c, 2 * w_c),
                mode="bilinear",
                align_corners=False,
            )
            x3 = fn.interpolate(
                x3,
                size=(1 * h_c, 1 * w_c),
                mode="bilinear",
                align_corners=False,
            )
            x4 = fn.interpolate(
                x4,
                size=(h_c // 2, w_c // 2),
                mode="bilinear",
                align_corners=False,
            )
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return {"1": f1, "2": f2, "3": f3, "4": f4}


# if __name__ == "__main__":
# backbone = vit_large()
# model = DINOv3Adapter(backbone=backbone)
# model = vit_large()
# model.init_weights()
# weights = ""
# state_dict = torch.load(weights, weights_only=True)
# # print(state_dict.keys())
# # print(model.state_dict().keys())
# model.load_state_dict(state_dict, strict=True)
# image = torch.randn(1, 3, 224, 224)
# output = model(image)
# for key, value in output.items():
#     print(key, value.shape)
