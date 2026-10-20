"""DINOv3 Layers."""

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as fn
from torch import Tensor, nn


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,  # noqa: FBT001, FBT002
    include_root: bool = False,  # noqa: FBT001, FBT002
) -> nn.Module:
    """Named apply."""
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name_str, child_module in module.named_children():
        child_name = f"{name}.{child_name_str}" if name else child_name_str
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def make_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """Make 2-tuple."""
    if isinstance(x, tuple):
        compare_val = 2
        assert len(x) == compare_val  # noqa: S101
        return x

    assert isinstance(x, int)  # noqa: S101
    return (x, x)


def rope_rotate_half(x: Tensor) -> Tensor:
    """Rope rotate half."""
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Rope apply."""
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


def cat_keep_shapes(x_list: list[Tensor]) -> tuple[Tensor, list[tuple[int]], list[int]]:
    """Cat keep shapes."""
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(
    flattened: Tensor,
    shapes: list[tuple[int]],
    num_tokens: list[int],
) -> list[Tensor]:
    """Uncat with shapes."""
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [
        shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes
    ]
    return [
        o.reshape(shape)
        for o, shape in zip(outputs_splitted, shapes_adjusted, strict=False)
    ]


class LayerScale(nn.Module):
    """LayerScale."""

    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
    ) -> None:
        """Initialize LayerScale."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ListForwardMixin:
    """List forward mixin."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        raise NotImplementedError

    def forward_list(self, x_list: list[Tensor]) -> list[Tensor]:
        """Forward list pass."""
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    """MLP."""

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
    ) -> None:
        """Initialize MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SwiGLUFFN(nn.Module, ListForwardMixin):
    """SwiGLU FFN."""

    def __init__(  # noqa: PLR0913
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,  # noqa: FBT001, FBT002
        align_to: int = 8,
        device: torch.device | None = None,
    ) -> None:
        """Initialize SwiGLU FFN."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(
            in_features,
            swiglu_hidden_features,
            bias=bias,
            device=device,
        )
        self.w2 = nn.Linear(
            in_features,
            swiglu_hidden_features,
            bias=bias,
            device=device,
        )
        self.w3 = nn.Linear(
            swiglu_hidden_features,
            out_features,
            bias=bias,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = fn.silu(x1) * x2
        return self.w3(hidden)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D).

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.

    """

    def __init__(  # noqa: PLR0913
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize PatchEmbed."""
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid_size = (
            image_hw[0] // patch_hw[0],
            image_hw[1] // patch_hw[1],
        )

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_hw,
            stride=patch_hw,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        _, _, height, width = x.shape

        x = self.proj(x)  # B C H W
        height, width = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, height, width, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """Flops."""
        ho, wo = self.patches_resolution
        flops = (
            ho
            * wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += ho * wo * self.embed_dim
        return flops

    def reset_parameters(self) -> None:
        """Reset parameters."""
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class RMSNorm(nn.Module):
    """RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize RMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        """Reset parameters."""
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        """Norm."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RopePositionEmbedding(nn.Module):
    """RopePositionEmbedding."""

    def __init__(  # noqa: PLR0913
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize RopePositionEmbedding."""
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0  # noqa: S101
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            err_msg = "Either `base` or `min_period`+`max_period` must be provided."
            raise ValueError(err_msg)

        d_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.d_head = d_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because
        # we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(d_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, height: int, width: int) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_hw = max(height, width)
            coords_h = torch.arange(0.5, height, **dd) / max_hw  # [H]
            coords_w = torch.arange(0.5, width, **dd) / max_hw  # [W]
        elif self.normalize_coords == "min":
            min_hw = min(height, width)
            coords_h = torch.arange(0.5, height, **dd) / min_hw  # [H]
            coords_w = torch.arange(0.5, width, **dd) / min_hw  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, height, **dd) / height  # [H]
            coords_w = torch.arange(0.5, width, **dd) / width  # [W]
        else:
            err_msg = f"Unknown normalize_coords: {self.normalize_coords}"
            raise ValueError(err_msg)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"),
            dim=-1,
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords,
                self.shift_coords,
            )
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1]
        # by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1]
        # by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self) -> None:
        """Initialize weights."""
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.d_head // 4, device=device, dtype=dtype)
                / (self.d_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0,
                1,
                self.d_head // 4,
                device=device,
                dtype=dtype,
            )  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


class LinearKMaskedBias(nn.Linear):
    """LinearKMaskedBias."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize LinearKMaskedBias."""
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0  # noqa: S101
        if self.bias is not None:
            self.register_buffer(
                "bias_mask",
                torch.full_like(self.bias, fill_value=math.nan),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        masked_bias = (
            self.bias * self.bias_mask.to(self.bias.dtype)
            if self.bias is not None
            else None
        )
        return fn.linear(x, self.weight, masked_bias)


class SelfAttention(nn.Module):
    """SelfAttention."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        proj_bias: bool = True,  # noqa: FBT001, FBT002
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
    ) -> None:
        """Initialize SelfAttention."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(
        self,
        q: Tensor,
        k: Tensor,
        rope: Tensor | tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Apply rope."""
        # All operations will use the dtype of rope,
        # the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        num_tokens = q.shape[-2]
        prefix = num_tokens - sin.shape[-2]
        assert prefix >= 0  # noqa: S101
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor | None = None,
        rope: Tensor | None = None,
    ) -> Tensor:
        """Forward pass."""
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        return self.proj_drop(x)

    def forward_list(
        self,
        x_list: list[Tensor],
        attn_bias: Tensor | None = None,
        rope_list: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Forward list pass."""
        assert len(x_list) == len(rope_list)  # noqa: S101
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(
            zip(qkv_list, shapes, rope_list, strict=False),
        ):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(
        self,
        qkv: Tensor,
        attn_bias: Tensor | None = None,
        rope: Tensor | None = None,
    ) -> Tensor:
        """Compute attention."""
        assert attn_bias is None  # noqa: S101
        batch_size, num_tokens, _ = qkv.shape
        head_dim = self.qkv.in_features

        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            head_dim // self.num_heads,
        )
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([batch_size, num_tokens, head_dim])


class SelfAttentionBlock(nn.Module):
    """SelfAttentionBlock."""

    def __init__(  # noqa: PLR0913
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        proj_bias: bool = True,  # noqa: FBT001, FBT002
        ffn_bias: bool = True,  # noqa: FBT001, FBT002
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
    ) -> None:
        """Initialize SelfAttentionBlock."""
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(
        rope: tuple[Tensor, Tensor] | None,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim  # noqa: S101
        compare_ndim = 4
        if sin.ndim == compare_ndim:
            # If the rope embedding has a batch dimension
            # (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        # No batch dimension, do not index.
        return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope: Tensor | None = None) -> Tensor:
        """Forward pass."""
        # This is the reference implementation for a single tensor,
        # matching what is done below for a list.
        # We call the list op on [x] instead of this function.
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(
        self,
        x_list: list[Tensor],
        rope_list: list[Tensor] | None = None,
    ) -> list[Tensor]:
        """Forward list pass."""
        # This list operator concatenates the tokens from the list of inputs together
        # to save on the elementwise operations.
        # Torch-compile memory-planning allows hiding the overhead related to concat ops
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [
            max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list
        ]
        residual_scale_factors = [
            b / sample_subset_size
            for b, sample_subset_size in zip(b_list, sample_subset_sizes, strict=False)
        ]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(
                    x_list,
                    b_list,
                    sample_subset_sizes,
                    strict=False,
                )
            ]
            x_subset_1_list = [
                x[indices_1]
                for x, indices_1 in zip(x_list, indices_1_list, strict=False)
            ]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1)
                    for rope, indices_1 in zip(rope_list, indices_1_list, strict=False)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list,
                    residual_1_list,
                    indices_1_list,
                    residual_scale_factors,
                    strict=False,
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(
                    x_list,
                    b_list,
                    sample_subset_sizes,
                    strict=False,
                )
            ]
            x_subset_2_list = [
                x[indices_2]
                for x, indices_2 in zip(x_attn_list, indices_2_list, strict=False)
            ]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list,
                    residual_2_list,
                    indices_2_list,
                    residual_scale_factors,
                    strict=False,
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list, strict=False):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(
        self,
        x_or_x_list: Tensor | list[Tensor],
        rope_or_rope_list: Tensor | list[Tensor] | None = None,
    ) -> Tensor | list[Tensor]:
        """Forward pass."""
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        if isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        raise AssertionError
