"""Utility functions for models."""

import warnings  # noqa: I001
import math
import torch

# import torch before MultiScaleDeformableAttention
import MultiScaleDeformableAttention as msda  # noqa: N813
import torch.nn.functional as fn
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_


class ConvModule(nn.Module):
    """Convolution module."""

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        dilation: int = 1,
        stride: int = 1,
        *,
        inplace: bool = False,
        transpose: bool = False,
        scale_factor: int | None = None,
    ) -> None:
        """Initialize the convolution module."""
        super().__init__()

        kind = "Transpose" if transpose else ""

        conv_name = f"Conv{kind}2d"

        if transpose:
            stride = scale_factor
            padding = (kernel_size - scale_factor) // 2

        conv_template = getattr(nn, conv_name)
        self.conv = conv_template(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.act(self.norm(self.conv(x)))


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(
        self,
        pool_scales: tuple[int, ...],
        in_channels: int,
        channels: int,
        *,
        align_corners: bool,
    ) -> None:
        """Initialize the Pooling Pyramid Module."""
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, inplace=True),
                ),
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


def resize(  # noqa: PLR0913
    input_: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "nearest",
    *,
    align_corners: bool | None = None,
    warning: bool = True,
) -> torch.Tensor:
    """Resize a tensor."""
    if scale_factor is not None:
        h, w = input_.shape[2:]
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        size = (new_h, new_w)
        scale_factor = None
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input_.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if output_h > input_h or (
            output_w > input_w
            and output_h > 1
            and output_w > 1
            and input_h > 1
            and input_w > 1
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            warnings.warn(
                f"When align_corners={align_corners}, "
                "the output would more aligned if "
                f"input size {(input_h, input_w)} is `x+1` and "
                f"out size {(output_h, output_w)} is `nx+1`",
                stacklevel=2,
            )
    return fn.interpolate(
        input_,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def patch_first_conv(
    model: nn.Module,
    new_in_channels: int,
    default_in_channels: int = 3,
    *,
    pretrained: bool = True,
) -> None:
    """Change first convolution layer input channels."""
    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break

    weight = module.weight.detach()
    module.in_channels = new_in_channels

    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(
                module.out_channels,
                new_in_channels // module.groups,
                *module.kernel_size,
            ),
        )
        module.reset_parameters()

    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)

    else:
        new_weight = torch.Tensor(
            module.out_channels,
            new_in_channels // module.groups,
            *module.kernel_size,
        )

        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]

        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


def is_power_of_2(n: int) -> bool:
    """Check if a number is a power of 2."""
    if (not isinstance(n, int)) or (n < 0):
        err_msg = f"invalid input for _is_power_of_2: {n} (type: {type(n)})"
        raise ValueError(err_msg)
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttnFunction(Function):
    """Multi-Scale Deformable Attention Function."""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(  # noqa: PLR0913
        ctx: object,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: int,
    ) -> torch.Tensor:
        """Forward function."""
        ctx.im2col_step = im2col_step

        # Use fast CUDA kernel (memory-efficient, processes in chunks)
        output = msda.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        # Fallback to PyTorch implementation (slower, more memory)
        # output = ms_deform_attn_core_pytorch(
        #     value,
        #     value_spatial_shapes,
        #     sampling_locations,
        #     attention_weights,
        # )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @custom_bwd
    @once_differentiable
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Backward function."""
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = msda.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Multi-Scale Deformable Attention Core PyTorch."""
    # for debug and test only,
    # need to use cuda version instead
    n_, s_, m_, d_ = value.shape
    _, lq_, m_, l_, p_, _ = sampling_locations.shape
    value_list = value.split([h_ * w_ for h_, w_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (h_, w_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(n_ * m_, d_, h_, w_)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = fn.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        n_ * m_,
        1,
        lq_,
        l_ * p_,
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(n_, m_ * d_, lq_)
    )
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module."""

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        ratio: float = 1.0,
    ) -> None:
        """
        Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     num of feature levels
        :param n_heads      num of attention heads
        :param n_points     num of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            err_msg = (
                f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}"
            )
            raise ValueError(err_msg)
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.",
                stacklevel=2,
            )
        self.im2col_step = 128
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Reset parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(  # noqa: PLR0913
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward function."""
        n, len_q, _ = query.shape
        n, len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == len_in  # noqa: S101

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(
            n,
            len_in,
            self.n_heads,
            int(self.ratio * self.d_model) // self.n_heads,
        )
        sampling_offsets = self.sampling_offsets(query).view(
            n,
            len_q,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            n,
            len_q,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = fn.softmax(attention_weights, -1).view(
            n,
            len_q,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )

        if reference_points.shape[-1] == 2:  # noqa: PLR2004
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:  # noqa: PLR2004
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            err_msg = (
                f"Last dim of reference_points must be 2 or 4, "
                f"but get {reference_points.shape[-1]} instead."
            )
            raise ValueError(err_msg)
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        return self.output_proj(output)
