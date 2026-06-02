"""Physically-grounded GSD simulation via MTF blur + spatial downsampling."""

import math
from typing import Any

import kornia
import torch

# import torch.nn.functional as F  # only needed for the sequential fallback below
from kornia.augmentation import IntensityAugmentationBase2D


class RandomGSDSimulation(IntensityAugmentationBase2D):
    r"""
    Simulates lower-resolution sensors via MTF blur and spatial downsampling.

    Args:
        target_gsd: (min, max) range in metres to sample simulated GSD from.
        p: probability of applying the augmentation per image.

    """

    def __init__(
        self, target_gsd: tuple[float, float] = (1.5, 3.0), p: float = 1.0,
    ) -> None:
        """Initialize RandomGSDSimulation."""
        super().__init__(p=p, same_on_batch=False, p_batch=1.0)
        self.target_gsd = target_gsd

    def generate_physics_parameters(
        self,
        shape: torch.Size,
        src_gsd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate physics parameters."""
        b = shape[0]
        lo, hi = self.target_gsd
        simulated_gsd = (
            torch.rand(b, device=src_gsd.device, dtype=src_gsd.dtype) * (hi - lo) + lo
        )
        ratio = simulated_gsd / src_gsd
        sigma = torch.where(ratio > 1.0, (ratio - 1.0) / 2.355, torch.zeros_like(ratio))
        return {"sigma": sigma, "scale_factor": 1.0 / ratio}

    def apply_transform(
        self,
        input: torch.Tensor,  # noqa: A002
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],  # noqa: ARG002
        transform: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Apply transform."""
        b, _, h, w = input.shape
        sigmas = params["sigma"]
        scale_factors = params["scale_factor"]

        max_sigma = torch.max(sigmas).item()
        if max_sigma > 0.0:
            kernel_size = max(3, int(2 * math.ceil(3.0 * max_sigma) + 1))
            blurred = kornia.filters.gaussian_blur2d(
                input,
                (kernel_size, kernel_size),
                sigmas.unsqueeze(1).repeat(1, 2),
            )
        else:
            blurred = input

        # --- Vectorized resampling via kornia.geometry.remap ---
        # Builds a (B, H, W) coordinate map per image, quantized to each image's
        # lower-res grid, then samples in a single batched GPU call.
        # Images with sigma=0 (scale_factor >= 1) get a near-identity map.
        grid = kornia.geometry.grid.create_meshgrid(
            h,
            w,
            normalized_coordinates=False,
            device=input.device,
        )
        xs = grid[..., 0].expand(b, -1, -1)  # (B, H, W) pixel x-coords
        ys = grid[..., 1].expand(b, -1, -1)  # (B, H, W) pixel y-coords
        s = scale_factors.view(b, 1, 1)
        map_x = torch.round(xs * s) / s
        map_y = torch.round(ys * s) / s
        resampled = kornia.geometry.transform.remap(
            blurred,
            map_x,
            map_y,
            normalized_coordinates=False,
        )
        return torch.clamp(resampled, 0.0, 1.0)

        # --- Sequential fallback (sequential per-image F.interpolate) ---
        # output = []
        # for i in range(b):
        #     if sigmas[i] <= 0.0:
        #         output.append(input[i])
        #         continue
        #     scale = scale_factors[i].item()
        #     down = F.interpolate(
        #         blurred[i : i + 1],
        #         scale_factor=scale,
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        #     up = F.interpolate(down, size=(h, w),
        #                        mode="bilinear",
        #                        align_corners=False)
        #     output.append(up.squeeze(0))
        # return torch.clamp(torch.stack(output, dim=0), 0.0, 1.0)
