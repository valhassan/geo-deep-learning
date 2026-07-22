"""Physically-grounded directional illumination via sun elevation/azimuth shading."""

from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D


class RandomDirectionalIllumination(IntensityAugmentationBase2D):
    """
    Simulates directional sunlight shading via a smooth per-image gradient field.

    Physics: low sun elevation produces long, low-angle light and a strong
    directional shading gradient across the scene; high sun elevation
    produces flat, near-uniform lighting (weak gradient). Azimuth sets the
    gradient direction. Purely spatial/geometric — illuminant chromaticity
    (color temperature) is intentionally NOT modeled here; that's owned
    exclusively by RandomPlanckianIllumination to avoid two augmentations
    independently perturbing the same channel-gain degree of freedom. No
    polygon/label dependency — safe for unlabeled pretraining imagery.

    Args:
        elevation_range: (min, max) sun elevation in degrees. Lower values
                          produce stronger directional shading.
        p: probability of applying the augmentation per image.

    """

    def __init__(
        self,
        elevation_range: tuple[float, float] = (5.0, 40.0),
        p: float = 1.0,
    ) -> None:
        """Initialize RandomDirectionalIllumination."""
        super().__init__(p=p, same_on_batch=False, p_batch=1.0)
        self.elevation_range = elevation_range

    def generate_physics_parameters(
        self,
        shape: torch.Size,
        input: torch.Tensor,  # noqa: A002
    ) -> dict[str, torch.Tensor]:
        """Generate physics parameters."""
        b = shape[0]
        device = input.device
        dtype = input.dtype

        # 1. Sample sun azimuth per image (direction of the shading gradient)
        azimuth = torch.rand(b, device=device, dtype=dtype) * 2.0 * torch.pi

        # 2. Sample sun elevation per image (degrees -> radians)
        lo_e, hi_e = self.elevation_range
        elevation_deg = torch.rand(b, device=device, dtype=dtype) * (hi_e - lo_e) + lo_e
        elevation = elevation_deg * (torch.pi / 180.0)

        # 3. Gradient amplitude scales inversely with elevation.
        # At elevation -> 90 deg (overhead sun), amplitude -> 0 (flat lighting).
        # At elevation -> 0 deg (grazing sun), amplitude -> its max (strong shading).
        amplitude = 1.0 - torch.sin(elevation)

        return {"azimuth": azimuth, "amplitude": amplitude}

    def apply_transform(
        self,
        input: torch.Tensor,  # noqa: A002
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],  # noqa: ARG002
        transform: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Apply transform."""
        b, _, h, w = input.shape
        azimuth = params["azimuth"]
        amplitude = params["amplitude"]

        # Normalized pixel coordinate grid in [-1, 1]
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=input.device, dtype=input.dtype),
            torch.linspace(-1.0, 1.0, w, device=input.device, dtype=input.dtype),
            indexing="ij",
        )
        xs = xs.expand(b, -1, -1)  # (B, H, W)
        ys = ys.expand(b, -1, -1)  # (B, H, W)

        # Project coordinates onto the sun direction vector -> linear ramp in [-1, 1]
        dir_x = torch.cos(azimuth).view(b, 1, 1)
        dir_y = torch.sin(azimuth).view(b, 1, 1)
        ramp = xs * dir_x + ys * dir_y
        ramp = ramp / ramp.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        # Shading field: 1.0 at ramp midpoint, +/- amplitude at the extremes
        shading = 1.0 + amplitude.view(b, 1, 1) * ramp
        shading = shading.unsqueeze(1)  # (B, 1, H, W), broadcasts over channels

        shaded = input * shading

        return torch.clamp(shaded, 0.0, 1.0)
