"""Physically-grounded directional illumination via sun elevation/azimuth shading."""

from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D


class RandomDirectionalIllumination(IntensityAugmentationBase2D):
    """
    Simulates directional sunlight shading via a smooth per-image gradient field.

    Physics: low sun elevation produces long, high-contrast directional shading
    (strong gradient across the scene) plus a warm color shift, since light
    travels through more atmosphere and Rayleigh scattering removes more of
    the shorter (blue) wavelengths than the longer (red) ones — the same
    mechanism as sunset reddening. High sun elevation produces flat, near-
    neutral lighting. Azimuth sets the gradient direction. No polygon/label
    dependency — safe for unlabeled pretraining imagery.

    Args:
        elevation_range: (min, max) sun elevation in degrees. Lower values
                          produce stronger directional shading and warmer color.
        warmth_strength: scales how strongly low elevation warms the color
                          temperature via wavelength-dependent scattering.
        p: probability of applying the augmentation per image.

    """

    def __init__(
        self,
        elevation_range: tuple[float, float] = (15.0, 75.0),
        warmth_strength: float = 0.4,
        p: float = 1.0,
    ) -> None:
        """Initialize RandomDirectionalIllumination."""
        super().__init__(p=p, same_on_batch=False, p_batch=1.0)
        self.elevation_range = elevation_range
        self.warmth_strength = warmth_strength

    def generate_physics_parameters(
        self,
        shape: torch.Size,
        wavelengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate physics parameters."""
        b = shape[0]
        device = wavelengths.device
        dtype = wavelengths.dtype

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

        # 4. Wavelength-coupled warm shift (Rayleigh scattering, same law as
        # haze's beta_lambda): shorter wavelengths attenuate more as the light
        # path through the atmosphere lengthens (i.e. as amplitude grows).
        # Using 0.55 um (Green) as the same anchor wavelength as haze.py.
        color_t = torch.exp(
            -self.warmth_strength * amplitude.view(b, 1) * (0.55 / wavelengths) ** 4,
        )

        return {"azimuth": azimuth, "amplitude": amplitude, "color_t": color_t}

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
        color_t = params["color_t"]

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
        color = color_t.view(b, -1, 1, 1)  # (B, C, 1, 1), broadcasts over H, W

        shaded = input * shading * color

        return torch.clamp(shaded, 0.0, 1.0)
