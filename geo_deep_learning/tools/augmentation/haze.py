"""Physically-grounded atmospheric haze via the Koschmieder model."""

from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D


class RandomKoschmiederHaze(IntensityAugmentationBase2D):
    """
    Simulates atmospheric haze using the wavelength-dependent Koschmieder model.

    Args:
        beta_range: (min, max) range for the base scattering coefficient.
                    Higher beta means thicker haze.
        airlight_range: (min, max) range for the atmospheric light intensity.
        p: probability of applying the augmentation per image.

    """

    def __init__(
        self,
        beta_range: tuple[float, float] = (0.8, 2.5),
        airlight_range: tuple[float, float] = (0.55, 0.95),
        p: float = 1.0,
    ) -> None:
        """Initialize RandomKoschmiederHaze."""
        super().__init__(p=p, same_on_batch=False, p_batch=1.0)
        self.beta_range = beta_range
        self.airlight_range = airlight_range

    def generate_physics_parameters(
        self,
        shape: torch.Size,
        wavelengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate physics parameters."""
        b = shape[0]
        device = wavelengths.device
        dtype = wavelengths.dtype

        # 1. Sample base scattering coefficient (beta) per image
        lo_b, hi_b = self.beta_range
        beta_base = torch.rand(b, 1, device=device, dtype=dtype) * (hi_b - lo_b) + lo_b

        # 2. Sample airlight (A) per image
        lo_a, hi_a = self.airlight_range
        airlight = (
            torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * (hi_a - lo_a) + lo_a
        )

        # 3. Apply Rayleigh scattering physics (beta varies by lambda^-4)
        # Using 0.55 µm (Green) as the anchor/reference wavelength
        # Shape of beta_lambda becomes (B, C)
        beta_lambda = beta_base * (0.55 / wavelengths) ** 4

        # 4. Calculate Transmission (t) = e^(-beta)
        # Reshape to (B, C, 1, 1) to easily broadcast over spatial dims (H, W)
        t = torch.exp(-beta_lambda).view(b, -1, 1, 1)

        return {"t": t, "A": airlight}

    def apply_transform(
        self,
        input: torch.Tensor,  # noqa: A002
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],  # noqa: ARG002
        transform: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Apply Koschmieder haze; preserve all-zero (void) pixels."""
        t = params["t"]
        a = params["A"]

        # Koschmieder equation: I_hazy = I_clear * t + A * (1 - t)
        hazy = torch.clamp(input * t + a * (1.0 - t), 0.0, 1.0)

        # Void / nodata fill is all-band 0; do not replace with airlight.
        valid = input.gt(0).any(dim=1, keepdim=True)
        return torch.where(valid, hazy, input)
