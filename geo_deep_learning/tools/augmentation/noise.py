"""Physically-grounded sensor noise simulation via Poisson photon counting."""

from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D


class RandomPoissonNoise(IntensityAugmentationBase2D):
    r"""
    Simulates sensor signal-to-noise ratio (SNR) using shot noise (Poisson,
    signal-dependent) plus a fixed read-noise floor (Gaussian, roughly
    signal-independent) from the sensor's readout electronics.

    Lower photon_scale simulates a smaller aperture or lower light conditions.

    Args:
        photon_range: (min, max) range for the photon scaling factor.
                      e.g., 100.0 is very noisy, 10000.0 is very clean.
        read_noise_range: (min, max) range for the fixed read-noise sigma,
                           in the same [0, 1] reflectance units as the input.
        p: probability of applying the augmentation per image.
    """

    def __init__(
        self,
        photon_range: tuple[float, float] = (30.0, 5000.0),
        read_noise_range: tuple[float, float] = (0.0, 0.04),
        p: float = 1.0,
    ) -> None:
        """Initialize RandomPoissonNoise."""
        super().__init__(p=p, same_on_batch=False, p_batch=1.0)
        self.photon_range = photon_range
        self.read_noise_range = read_noise_range

    def generate_physics_parameters(
        self,
        shape: torch.Size,
        input: torch.Tensor,  # noqa: A002
    ) -> dict[str, torch.Tensor]:
        """Generate physics parameters (photon scale + read noise, per image)."""
        b = shape[0]
        device = input.device
        dtype = input.dtype

        # 1. Sample a photon scale per image (governs shot-noise magnitude)
        lo_p, hi_p = self.photon_range
        scale = torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * (hi_p - lo_p) + lo_p

        # 2. Sample a read-noise sigma per image (fixed noise floor, independent
        # of signal level, from the sensor's readout electronics)
        lo_r, hi_r = self.read_noise_range
        read_sigma = (
            torch.rand(b, 1, 1, 1, device=device, dtype=dtype) * (hi_r - lo_r) + lo_r
        )

        return {"photon_scale": scale, "read_sigma": read_sigma}

    def apply_transform(
        self,
        input: torch.Tensor,  # noqa: A002
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],  # noqa: ARG002
        transform: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Apply the Poisson + read-noise transform."""
        scale = params["photon_scale"]
        read_sigma = params["read_sigma"]

        # 1. Convert normalized [0, 1] reflectance to an artificial photon count
        photons = input * scale

        # 2. Apply quantum Poisson (shot) noise. Note: torch.poisson isn't
        # perfectly differentiable in the backward pass, but for data
        # augmentation in the forward pass, it is perfectly fine.
        noisy_photons = torch.poisson(torch.clamp(photons, min=0.0))

        # 3. Scale back to reflectance [0, 1]
        noisy_input = noisy_photons / scale

        # 4. Add the fixed read-noise floor (signal-independent Gaussian)
        noisy_input = noisy_input + torch.randn_like(input) * read_sigma

        return torch.clamp(noisy_input, 0.0, 1.0)
