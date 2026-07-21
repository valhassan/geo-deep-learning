"""Physically-grounded multi-band illuminant jitter via Planck's law."""

from typing import Any

import torch
from kornia.augmentation import IntensityAugmentationBase2D

# Second radiation constant in µm·K (wavelengths are always µm).
_C2_UM_K = 14388.0


class RandomPlanckianIllumination(IntensityAugmentationBase2D):
    """
    Simulates illuminant chromaticity changes via relative blackbody spectra.

    Samples a correlated color temperature T per image and multiplies each
    band by B(λ, T) / B(λ, T_ref). Gains are L1-normalized to mean 1 so
    average intensity is preserved across arbitrary band counts (VIS+NIR).

    λ^-5 cancels in the same-λ ratio, leaving:
        g(λ, T) = (exp(c2/(λ T_ref)) - 1) / (exp(c2/(λ T)) - 1)

    Args:
        cct_range: (min, max) CCT in Kelvin. Lower is warmer (redder).
        t_ref: reference CCT assumed for the input (≈ D65 daylight).
        p: probability of applying the augmentation per image.

    """

    def __init__(
        self,
        cct_range: tuple[float, float] = (3200.0, 10000.0),
        t_ref: float = 6500.0,
        p: float = 1.0,
    ) -> None:
        """Initialize RandomPlanckianIllumination."""
        super().__init__(p=p, same_on_batch=False, p_batch=1.0)
        self.cct_range = cct_range
        self.t_ref = t_ref

    def generate_physics_parameters(
        self,
        shape: torch.Size,
        wavelengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Generate per-band illuminant gains from sampled CCT."""
        b, c = shape[0], shape[1]
        device = wavelengths.device
        dtype = wavelengths.dtype

        if wavelengths.shape[-1] != c:
            msg = (
                f"wavelengths last dim {wavelengths.shape[-1]} != channels {c}"
            )
            raise ValueError(msg)

        # (B, C) — accepts (C,) or (B, C)
        lam = wavelengths.expand(b, c) if wavelengths.ndim == 1 else wavelengths

        lo, hi = self.cct_range
        temp = torch.rand(b, 1, device=device, dtype=dtype) * (hi - lo) + lo

        # g = (e^{c2/(λ T_ref)} - 1) / (e^{c2/(λ T)} - 1)
        gains = torch.expm1(_C2_UM_K / (lam * self.t_ref)) / torch.expm1(
            _C2_UM_K / (lam * temp),
        )

        # L1 brightness lock: mean channel gain == 1
        gains = gains / gains.mean(dim=-1, keepdim=True).clamp_min(1e-12)

        return {"gains": gains.view(b, c, 1, 1)}

    def apply_transform(
        self,
        input: torch.Tensor,  # noqa: A002
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],  # noqa: ARG002
        transform: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Apply relative blackbody illuminant gains."""
        return torch.clamp(input * params["gains"], 0.0, 1.0)
