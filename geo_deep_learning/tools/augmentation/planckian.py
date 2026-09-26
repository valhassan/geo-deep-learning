"""Planckian illuminant jitter (wavelength-aware, no interpolation)."""

import torch
from torch import Tensor, nn

# Second radiation constant in µm·K (wavelengths are µm).
_C2_UM_K = 14388.0
_MIRED_K = 1e6


class RandomPlanckian(nn.Module):
    """
    Relative blackbody gains B(λ, T) / B(λ, T_ref), mean-locked to 1.

    Image-only. λ^-5 cancels in the ratio. Gains computed in fp32.
    CCT sampled uniform in mireds (1e6/T) over cct_range, then converted
    to Kelvin.
    """

    def __init__(
        self,
        cct_range: tuple[float, float] = (3200.0, 10000.0),
        t_ref: float = 6500.0,
        p: float = 0.1,
    ) -> None:
        """Initialize RandomPlanckian."""
        super().__init__()
        self.cct_range = cct_range
        self.t_ref = t_ref
        self.p = p

    def sample(self, x: Tensor, wavelengths: Tensor) -> Tensor:
        """Per-image gains of shape (B, C, 1, 1). wavelengths is (C,) or (B, C)."""
        b, c = x.shape[:2]
        lam = wavelengths.to(device=x.device, dtype=torch.float32)
        if lam.shape[-1] != c:
            msg = f"wavelengths last dim {lam.shape[-1]} != channels {c}"
            raise ValueError(msg)
        if lam.ndim == 1:
            lam = lam.expand(b, c)
        lo, hi = self.cct_range
        m_lo, m_hi = _MIRED_K / hi, _MIRED_K / lo
        mired = torch.rand(b, 1, device=lam.device, dtype=torch.float32)
        mired = mired * (m_hi - m_lo) + m_lo
        temp = _MIRED_K / mired
        gains = torch.expm1(_C2_UM_K / (lam * self.t_ref)) / torch.expm1(
            _C2_UM_K / (lam * temp),
        )
        gains = gains / gains.mean(dim=-1, keepdim=True).clamp_min(1e-12)
        return gains.view(b, c, 1, 1)

    @torch.no_grad()
    def forward(self, x: Tensor, wavelengths: Tensor) -> Tensor:
        """Apply illuminant gains to x in [0, 1]."""
        gains = self.sample(x, wavelengths)
        if self.p < 1.0:
            use = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
            gains = torch.where(use, gains, 1.0)
        return (x * gains.to(dtype=x.dtype)).clamp(0.0, 1.0)
