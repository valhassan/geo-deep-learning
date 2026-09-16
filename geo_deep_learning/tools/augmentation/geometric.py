"""Pixel-exact D4 geometric transforms (no interpolation)."""

import torch
from torch import Tensor, nn


def apply_d4(x: Tensor, k: Tensor, hflip: Tensor) -> Tensor:
    """Per-sample rot90(k) then optional hflip. k in {0,1,2,3}."""
    v = (-1,) + (1,) * (x.ndim - 1)
    k, hflip = k.view(v), hflip.view(v)
    out = x
    for kk in (1, 2, 3):
        out = torch.where(k == kk, x.rot90(kk, (-2, -1)), out)
    return torch.where(hflip, out.flip(-1), out)


class RandomD4(nn.Module):
    """
    Uniform D4: k ~ U{0,1,2,3} and Bernoulli hflip, shared across maps.

    Identity is 1/8. Inputs are (B, C, H, W). One tensor in → tensor out;
    several → tuple, same (k, hflip) on each.
    """

    def sample(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Draw per-image k and hflip."""
        k = torch.randint(0, 4, (batch,), device=device)
        hflip = torch.randint(0, 2, (batch,), device=device, dtype=torch.bool)
        return k, hflip

    @torch.no_grad()
    def forward(self, *tensors: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Apply one sampled D4 element to every spatial map."""
        ref = tensors[0]
        k, hflip = self.sample(ref.shape[0], ref.device)
        out = tuple(apply_d4(t, k, hflip) for t in tensors)
        return out[0] if len(out) == 1 else out
