"""LeJEPA-SIGReg loss."""

# Adapted from https://github.com/galilai-group/lejepa/tree/main

from __future__ import annotations

import torch
from torch import distributed as dist
from torch import nn
from torch.distributed._functional_collectives import (
    all_reduce as functional_all_reduce,
)


def ddp_all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
    """Average across ranks (no-op if not in DDP)."""
    if dist.is_available() and dist.is_initialized():
        return functional_all_reduce(x, "avg", dist.group.WORLD)
    return x


def ddp_all_reduce_max(x: torch.Tensor) -> torch.Tensor:
    """Max across ranks (no-op if not in DDP)."""
    if dist.is_available() and dist.is_initialized():
        return functional_all_reduce(x, "max", dist.group.WORLD)
    return x


class EppsPulley(nn.Module):
    """Univariate Epps-Pulley statistic (fast CF-based normality test)."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        """Initialize the EppsPulley statistic."""
        super().__init__()
        if n_points % 2 != 1:
            msg = "n_points must be odd (per official implementation)."
            raise ValueError(msg)

        # Linearly spaced positive points (including 0) over [0, t_max]
        t = torch.linspace(0.0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)

        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # half-weight at endpoints

        # phi(t) = exp(-t^2/2), and bake it into weights
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the EppsPulley statistic."""
        expected_shape = 2
        if x.ndim < expected_shape:
            msg = f"EppsPulley expects at least 2 dims (*, N, K). Got {x.shape}"
            raise ValueError(msg)

        num_samples = x.size(-2)
        world_size = (
            dist.get_world_size()
            if (dist.is_available() and dist.is_initialized())
            else 1
        )

        # x_t: (*, N, K, T) where T=n_points
        x_t = x.unsqueeze(-1) * self.t.to(dtype=x.dtype, device=x.device)

        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # mean over samples dimension (-3) => (*, K, T)
        cos_mean = cos_vals.mean(dim=-3)
        sin_mean = sin_vals.mean(dim=-3)

        # DDP average across ranks
        cos_mean = ddp_all_reduce_avg(cos_mean)
        sin_mean = ddp_all_reduce_avg(sin_mean)

        # err: (*, K, T)
        phi = self.phi.to(dtype=x.dtype, device=x.device)
        err = (cos_mean - phi).square() + sin_mean.square()

        # integrate over T => (*, K)
        weights = self.weights.to(
            dtype=x.dtype,
            device=x.device,
        )  # already includes symmetry + phi
        return (err @ weights) * (num_samples * world_size)  # scale by total samples


class SlicingUnivariateTest(nn.Module):
    """Multivariate extension via random slicing."""

    def __init__(
        self,
        univariate_test: nn.Module,
        num_slices: int = 256,
        reduction: str = "mean",
        clip_value: float | None = None,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the SlicingUnivariateTest."""
        super().__init__()
        if reduction not in ("mean", "sum", None):
            msg = "reduction must be 'mean', 'sum', or None"
            raise ValueError(msg)

        self.univariate_test = univariate_test
        self.num_slices = int(num_slices)
        self.reduction = reduction
        self.clip_value = clip_value
        self.eps = float(eps)

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(int(seed))
        return self._generator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SlicingUnivariateTest."""
        expected_shape = 2
        if x.ndim < expected_shape:
            msg = f"SlicingUnivariateTest expects (*, N, D). Got {x.shape}"
            raise ValueError(msg)

        with torch.no_grad():
            # sync step across ranks via MAX (exact intent of slicing.py)
            step = ddp_all_reduce_max(self.global_step.clone()).item()
            g = self._get_generator(x.device, seed=step)

            num_features = x.size(-1)
            random_matrix = torch.randn(
                (num_features, self.num_slices),
                device=x.device,
                dtype=x.dtype,
                generator=g,
            )
            random_matrix = random_matrix / (
                random_matrix.norm(p=2, dim=0, keepdim=True) + self.eps
            )  # normalize like slicing.py

            self.global_step.add_(1)

        # project: (*, N, D) @ (D, K) -> (*, N, K)  (matches docstring)
        stats = self.univariate_test(x @ random_matrix)  # expects (*, N, K) -> (*, K)

        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, torch.zeros_like(stats), stats)

        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization."""

    def __init__(
        self,
        num_slices: int = 1024,
        t_max: float = 5.0,
        n_points: int = 17,
        reduction: str = "mean",
        clip_value: float | None = None,
    ) -> None:
        """Initialize the SIGReg."""
        super().__init__()
        univariate = EppsPulley(t_max=t_max, n_points=n_points)
        self.test = SlicingUnivariateTest(
            univariate_test=univariate,
            num_slices=num_slices,
            reduction=reduction,
            clip_value=clip_value,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Run the SIGReg."""
        return self.test(embeddings)


class GeoJEPALoss(nn.Module):
    """GeoJEPA SSL loss."""

    def __init__(
        self,
        lambda_sig: float = 0.05,
        n_samples: int = 512,
        num_slices: int = 256,
        **kwargs: object,
    ) -> None:
        """Initialize the GeoJEPALoss."""
        super().__init__()
        self.lambda_sig = lambda_sig
        self.n_samples = n_samples
        self.sigreg = SIGReg(num_slices=num_slices, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the GeoJEPALoss."""
        inv_loss = x.var(dim=0).mean()
        v, b, n, d = x.shape
        tokens = x.reshape(v * b, n, d)
        if self.n_samples is not None and self.n_samples < n:
            idx = torch.randperm(n, device=x.device)[: self.n_samples]
            tokens = tokens[:, idx, :]  # [V*B, n_samples, D]
        sig_loss = self.sigreg(tokens)
        return (1.0 - self.lambda_sig) * inv_loss + self.lambda_sig * sig_loss
