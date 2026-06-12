"""clDice Loss for roads — topology-preserving via soft skeleton."""

import torch
import torch.nn.functional as fn
from torch import nn


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """
    One step of soft morphological erosion via min-pooling.

    min_pool2d is not native in PyTorch; implemented as -max_pool2d(-x).
    Operates on (N, 1, H, W) float tensor.
    """
    return -fn.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """One step of soft morphological dilation via max-pooling."""
    return fn.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_skeleton(x: torch.Tensor, num_iter: int) -> torch.Tensor:
    """
    Differentiable soft skeleton via iterative morphological thinning.

    Each iteration erodes x once, then computes the residual between
    the eroded mask and its dilation (= opening of the already-eroded x).
    The eroded result is reused directly — no redundant second erosion.

    Args:
        x:         (N, 1, H, W) float tensor in [0, 1].
        num_iter:  Number of erosion iterations. Should be >= half the
                   maximum object width in pixels. 10 to 15 is sufficient
                   for road widths up to ~30px.

    Returns:
        (N, 1, H, W) soft skeleton in [0, 1].

    """
    # Iteration 0: residual before any erosion
    x_e = _soft_erode(x)
    skel = fn.relu(x - _soft_dilate(x_e))
    x = x_e

    for _ in range(num_iter - 1):
        x_e = _soft_erode(x)
        skel = skel + fn.relu(x - _soft_dilate(x_e))
        x = x_e

    return skel


class CLDiceLoss(nn.Module):
    """
    clDice Loss for tubular / curvilinear structures (roads).

    Computes topology-aware loss via soft skeletons of both prediction
    and ground truth, enforcing connectivity preservation along the
    road centerline.

    Reference:
        Shit et al., "clDice — a Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation", CVPR 2021.
    """

    def __init__(
        self,
        road_class: int = 3,
        num_iter: int = 10,
        smooth: float = 1e-5,
    ) -> None:
        """
        Initialise CLDiceLoss.

        Args:
            road_class:  Class index for roads in the multiclass output.
            num_iter:    Soft skeleton erosion iterations.
                         10 covers road widths up to ~30px (15cm to 1m GSD).
            smooth:      Numerical stability epsilon.

        """
        super().__init__()
        self.road_class = road_class
        self.num_iter = num_iter
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pred:         (N, C, H, W) raw logits.
            gt:           (N, H, W) integer class indices (ignore pixels
                          already zeroed out by the caller).
            ignore_mask:  (N, H, W) float, 1 = valid, 0 = ignored.

        Returns:
            Scalar clDice loss in [0, 1].

        """
        # Extract roads channel probability without full softmax over C.
        # log_softmax then exp on one slice is equivalent but we only need
        # the probability for one class, so we compute it directly:
        #   p_k = exp(l_k) / sum_j(exp(l_j))
        # Using logsumexp for numerical stability.
        road_logit = pred[:, self.road_class : self.road_class + 1]  # (N,1,H,W)
        v_p = torch.exp(
            road_logit - torch.logsumexp(pred, dim=1, keepdim=True),
        )  # (N, 1, H, W)

        v_l = (gt == self.road_class).float().unsqueeze(1)  # (N, 1, H, W)

        if ignore_mask is not None:
            mask = ignore_mask.unsqueeze(1)
            v_p = v_p * mask
            v_l = v_l * mask

        # GT skeleton carries no gradient — skip autograd graph construction.
        with torch.no_grad():
            s_l = soft_skeleton(v_l, self.num_iter)

        s_p = soft_skeleton(v_p, self.num_iter)

        # clDice: harmonic mean of skeleton precision and sensitivity.
        # precision: predicted skeleton covered by GT mask
        # recall:    GT skeleton covered by predicted mask
        tprec = (torch.sum(s_p * v_l) + self.smooth) / (
            torch.sum(s_p) + self.smooth
        )
        tsens = (torch.sum(s_l * v_p) + self.smooth) / (
            torch.sum(s_l) + self.smooth
        )

        return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
