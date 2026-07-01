"""Image-quality metrics (PSNR / SSIM) used by the validation step.

Primary implementation uses :mod:`torchmetrics`, which is the canonical
PyTorch-native metric library.  If torchmetrics is not installed we fall
back to a pure-torch implementation (slower, but no extra dependency).

All metrics expect images in ``[0, 1]`` with shape ``(B, 3, H, W)`` and
return higher-is-better values (PSNR in dB, SSIM in ``[-1, 1]``).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primary backend: torchmetrics
# ---------------------------------------------------------------------------
try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    _HAS_TORCHMETRICS = True
except ImportError:                       # pragma: no cover - exercised manually
    _HAS_TORCHMETRICS = False


# ---------------------------------------------------------------------------
# Pure-torch fallback implementations
# ---------------------------------------------------------------------------
def _psnr_torch(pred: torch.Tensor, target: torch.Tensor,
                data_range: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Element-wise PSNR in dB.  pred / target in ``[0, data_range]``."""
    mse = F.mse_loss(pred, target, reduction="mean")
    psnr = 20.0 * torch.log10(torch.tensor(data_range, device=pred.device)) \
        - 10.0 * torch.log10(mse + eps)
    return psnr


def _gaussian_kernel(window_size: int, sigma: float,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1-D Gaussian kernel, returned as ``(1, 1, W)`` for ``F.conv1d`` use."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.view(1, 1, -1)


def _ssim_torch(pred: torch.Tensor, target: torch.Tensor,
                data_range: float = 1.0,
                window_size: int = 11,
                sigma: float = 1.5) -> torch.Tensor:
    """SSIM (mean over the batch).  pred / target in ``[0, data_range]``."""
    K1 = 0.01 ** 2
    K2 = 0.03 ** 2
    c1 = (K1 * data_range) ** 2
    c2 = (K2 * data_range) ** 2

    device, dtype = pred.device, pred.dtype
    g1d = _gaussian_kernel(window_size, sigma, device, dtype).squeeze(0).squeeze(0)
    # Outer product gives a 2-D Gaussian of shape (W, W).
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)            # (W, 1) @ (1, W) -> (W, W)
    kernel = g2d.unsqueeze(0).unsqueeze(0)               # (1, 1, W, W)
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_p = F.conv2d(pred,   kernel, padding=pad, groups=pred.shape[1])
    mu_t = F.conv2d(target, kernel, padding=pad, groups=target.shape[1])
    mu_p_sq, mu_t_sq, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sig_p_sq = F.conv2d(pred ** 2,   kernel, padding=pad, groups=pred.shape[1]) - mu_p_sq
    sig_t_sq = F.conv2d(target ** 2, kernel, padding=pad, groups=target.shape[1]) - mu_t_sq
    sig_pt   = F.conv2d(pred * target, kernel, padding=pad, groups=target.shape[1]) - mu_pt

    num = (2 * mu_pt + c1) * (2 * sig_pt + c2)
    den = (mu_p_sq + mu_t_sq + c1) * (sig_p_sq + sig_t_sq + c2)
    ssim_map = num / den
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Unified meter
# ---------------------------------------------------------------------------
class ImageQualityMeter:
    """Running PSNR + SSIM meter.

    Designed to be reused across the whole validation epoch: call
    :meth:`update` on each (predicted, target) batch and :meth:`compute`
    at the end to get aggregated scalar metrics.

    All inputs are expected in ``[0, data_range]`` with shape
    ``(B, 3, H, W)``.
    """

    def __init__(self, data_range: float = 1.0, use_torchmetrics: bool = True,
                 device: torch.device = None):
        self.data_range = float(data_range)
        self._use_torchmetrics = bool(use_torchmetrics and _HAS_TORCHMETRICS)
        if self._use_torchmetrics:
            self._psnr = PeakSignalNoiseRatio(data_range=self.data_range)
            self._ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range)
            if device is not None:
                self._psnr = self._psnr.to(device)
                self._ssim = self._ssim.to(device)

    # ------------------------------------------------------------------ #
    # Backend helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def backend() -> str:
        return "torchmetrics" if _HAS_TORCHMETRICS else "pure-torch"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        if self._use_torchmetrics:
            self._psnr.reset()
            self._ssim.reset()
        # Clear pure-torch accumulator (it is otherwise not stateless).
        if hasattr(self, "_fallback_buf"):
            self._fallback_buf = {"psnr_sum": 0.0, "ssim_sum": 0.0, "n": 0}

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        pred = pred.clamp(0.0, self.data_range)
        target = target.clamp(0.0, self.data_range)
        if self._use_torchmetrics:
            self._psnr.update(pred, target)
            self._ssim.update(pred, target)
        else:
            # In pure-torch mode we accumulate a (sum_psnr, sum_ssim, count)
            # tuple so compute() can return the average.
            psnr_val = _psnr_torch(pred, target, data_range=self.data_range)
            ssim_val = _ssim_torch(pred, target, data_range=self.data_range)
            n = pred.shape[0]
            if not hasattr(self, "_fallback_buf"):
                self._fallback_buf = {"psnr_sum": 0.0, "ssim_sum": 0.0, "n": 0}
            self._fallback_buf["psnr_sum"] += float(psnr_val) * n
            self._fallback_buf["ssim_sum"] += float(ssim_val) * n
            self._fallback_buf["n"] += n

    def compute(self) -> Dict[str, float]:
        if self._use_torchmetrics:
            return {
                "psnr": float(self._psnr.compute()),
                "ssim": float(self._ssim.compute()),
            }
        buf = getattr(self, "_fallback_buf", None)
        if not buf or buf["n"] == 0:
            return {"psnr": 0.0, "ssim": 0.0}
        return {
            "psnr": buf["psnr_sum"] / buf["n"],
            "ssim": buf["ssim_sum"] / buf["n"],
        }


__all__ = ["ImageQualityMeter"]