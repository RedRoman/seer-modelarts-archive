"""Evaluation metrics for low-light video enhancement.

All input tensors are (B, C, T, H, W), float, in [0, 1].
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# PSNR (per-frame, averaged)
# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, gt: torch.Tensor,
         max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    assert pred.shape == gt.shape
    if pred.ndim == 5:
        B, C, T, H, W = pred.shape
        p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        g = gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        p, g = pred, gt
    mse = (p - g).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return (20 * torch.log10(torch.tensor(max_val, device=p.device))
            - 10 * torch.log10(mse)).mean()


# ---------------------------------------------------------------------------
# SSIM (Gaussian window, standard)
# ---------------------------------------------------------------------------
def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_2d(img1: torch.Tensor, img2: torch.Tensor,
             window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    C = img1.shape[1]
    device = img1.device
    win = _gaussian_window(window_size, sigma).to(device)
    window = win[:, None] * win[None, :]
    window = window.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
                / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))
    return ssim_map.mean(dim=(1, 2, 3))


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = pred.shape
    p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    g = gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    return _ssim_2d(p, g).mean()


# ---------------------------------------------------------------------------
# LPIPS (lazy singleton via package `lpips`)
# ---------------------------------------------------------------------------
class _LPIPSHolder:
    _cache = {}

    @classmethod
    def get(cls, device: torch.device, net: str = "alex"):
        if device not in cls._cache:
            import lpips  # type: ignore
            m = lpips.LPIPS(net=net).to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            cls._cache[device] = m
        return cls._cache[device]


def lpips_score(pred: torch.Tensor, gt: torch.Tensor,
                net: str = "alex") -> torch.Tensor:
    """Lower is better. Input: (B, C, T, H, W) in [0, 1]."""
    B, C, T, H, W = pred.shape
    p = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).clamp(0, 1) * 2 - 1
    g = gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).clamp(0, 1) * 2 - 1
    model = _LPIPSHolder.get(pred.device, net=net)
    vals = []
    for i in range(0, p.shape[0], 8):
        vals.append(model(p[i:i + 8], g[i:i + 8]))
    return torch.cat(vals, dim=0).mean()


# ---------------------------------------------------------------------------
# MABD: Mean Absolute Brightness Difference (2026-era LLVE metric, replaces Ewarp)
# ---------------------------------------------------------------------------
def mabd(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Brightness Difference between consecutive frames.

    Used by recent LLVE papers (DL-Diff 2026, NTIRE 2026) as a temporal
    consistency proxy that's robust to global illumination shifts. Lower
    = more consistent.

        MABD = mean over t of | mean_brightness(pred_t) - mean_brightness(pred_{t-1})
                                - (mean_brightness(gt_t) - mean_brightness(gt_{t-1})) |

    pred, gt: (B, C, T, H, W) in [0, 1]. Returns scalar.
    """
    if pred.shape[2] < 2:
        return pred.new_zeros(())
    # Per-frame mean brightness (luminance proxy = mean over RGB)
    pred_b = pred.mean(dim=(1, 3, 4))   # (B, T)
    gt_b   = gt.mean(dim=(1, 3, 4))
    pd_b = pred_b[:, 1:] - pred_b[:, :-1]
    gd_b = gt_b[:, 1:]   - gt_b[:, :-1]
    return (pd_b - gd_b).abs().mean()


# ---------------------------------------------------------------------------
# Ewarp proxy (kept for backward compat; not the main metric anymore)
# ---------------------------------------------------------------------------
def ewarp_proxy(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Difference-of-differences. Same form as our training temporal loss.
    Not directly comparable to published Ewarp numbers (which use RAFT flow)."""
    if pred.shape[2] < 2:
        return pred.new_zeros(())
    pd = pred[:, :, 1:] - pred[:, :, :-1]
    gd = gt[:, :, 1:]   - gt[:, :, :-1]
    return (pd - gd).abs().mean()
