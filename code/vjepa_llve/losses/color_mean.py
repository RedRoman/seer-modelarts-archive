"""v10 M5 — per-frame per-channel mean-color L1 loss.

Penalizes the difference between pred and GT in the spatial mean of each
(batch, channel, frame). This targets the photometric/color-cast component
of the v9.0b residual error (D3 oracle: +3.62 dB outdoor, +0.33 dB indoor).

Spec: V10_DESIGN_SPEC.md §5 / M5.
Weight: 0.1 relative to L_pixel (CE3, CH7-Q2). Do NOT raise to 0.5.
"""
from __future__ import annotations

import torch


def color_mean_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-frame per-channel mean-color L1.

    Args:
        pred: (B, 3, T, H, W) in [0, 1]
        gt:   (B, 3, T, H, W) in [0, 1]

    Returns:
        scalar tensor: mean over (B, C, T) of |spatial_mean(pred) - spatial_mean(gt)|.
    """
    pred_mean = pred.mean(dim=(-2, -1))  # (B, 3, T)
    gt_mean = gt.mean(dim=(-2, -1))
    return (pred_mean - gt_mean).abs().mean()
