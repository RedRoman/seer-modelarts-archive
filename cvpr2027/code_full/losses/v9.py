"""v9.0 loss bundle — exact spec from reviewer G2.

L_total = 1.00 · Charbonnier(pred, HQ)
        + 0.10 · LPIPS(pred, HQ)
        + 0.50 · Charbonnier(R̂ ⊙ L̂, HQ)            # Retinex compose aux
        + 0.05 · TV_spatial(L̂)                      # illumination spatial smooth
        + 0.10 · L1(L̂[t] - L̂[t-1])                  # ★ illumination temporal smooth (STCD)
        + 0.05 · L1(R̂_low[t] - R̂_low[t-1])          # ★ reflectance adj at half-res (low-weight, motion-tolerant)
        + 0.05 · DoD(pred, HQ)                       # cheap temporal baseline (kept)

Per F2.5 implementation note: R̂ adjacent consistency is computed at half spatial
resolution to be more motion-tolerant. Per G2: this is intentionally low weight
because there is no flow alignment in v9.0; will escalate to 0.20 in v9.1 with RAFT.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v6 import charbonnier_loss, lpips_video_loss, temporal_dod_loss
from .v7 import retinex_compose_loss, illumination_smoothness_loss


def illumination_temporal_smoothness(L_hat: torch.Tensor) -> torch.Tensor:
    """L1 of frame-to-frame difference of L_hat (illumination should change smoothly in time)."""
    if L_hat.shape[2] < 2:
        return L_hat.new_zeros(())
    return torch.abs(L_hat[:, :, 1:] - L_hat[:, :, :-1]).mean()


def reflectance_adjacent_low_res(R_hat: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """L1 of frame-to-frame difference of R_hat at lower spatial resolution.

    Per F2.5 note: avg-pool spatially before adjacent-frame L1 to reduce motion
    over-penalization. Without flow alignment, full-res adjacent L1 would punish
    real motion as flicker; downsampling smooths small motions.
    """
    if R_hat.shape[2] < 2:
        return R_hat.new_zeros(())
    B, C, T, H, W = R_hat.shape
    R_bt = R_hat.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    R_low_bt = F.avg_pool2d(R_bt, kernel_size=scale, stride=scale)
    Hl, Wl = R_low_bt.shape[-2:]
    R_low = R_low_bt.reshape(B, T, C, Hl, Wl).permute(0, 2, 1, 3, 4)
    return torch.abs(R_low[:, :, 1:] - R_low[:, :, :-1]).mean()


class LLVELoss_v9(nn.Module):
    def __init__(
        self,
        pix_weight: float = 1.0,
        lpips_weight: float = 0.1,
        retinex_compose_weight: float = 0.5,
        L_smooth_spatial_weight: float = 0.05,
        L_smooth_temporal_weight: float = 0.10,
        R_adjacent_weight: float = 0.05,
        R_adjacent_scale: int = 2,
        pred_dod_weight: float = 0.05,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.pix_weight = pix_weight
        self.lpips_weight = lpips_weight
        self.retinex_compose_weight = retinex_compose_weight
        self.L_smooth_spatial_weight = L_smooth_spatial_weight
        self.L_smooth_temporal_weight = L_smooth_temporal_weight
        self.R_adjacent_weight = R_adjacent_weight
        self.R_adjacent_scale = int(R_adjacent_scale)
        self.pred_dod_weight = pred_dod_weight
        self.lpips_net = lpips_net

    def forward(
        self,
        pred_pixels: torch.Tensor,
        hq: torch.Tensor,
        R_hat: Optional[torch.Tensor] = None,
        L_hat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = pred_pixels.new_zeros(())

        if self.pix_weight > 0:
            l = charbonnier_loss(pred_pixels, hq)
            out["pix"] = l.detach(); total = total + self.pix_weight * l

        if self.lpips_weight > 0:
            l = lpips_video_loss(pred_pixels, hq, net=self.lpips_net)
            out["lpips"] = l.detach(); total = total + self.lpips_weight * l

        if self.retinex_compose_weight > 0 and R_hat is not None and L_hat is not None:
            l = retinex_compose_loss(R_hat, L_hat, hq)
            out["retinex"] = l.detach(); total = total + self.retinex_compose_weight * l

        if self.L_smooth_spatial_weight > 0 and L_hat is not None:
            l = illumination_smoothness_loss(L_hat)
            out["L_smooth"] = l.detach(); total = total + self.L_smooth_spatial_weight * l

        if self.L_smooth_temporal_weight > 0 and L_hat is not None:
            l = illumination_temporal_smoothness(L_hat)
            out["L_temp"] = l.detach(); total = total + self.L_smooth_temporal_weight * l

        if self.R_adjacent_weight > 0 and R_hat is not None:
            l = reflectance_adjacent_low_res(R_hat, scale=self.R_adjacent_scale)
            out["R_adj"] = l.detach(); total = total + self.R_adjacent_weight * l

        if self.pred_dod_weight > 0:
            l = temporal_dod_loss(pred_pixels, hq)
            out["dod"] = l.detach(); total = total + self.pred_dod_weight * l

        out["total"] = total
        return out
