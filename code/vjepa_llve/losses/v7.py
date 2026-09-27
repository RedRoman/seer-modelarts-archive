"""v7 loss bundle: pure supervised pixel reconstruction with Retinex auxiliary.

L = w_pix · Charbonnier(pred, hq)              # main reconstruction
  + w_lpips · LPIPS(pred, hq)                  # perceptual
  + w_temp · ||Δt pred - Δt hq||_1             # temporal consistency
  + w_retinex · Charbonnier(R̂ ⊙ L̂, hq)        # Retinex composition supervision
  + w_smoothL · ||∇spatial L̂||_1              # illumination smoothness
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v6 import (
    charbonnier_loss, lpips_video_loss, temporal_dod_loss,
)


def retinex_compose_loss(
    R_hat: torch.Tensor, L_hat: torch.Tensor, hq: torch.Tensor,
) -> torch.Tensor:
    """L1/Charbonnier between R_hat * L_hat and hq.

    R_hat: (B, 3, T, H, W) in [0,1]
    L_hat: (B, 1, T, H, W) in [0,1]
    """
    composed = R_hat * L_hat  # broadcast over channel
    return charbonnier_loss(composed, hq)


def illumination_smoothness_loss(L_hat: torch.Tensor) -> torch.Tensor:
    """Spatial total-variation L1 on illumination map (encourages smooth lighting)."""
    dh = (L_hat[..., 1:, :] - L_hat[..., :-1, :]).abs().mean()
    dw = (L_hat[..., :, 1:] - L_hat[..., :, :-1]).abs().mean()
    return dh + dw


class LLVELoss_v7(nn.Module):
    def __init__(
        self,
        pix_weight: float = 1.0,
        lpips_weight: float = 0.1,
        temporal_weight: float = 0.1,
        retinex_weight: float = 0.5,
        smooth_L_weight: float = 0.05,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.pix_weight = pix_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.retinex_weight = retinex_weight
        self.smooth_L_weight = smooth_L_weight
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
            l_pix = charbonnier_loss(pred_pixels, hq)
            out["pix"] = l_pix.detach()
            total = total + self.pix_weight * l_pix

        if self.lpips_weight > 0:
            l_lpips = lpips_video_loss(pred_pixels, hq, net=self.lpips_net)
            out["lpips"] = l_lpips.detach()
            total = total + self.lpips_weight * l_lpips

        if self.temporal_weight > 0:
            l_temp = temporal_dod_loss(pred_pixels, hq)
            out["temporal"] = l_temp.detach()
            total = total + self.temporal_weight * l_temp

        if self.retinex_weight > 0 and R_hat is not None and L_hat is not None:
            l_ret = retinex_compose_loss(R_hat, L_hat, hq)
            out["retinex"] = l_ret.detach()
            total = total + self.retinex_weight * l_ret

        if self.smooth_L_weight > 0 and L_hat is not None:
            l_sm = illumination_smoothness_loss(L_hat)
            out["smoothL"] = l_sm.detach()
            total = total + self.smooth_L_weight * l_sm

        out["total"] = total
        return out
