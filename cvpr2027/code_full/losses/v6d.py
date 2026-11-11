"""v6.D loss bundle: extends v6 with Retinex R/L regularizers in latent.

L = w_flow · MSE(v_pred, v_target)               # latent flow target (unchanged)
  + w_pix  · Charbonnier(pred_pixels, hq)        # pixel accuracy
  + w_lpips · LPIPS(pred_pixels, hq)             # perceptual quality
  + w_temp · ||Δt pred - Δt hq||_1               # temporal consistency
  + w_smoothL  · ||∇spatial z_L||_1              # illumination latent smoothness
  + w_tempR    · ||z_R[t+1] - z_R[t]||_1         # reflectance latent stability

Spatial gradient is computed on the (T, H, W) reshaped z_L (after collapsing
branch dim). Reflectance temporal stability is similar on z_R.

These regularizers are CHEAP — they operate on small latent tensors
(B, T*H*W, dim_branch) and add no parameters.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v6 import (
    charbonnier_loss, flow_mse_loss, lpips_video_loss, temporal_dod_loss,
)


def _reshape_latent_to_grid(
    z: torch.Tensor, grid: Tuple[int, int, int],
) -> torch.Tensor:
    """(B, T*H*W, C) → (B, C, T, H, W)."""
    B, N, C = z.shape
    T, H, W = grid
    assert N == T * H * W, f"latent N={N} != grid product {T*H*W}"
    return z.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


def latent_smoothness_loss(z_L: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
    """Spatial total-variation on illumination latent.

    Encourages z_L to vary smoothly across spatial dims (since illumination is
    locally smooth) while leaving temporal variation unconstrained.
    """
    z = _reshape_latent_to_grid(z_L, grid)  # (B, C, T, H, W)
    dh = (z[..., 1:, :] - z[..., :-1, :]).abs().mean()
    dw = (z[..., :, 1:] - z[..., :, :-1]).abs().mean()
    return dh + dw


def latent_temporal_stability_loss(
    z_R: torch.Tensor, grid: Tuple[int, int, int],
) -> torch.Tensor:
    """Frame-to-frame L1 on reflectance latent.

    Encourages z_R to remain consistent across frames (objects don't change
    color/material instantaneously); ALL motion modeling is left to z_L.
    """
    z = _reshape_latent_to_grid(z_R, grid)  # (B, C, T, H, W)
    if z.shape[2] < 2:
        return z.new_zeros(())
    return (z[:, :, 1:] - z[:, :, :-1]).abs().mean()


class LLVELoss_v6D(nn.Module):
    def __init__(
        self,
        flow_weight: float = 1.0,
        pix_weight: float = 1.0,
        lpips_weight: float = 0.1,
        temporal_weight: float = 0.1,
        smooth_L_weight: float = 0.01,
        temp_R_weight: float = 0.01,
        latent_grid: Tuple[int, int, int] = (8, 16, 16),
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.pix_weight = pix_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.smooth_L_weight = smooth_L_weight
        self.temp_R_weight = temp_R_weight
        self.latent_grid = latent_grid
        self.lpips_net = lpips_net

    def forward(
        self,
        v_pred: Optional[torch.Tensor],
        v_target: Optional[torch.Tensor],
        pred_pixels: torch.Tensor,
        hq: torch.Tensor,
        z_R_out: Optional[torch.Tensor] = None,
        z_L_out: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = pred_pixels.new_zeros(())

        if self.flow_weight > 0 and v_pred is not None and v_target is not None:
            l_flow = flow_mse_loss(v_pred, v_target)
            out["flow"] = l_flow.detach()
            total = total + self.flow_weight * l_flow

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

        if self.smooth_L_weight > 0 and z_L_out is not None:
            l_smooth = latent_smoothness_loss(z_L_out, self.latent_grid)
            out["smoothL"] = l_smooth.detach()
            total = total + self.smooth_L_weight * l_smooth

        if self.temp_R_weight > 0 and z_R_out is not None:
            l_tR = latent_temporal_stability_loss(z_R_out, self.latent_grid)
            out["tempR"] = l_tR.detach()
            total = total + self.temp_R_weight * l_tR

        out["total"] = total
        return out
