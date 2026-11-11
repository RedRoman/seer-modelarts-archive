"""v9.1 loss bundle — adds RAFT-based motion-aware warp consistency to v9.0.

Per reviewer J4–J7 lock spec:

L_total = 1.00 · Charbonnier(pred, HQ)
        + 0.10 · LPIPS(pred, HQ)
        + 0.50 · Charbonnier(R̂ ⊙ L̂, HQ)            # Retinex compose aux
        + 0.05 · TV_spatial(L̂)                       # illumination spatial smooth
        + 0.10 · L1(L̂[t] - L̂[t-1])                   # illumination temporal smooth (STCD)
        + 0.01 · L1(R̂_low[t] - R̂_low[t-1])           # ★ R adjacent (lowered from 0.05)
        + 0.05 · DoD(pred, HQ)
        + 0.05 · pred_warp                            # ★ NEW: ||pred_t - W(pred_{t+1}, flow_{t+1→t})||·M
        + 0.10 · R_warp                               # ★ NEW: ||R_t    - W(R_{t+1},    flow_{t+1→t})||·M

Warp target: Option B (self-consistent). Mask M is forward-backward occlusion
mask from RAFT (threshold 1.0 px). Mask-normalized L1: sum(|d|·M)/(sum(M)+eps).

Flow tensors are precomputed per-step in train loop (frozen RAFT-small).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v6 import charbonnier_loss, lpips_video_loss, temporal_dod_loss
from .v7 import retinex_compose_loss, illumination_smoothness_loss
from .v9 import illumination_temporal_smoothness, reflectance_adjacent_low_res


def _flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert pixel flow (B,2,H,W) to grid_sample grid in [-1,1].

    flow[:, 0] = dx in pixels (positive = right)
    flow[:, 1] = dy in pixels (positive = down)
    Returns grid (B, H, W, 2) where grid[..., 0] = x in [-1,1], grid[..., 1] = y in [-1,1].
    """
    B, _, H, W = flow.shape
    device = flow.device
    dtype = flow.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    abs_pos = base + flow
    norm_x = 2.0 * abs_pos[:, 0] / max(W - 1, 1) - 1.0
    norm_y = 2.0 * abs_pos[:, 1] / max(H - 1, 1) - 1.0
    return torch.stack([norm_x, norm_y], dim=-1)


def warp_with_flow(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp src by flow. Both (B,C,H,W). flow encodes target→source displacement.

    Output[i, j] = src[i + flow_y, j + flow_x]
    """
    grid = _flow_to_grid(flow.float()).to(src.dtype)
    return F.grid_sample(
        src, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    )


def occlusion_mask_fb(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    threshold_px: float = 1.0,
) -> torch.Tensor:
    """Forward-backward occlusion mask. (B,2,H,W) flows → (B,1,H,W) in {0,1}.

    Pixel is valid if |flow_fwd + warp(flow_bwd, flow_fwd)| < threshold_px.
    flow_fwd is t→(t+1); flow_bwd is (t+1)→t.
    """
    flow_bwd_warped = warp_with_flow(flow_bwd, flow_fwd)
    diff = flow_fwd + flow_bwd_warped
    err = torch.linalg.norm(diff, dim=1, keepdim=True)
    mask = (err < threshold_px).to(flow_fwd.dtype)
    return mask


def masked_l1(diff: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """sum(|diff| * mask) / (sum(mask) + eps), broadcasting mask across diff channels."""
    abs_diff = diff.abs()
    if mask.shape[1] == 1 and abs_diff.shape[1] > 1:
        mask = mask.expand_as(abs_diff)
    num = (abs_diff * mask).sum()
    den = mask.sum() + eps
    return num / den


def warp_consistency_loss(
    feat: torch.Tensor,
    flow_back: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Self-consistent warp loss across adjacent frames.

    feat:      (B, C, T, H, W) — pred_pixels or R_hat
    flow_back: (B, T-1, 2, H, W) — flow_{t+1 → t} for t = 0..T-2
    mask:      (B, T-1, 1, H, W) — occlusion mask for each pair

    L = mean over (t = 0..T-2) of masked_L1(feat[t] - W(feat[t+1], flow_back[t]))
    """
    B, C, T, H, W = feat.shape
    if T < 2:
        return feat.new_zeros(())
    feat_t = feat[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(B * (T - 1), C, H, W)
    feat_tp1 = feat[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(B * (T - 1), C, H, W)
    fb = flow_back.reshape(B * (T - 1), 2, H, W)
    m = mask.reshape(B * (T - 1), 1, H, W)
    warped = warp_with_flow(feat_tp1, fb)
    return masked_l1(feat_t - warped, m)


class LLVELoss_v9_1(nn.Module):
    """v9.1 = v9.0 losses + pred_warp + R_warp."""

    def __init__(
        self,
        pix_weight: float = 1.0,
        lpips_weight: float = 0.1,
        retinex_compose_weight: float = 0.5,
        L_smooth_spatial_weight: float = 0.05,
        L_smooth_temporal_weight: float = 0.10,
        R_adjacent_weight: float = 0.01,
        R_adjacent_scale: int = 2,
        pred_dod_weight: float = 0.05,
        pred_warp_weight: float = 0.05,
        R_warp_weight: float = 0.10,
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
        self.pred_warp_weight = pred_warp_weight
        self.R_warp_weight = R_warp_weight
        self.lpips_net = lpips_net

    def forward(
        self,
        pred_pixels: torch.Tensor,
        hq: torch.Tensor,
        R_hat: Optional[torch.Tensor] = None,
        L_hat: Optional[torch.Tensor] = None,
        flow_back: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
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

        if self.pred_warp_weight > 0 and flow_back is not None and mask is not None:
            l = warp_consistency_loss(pred_pixels, flow_back, mask)
            out["pred_warp"] = l.detach(); total = total + self.pred_warp_weight * l

        if self.R_warp_weight > 0 and R_hat is not None and flow_back is not None and mask is not None:
            l = warp_consistency_loss(R_hat, flow_back, mask)
            out["R_warp"] = l.detach(); total = total + self.R_warp_weight * l

        out["total"] = total
        return out
