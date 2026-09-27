"""v6 loss bundle: V-JEPA latent flow + pixel reconstruction.

L = w_flow · MSE(v_pred, v_target)               # latent flow target
  + w_pix  · Charbonnier(pred_pixels, hq)        # pixel accuracy
  + w_lpips · LPIPS(pred_pixels, hq)             # perceptual quality
  + w_temp · ||Δt pred - Δt hq||_1               # temporal consistency
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LPIPSHolder:
    _cache = {}

    @classmethod
    def get(cls, device, net="alex"):
        if device not in cls._cache:
            import lpips  # type: ignore
            m = lpips.LPIPS(net=net).to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            cls._cache[device] = m
        return cls._cache[device]


def _bcthw_to_bchw(x: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


def charbonnier_loss(pred, gt, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((pred - gt) ** 2 + eps ** 2))


def lpips_video_loss(pred, gt, net="alex", chunk: int = 8) -> torch.Tensor:
    device = pred.device
    model = _LPIPSHolder.get(device, net=net)
    p = _bcthw_to_bchw(pred).clamp(0, 1) * 2 - 1
    g = _bcthw_to_bchw(gt).clamp(0, 1) * 2 - 1
    vals = []
    for i in range(0, p.shape[0], chunk):
        vals.append(model(p[i:i + chunk], g[i:i + chunk]))
    return torch.cat(vals, dim=0).mean()


def temporal_dod_loss(pred, gt) -> torch.Tensor:
    if pred.shape[2] < 2:
        return pred.new_zeros(())
    pd = pred[:, :, 1:] - pred[:, :, :-1]
    gd = gt[:, :, 1:] - gt[:, :, :-1]
    return F.l1_loss(pd, gd)


def flow_mse_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    if v_target.requires_grad:
        v_target = v_target.detach()
    return F.mse_loss(v_pred, v_target)


class LLVELoss_v6(nn.Module):
    def __init__(
        self,
        flow_weight: float = 1.0,
        pix_weight: float = 1.0,
        lpips_weight: float = 0.1,
        temporal_weight: float = 0.1,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.pix_weight = pix_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.lpips_net = lpips_net

    def forward(
        self,
        v_pred: Optional[torch.Tensor],
        v_target: Optional[torch.Tensor],
        pred_pixels: torch.Tensor,
        hq: torch.Tensor,
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

        out["total"] = total
        return out
