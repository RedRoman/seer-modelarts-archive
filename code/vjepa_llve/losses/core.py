"""Loss functions for V-JEPA-LLVE v5 (latent flow matching).

Total training loss:
    L = w_flow · MSE(v_pred, v_target)              # rectified flow target
      + w_cf   · L1(refined, E(HQ).detach())        # counterfactual supervision
      + w_lpips · LPIPS(decoded_pred, hq)            # optional perceptual
      + w_temp  · ||Δt decoded_pred - Δt hq||_1     # optional temporal

The flow term drives the velocity field; counterfactual drives the
predictor LoRA to learn LQ→HQ in latent space. LPIPS and temporal are
off by default (decoding is expensive; turn on for quality polish).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LPIPS singleton
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


def _bcthw_to_bchw(x: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------
def flow_mse_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """MSE on the velocity field. v_target = z_HQ - z_LQ for rectified flow LQ→HQ."""
    if v_target.requires_grad:
        # Flow target is constructed from frozen z_LQ/z_HQ; should be detached.
        v_target = v_target.detach()
    return F.mse_loss(v_pred, v_target)


def counterfactual_loss(
    refined: torch.Tensor, e_hq_target: torch.Tensor,
) -> torch.Tensor:
    """L1 between refined V-JEPA features (Predictor output) and E(HQ).

    refined : (B, N, D) — output of CounterfactualRefiner on E(LQ)
    e_hq_target : (B, N, D) — V-JEPA encoder on HQ at the SAME layer (detached)
    """
    if e_hq_target.requires_grad:
        raise ValueError("counterfactual_loss: e_hq_target must be detached")
    if refined.dtype != e_hq_target.dtype:
        e_hq_target = e_hq_target.to(refined.dtype)
    return F.l1_loss(refined, e_hq_target)


def lpips_video_loss(
    pred: torch.Tensor, gt: torch.Tensor, net: str = "alex", chunk: int = 8,
) -> torch.Tensor:
    device = pred.device
    model = _LPIPSHolder.get(device, net=net)
    p = _bcthw_to_bchw(pred).clamp(0, 1) * 2 - 1
    g = _bcthw_to_bchw(gt).clamp(0, 1) * 2 - 1
    vals = []
    for i in range(0, p.shape[0], chunk):
        vals.append(model(p[i:i + chunk], g[i:i + chunk]))
    return torch.cat(vals, dim=0).mean()


def temporal_dod_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 between first-order temporal differences. No flow needed."""
    if pred.shape[2] < 2:
        return pred.new_zeros(())
    pd = pred[:, :, 1:] - pred[:, :, :-1]
    gd = gt[:, :, 1:]   - gt[:, :, :-1]
    return F.l1_loss(pd, gd)


# ---------------------------------------------------------------------------
# Aggregated v5 loss
# ---------------------------------------------------------------------------
class LLVELossV5(nn.Module):
    """Weighted sum of v5 loss terms."""

    def __init__(
        self,
        flow_weight: float = 1.0,
        counterfactual_weight: float = 0.1,
        lpips_weight: float = 0.0,
        temporal_weight: float = 0.0,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.cf_weight = counterfactual_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.lpips_net = lpips_net

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        refined: Optional[torch.Tensor] = None,
        e_hq_target: Optional[torch.Tensor] = None,
        decoded_pred: Optional[torch.Tensor] = None,
        hq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = v_pred.new_zeros(())

        if self.flow_weight > 0:
            l_flow = flow_mse_loss(v_pred, v_target)
            out["flow"] = l_flow.detach()
            total = total + self.flow_weight * l_flow

        if self.cf_weight > 0:
            if refined is None or e_hq_target is None:
                raise ValueError(
                    "counterfactual_weight > 0 requires refined and e_hq_target"
                )
            l_cf = counterfactual_loss(refined, e_hq_target)
            out["counterfactual"] = l_cf.detach()
            total = total + self.cf_weight * l_cf

        if self.lpips_weight > 0:
            if decoded_pred is None or hq is None:
                raise ValueError("lpips_weight > 0 requires decoded_pred and hq")
            l_lpips = lpips_video_loss(decoded_pred, hq, net=self.lpips_net)
            out["lpips"] = l_lpips.detach()
            total = total + self.lpips_weight * l_lpips

        if self.temporal_weight > 0:
            if decoded_pred is None or hq is None:
                raise ValueError("temporal_weight > 0 requires decoded_pred and hq")
            l_temp = temporal_dod_loss(decoded_pred, hq)
            out["temporal"] = l_temp.detach()
            total = total + self.temporal_weight * l_temp

        out["total"] = total
        return out
