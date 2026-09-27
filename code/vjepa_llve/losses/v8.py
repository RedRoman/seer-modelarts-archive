"""v8 loss bundle.

L = w_pix · Charbonnier(pred, hq)
  + w_lpips · LPIPS(pred, hq)
  + w_temp · ||Δt pred - Δt hq||_1
  + w_retinex · Charbonnier(R̂ ⊙ L̂, hq)
  + w_smoothL · ||∇L̂||_1
  + w_counter · L1(z_hq_pred, z_hq_target)            ← counterfactual latent
  + w_mono · Σ_k max(0, cos(z_{k-1}, z_target) − cos(z_k, z_target))  ← monotonic refinement
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .v6 import charbonnier_loss, lpips_video_loss, temporal_dod_loss
from .v7 import retinex_compose_loss, illumination_smoothness_loss


def counterfactual_latent_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(z_pred, z_target.detach())


def monotonicity_loss(
    intermediates: List[torch.Tensor], z_target: torch.Tensor,
) -> torch.Tensor:
    """Encourage cos(z_k, z_target) to monotonically increase across k=0..K."""
    if len(intermediates) <= 1:
        return intermediates[0].new_zeros(())
    z_target_d = z_target.detach()
    z_target_n = F.normalize(z_target_d.flatten(1), dim=-1)

    cosines = []
    for z in intermediates:
        z_n = F.normalize(z.flatten(1), dim=-1)
        cosines.append((z_n * z_target_n).sum(dim=-1).mean())  # scalar per batch

    loss = intermediates[0].new_zeros(())
    for i in range(1, len(cosines)):
        # Penalize when cos(z_i, target) < cos(z_{i-1}, target)
        loss = loss + F.relu(cosines[i - 1] - cosines[i])
    return loss


class LLVELoss_v8(nn.Module):
    def __init__(
        self,
        pix_weight: float = 1.0,
        lpips_weight: float = 0.1,
        temporal_weight: float = 0.1,
        retinex_weight: float = 0.5,
        smooth_L_weight: float = 0.05,
        counter_weight: float = 0.3,
        monotonic_weight: float = 0.05,
        lpips_net: str = "alex",
    ) -> None:
        super().__init__()
        self.pix_weight = pix_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.retinex_weight = retinex_weight
        self.smooth_L_weight = smooth_L_weight
        self.counter_weight = counter_weight
        self.monotonic_weight = monotonic_weight
        self.lpips_net = lpips_net

    def forward(
        self,
        pred_pixels: torch.Tensor,
        hq: torch.Tensor,
        R_hat: Optional[torch.Tensor] = None,
        L_hat: Optional[torch.Tensor] = None,
        z_hq_pred: Optional[torch.Tensor] = None,
        z_hq_target: Optional[torch.Tensor] = None,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = pred_pixels.new_zeros(())

        if self.pix_weight > 0:
            l = charbonnier_loss(pred_pixels, hq)
            out["pix"] = l.detach(); total = total + self.pix_weight * l

        if self.lpips_weight > 0:
            l = lpips_video_loss(pred_pixels, hq, net=self.lpips_net)
            out["lpips"] = l.detach(); total = total + self.lpips_weight * l

        if self.temporal_weight > 0:
            l = temporal_dod_loss(pred_pixels, hq)
            out["temporal"] = l.detach(); total = total + self.temporal_weight * l

        if self.retinex_weight > 0 and R_hat is not None and L_hat is not None:
            l = retinex_compose_loss(R_hat, L_hat, hq)
            out["retinex"] = l.detach(); total = total + self.retinex_weight * l

        if self.smooth_L_weight > 0 and L_hat is not None:
            l = illumination_smoothness_loss(L_hat)
            out["smoothL"] = l.detach(); total = total + self.smooth_L_weight * l

        if self.counter_weight > 0 and z_hq_pred is not None and z_hq_target is not None:
            l = counterfactual_latent_loss(z_hq_pred, z_hq_target)
            out["counter"] = l.detach(); total = total + self.counter_weight * l

        if self.monotonic_weight > 0 and intermediates is not None and z_hq_target is not None:
            l = monotonicity_loss(intermediates, z_hq_target)
            out["mono"] = l.detach(); total = total + self.monotonic_weight * l

        out["total"] = total
        return out
