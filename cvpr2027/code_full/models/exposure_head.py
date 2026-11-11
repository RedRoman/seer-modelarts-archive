"""Exposure-Decoupled Conditioning (EDC).

Why this exists. V-JEPA was chosen as a prior because its features barely move
when illumination changes. That invariance is also why conditioning on it does
not transfer: the features carry no information about *how bright the target
should be*, so a model trained on SDSD (median GT/input luma ratio 3.5-4.5x)
silently applies that prior to LOL (7.2x) and comes out too dark. Measured on
our splits, normalising each output to the ground-truth mean luma ("GT-mean")
gains every method 4-7 dB on the unseen datasets, i.e. most of the cross-domain
error is global exposure, not structure.

EDC therefore splits the job in two:
  * the pixel branch, conditioned on V-JEPA, predicts *structure* at whatever
    exposure it likes;
  * this head predicts the scalar log-gain separately, from quantities that do
    carry exposure information:
      - statistics of the input clip (mean/percentile luma, dark-region noise),
        which say how dark the capture is, and
      - the pooled V-JEPA token, which says what *kind* of scene it is. The
        feature is illumination-invariant, so it cannot report the current
        brightness, but it does identify the scene, and scene identity carries a
        prior over plausible target brightness (a lit indoor room vs. a night
        street). The head uses it as a prior, not as an observation.

At training time the head is supervised with the true log-gain of the pair; at
inference the predicted gain is applied. Setting --edc_detach keeps the gain
gradient out of the pixel branch so the two paths stay functionally separate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def luma(x: torch.Tensor) -> torch.Tensor:
    """(B,3,T,H,W) -> (B,1,T,H,W) BT.601 luma."""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def input_exposure_stats(lq: torch.Tensor) -> torch.Tensor:
    """(B,3,T,H,W) in [0,1] -> (B,8) exposure-bearing statistics of the input."""
    y = luma(lq).flatten(1)                                  # (B, T*H*W)
    qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=lq.device, dtype=y.dtype)
    perc = torch.quantile(y, qs, dim=1).T                    # (B,5)
    mean = y.mean(1, keepdim=True)
    # noise proxy: std inside the darkest quartile, where sensor noise dominates
    thr = perc[:, 1:2]
    darkm = (y <= thr).float()
    dmean = (y * darkm).sum(1, keepdim=True) / darkm.sum(1, keepdim=True).clamp_min(1)
    dvar = (((y - dmean) ** 2) * darkm).sum(1, keepdim=True) / darkm.sum(1, keepdim=True).clamp_min(1)
    return torch.cat([perc, mean, dmean, dvar.sqrt()], dim=1)  # (B,8)


class ExposureHead(nn.Module):
    """Predicts log-gain from input statistics plus a pooled V-JEPA token."""

    def __init__(self, vjepa_dim: int = 1024, hidden: int = 256, use_vjepa: bool = True):
        super().__init__()
        self.use_vjepa = use_vjepa
        self.stat_mlp = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        if use_vjepa:
            self.sem_mlp = nn.Sequential(nn.LayerNorm(vjepa_dim), nn.Linear(vjepa_dim, hidden),
                                         nn.SiLU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden // 2), nn.SiLU(),
                                 nn.Linear(hidden // 2, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)      # start at log-gain 0 (identity)

    def forward(self, lq: torch.Tensor | None = None,
                vjepa_token: torch.Tensor | None = None,
                stats: torch.Tensor | None = None) -> torch.Tensor:
        # stats lets a caller precompute input_exposure_stats once and keep only
        # the 8 numbers, instead of holding whole clips in memory; passing lq
        # alone is unchanged.
        h = self.stat_mlp(input_exposure_stats(lq) if stats is None else stats)
        if self.use_vjepa and vjepa_token is not None:
            h = h + self.sem_mlp(vjepa_token.mean(dim=1))    # pool over tokens -> (B, D)
        return self.out(h).squeeze(-1)                       # (B,) log-gain


def target_log_gain(lq: torch.Tensor, gt: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Ground-truth log-gain of a pair: log(mean luma GT / mean luma LQ)."""
    a = luma(gt).flatten(1).mean(1).clamp_min(eps)
    b = luma(lq).flatten(1).mean(1).clamp_min(eps)
    return torch.log(a / b)


def apply_gain(pred: torch.Tensor, log_gain: torch.Tensor, ref: torch.Tensor,
               eps: float = 1e-4) -> torch.Tensor:
    """Rescale pred so its mean luma equals exp(log_gain) * mean luma of ref.

    pred/ref: (B,3,T,H,W); log_gain: (B,). The structure path is free to output
    any exposure, so we renormalise it to the requested one rather than simply
    multiplying by the gain.
    """
    cur = luma(pred).flatten(1).mean(1).clamp_min(eps)
    tgt = luma(ref).flatten(1).mean(1).clamp_min(eps) * torch.exp(log_gain)
    s = (tgt / cur).view(-1, 1, 1, 1, 1)
    return (pred * s).clamp(0.0, 1.0)
