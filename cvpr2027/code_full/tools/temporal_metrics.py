"""Clip-level temporal metrics shared by Seer and baseline evaluation.

All functions take (T, 3, H, W) float tensors in [0, 1] on the GPU.

  tlp      mean LPIPS-alex(pred_t, pred_{t+1})                       (unwarped)
  mabd     mean |mean-luma(pred_t) - mean-luma(pred_{t+1})|
  flicker  per-pixel luma std over T, averaged over pixels
  warp_err mean over t of masked |pred_{t+1} - warp(pred_t, F^gt_{t+1->t})|,
           flow estimated on the GT clip with RAFT-small, occlusions removed
           by a forward-backward consistency check. A temporally frozen
           output is penalised wherever the GT moves, unlike tlp/mabd.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F

RAFT_WEIGHTS = os.environ.get("RAFT_WEIGHTS", "/cache/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth")
_RAFT = None


def _luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]


def _get_raft(device):
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_small
        m = raft_small(weights=None, progress=False)
        m.load_state_dict(torch.load(RAFT_WEIGHTS, map_location="cpu"))
        _RAFT = m.to(device).eval()
    return _RAFT


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp x (B,C,H,W) with flow (B,2,H,W) in pixels."""
    B, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device),
                            indexing="ij")
    gx = (xx[None] + flow[:, 0]) / (W - 1) * 2 - 1
    gy = (yy[None] + flow[:, 1]) / (H - 1) * 2 - 1
    return F.grid_sample(x, torch.stack([gx, gy], dim=-1), mode="bilinear",
                         padding_mode="border", align_corners=True)


@torch.no_grad()
def gt_flows(gt: torch.Tensor):
    """Flow t+1 -> t on the GT clip plus a non-occlusion mask, per adjacent pair."""
    raft = _get_raft(gt.device)
    a = gt[1:] * 2 - 1          # frames t+1
    b = gt[:-1] * 2 - 1         # frames t
    fwd = raft(a, b)[-1]        # t+1 -> t
    bwd = raft(b, a)[-1]        # t -> t+1
    fb = fwd + _warp(bwd, fwd)
    mag = (fwd ** 2).sum(1) + (_warp(bwd, fwd) ** 2).sum(1)
    mask = ((fb ** 2).sum(1) < 0.01 * mag + 0.5).float()[:, None]
    return fwd, mask


@torch.no_grad()
def temporal_metrics(pred: torch.Tensor, lpips_fn, flows=None) -> Dict[str, float]:
    T = pred.shape[0]
    out: Dict[str, float] = {}
    if T < 2:
        return out
    out["tlp"] = float(sum(lpips_fn(pred[t:t + 1], pred[t + 1:t + 2]) for t in range(T - 1)) / (T - 1))
    y = _luma(pred)
    out["mabd"] = float((y.mean(dim=(1, 2))[1:] - y.mean(dim=(1, 2))[:-1]).abs().mean())
    out["flicker"] = float(y.std(dim=0).mean())
    if flows is not None:
        fwd, mask = flows
        warped = _warp(pred[:-1], fwd)
        err = ((pred[1:] - warped).abs().mean(1, keepdim=True) * mask).sum() / mask.sum().clamp_min(1)
        out["warp_err"] = float(err)
    return out


def safe_gt_flows(gt: torch.Tensor) -> Optional[tuple]:
    try:
        return gt_flows(gt)
    except Exception as e:  # keep the main metrics even if RAFT is unavailable
        print(f"[temporal] RAFT unavailable: {e}")
        return None
