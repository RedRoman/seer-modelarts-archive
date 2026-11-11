"""Frozen RAFT-small wrapper for v9.1+ training.

torchvision.models.optical_flow.raft_small expects (N, 3, H, W) images normalized
to [-1, 1]. Returns a list of refined flow tensors; the final entry is the best.

Usage:
    raft = build_raft_small(device=device)
    # video frames in [0,1], shape (B, 3, T, H, W)
    flow_back, mask = compute_flow_and_mask(raft, video_lq)
    # flow_back: (B, T-1, 2, H, W) — flow_{t+1 → t}
    # mask:      (B, T-1, 1, H, W) — forward-backward consistency mask
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from losses.v9_1 import occlusion_mask_fb


def build_raft_small(device: torch.device, weights: str = "C_T_V2") -> nn.Module:
    """Build a frozen RAFT-small. Eval mode, no_grad ready."""
    w = getattr(Raft_Small_Weights, weights)
    model = raft_small(weights=w, progress=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _raft_norm(x: torch.Tensor) -> torch.Tensor:
    """[0,1] → [-1,1] for RAFT."""
    return x * 2.0 - 1.0


@torch.no_grad()
def _raft_pair(raft: nn.Module, img1: torch.Tensor, img2: torch.Tensor,
               n_iter: int = 12) -> torch.Tensor:
    """RAFT flow img1 → img2. Inputs (N,3,H,W) in [0,1]. Returns (N,2,H,W)."""
    # RAFT requires multiples of 8 in spatial dims
    H, W = img1.shape[-2:]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        img1 = torch.nn.functional.pad(img1, (0, pad_w, 0, pad_h))
        img2 = torch.nn.functional.pad(img2, (0, pad_w, 0, pad_h))
    flows = raft(_raft_norm(img1).float(), _raft_norm(img2).float(),
                 num_flow_updates=n_iter)
    flow = flows[-1]
    if pad_h or pad_w:
        flow = flow[..., :H, :W]
    return flow


@torch.no_grad()
def compute_flow_and_mask(
    raft: nn.Module,
    video_lq: torch.Tensor,
    threshold_px: float = 1.0,
    n_iter: int = 12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute backward flow {t+1→t} and FB occlusion mask for adjacent pairs.

    video_lq: (B, 3, T, H, W) in [0,1].
    Returns:
        flow_back: (B, T-1, 2, H, W) — flow_{t+1 → t} (used to warp frame t+1 to t's grid)
        mask:      (B, T-1, 1, H, W) — 1 where forward-backward consistent
    """
    B, C, T, H, W = video_lq.shape
    if T < 2:
        empty_f = video_lq.new_zeros(B, 0, 2, H, W)
        empty_m = video_lq.new_zeros(B, 0, 1, H, W)
        return empty_f, empty_m

    bt = (B * (T - 1))
    img_t = video_lq[:, :, :-1].permute(0, 2, 1, 3, 4).reshape(bt, C, H, W)
    img_tp1 = video_lq[:, :, 1:].permute(0, 2, 1, 3, 4).reshape(bt, C, H, W)

    flow_fwd = _raft_pair(raft, img_t, img_tp1, n_iter=n_iter)   # t → t+1
    flow_bwd = _raft_pair(raft, img_tp1, img_t, n_iter=n_iter)   # t+1 → t

    mask = occlusion_mask_fb(flow_fwd, flow_bwd, threshold_px=threshold_px)

    flow_back = flow_bwd.reshape(B, T - 1, 2, H, W)
    mask_5d = mask.reshape(B, T - 1, 1, H, W)
    return flow_back, mask_5d
