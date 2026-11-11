"""C-2-Direct ContentAttnAdapter v3 — iREPA conv projector variant (E1).

Replaces 1×1×1 q_proj with 3×3×3 spatial Conv3d, adds spatial-temporal norm.
iREPA evidence: spatial structure preservation is what matters, not semantics.

Spec: VJEPA_LLVE_CODE_REVIEW.md §E1.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spatial_to_grid(x: torch.Tensor) -> torch.Tensor:
    B, D, T, H, W = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, D).contiguous()


def _spatial_norm(x: torch.Tensor) -> torch.Tensor:
    """iREPA-style: subtract mean / divide std along token dim (per-batch, per-channel)."""
    # x: (B, N, D)
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, keepdim=True) + 1e-6
    return (x - m) / s


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ffn_mult: float = 1.0):
        super().__init__()
        self.dim = dim; self.heads = heads
        self.norm_q = nn.LayerNorm(dim); self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)
        ffn_h = int(dim * ffn_mult)
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_h), nn.GELU(), nn.Linear(ffn_h, dim))
        nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q, kv):
        B, Nq, D = q.shape; Nk = kv.size(1); H = self.heads; Dh = D // H
        Q = self.to_q(self.norm_q(q)).view(B, Nq, H, Dh).transpose(1, 2)
        K = self.to_k(self.norm_kv(kv)).view(B, Nk, H, Dh).transpose(1, 2)
        V = self.to_v(self.norm_kv(kv)).view(B, Nk, H, Dh).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        out = self.to_out(out)
        q = q + out
        q = q + self.ffn(self.norm_f(q))
        return q


class ContentAttnAdapterIREPA(nn.Module):
    """E1 variant: 3×3×3 conv projector + spatial-temporal norm on Q tokens.

    Input still includes h_b (so this is C-2-Direct + iREPA projector, not no_hb).
    """

    def __init__(
        self,
        bottleneck_channels: int = 256,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_dim: int = 1024,
        active_scales: Tuple[int, ...] = (3, 4),
        ref_layer: int = 3,
        attn_heads: int = 8,
        ffn_mult: float = 1.0,
    ):
        super().__init__()
        self.vjepa_grid = vjepa_grid
        self.vjepa_dim = vjepa_dim
        self.active_scales = tuple(int(s) for s in active_scales)
        self.ref_layer = int(ref_layer)

        # iREPA: 3x3x3 conv projector instead of 1x1x1
        self.q_proj = nn.Conv3d(bottleneck_channels, vjepa_dim,
                                kernel_size=3, padding=1)

        self.body = _CrossAttnBlock(vjepa_dim, heads=attn_heads, ffn_mult=ffn_mult)
        self.scale_heads = nn.ModuleDict({
            str(s): nn.Linear(vjepa_dim, vjepa_dim) for s in self.active_scales
        })
        for s in self.active_scales:
            nn.init.zeros_(self.scale_heads[str(s)].weight)
            nn.init.zeros_(self.scale_heads[str(s)].bias)

    def _q_tokens(self, h_b: torch.Tensor) -> torch.Tensor:
        x = self.q_proj(h_b)                                              # (B, D, T_b, H_b, W_b)
        x = F.interpolate(x, size=self.vjepa_grid, mode="trilinear",
                          align_corners=False)
        q = _spatial_to_grid(x)                                           # (B, N_v, D)
        return _spatial_norm(q)                                           # iREPA spatial-temporal norm

    def forward(self, z_L_layers, h_b):
        assert len(z_L_layers) > self.ref_layer
        q = self._q_tokens(h_b)
        kv = z_L_layers[self.ref_layer]
        body_out = self.body(q, kv)
        return {s: self.scale_heads[str(s)](body_out) for s in self.active_scales}

    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
