"""C-2-Direct ContentAttnAdapter (DAS1 / DAT-locked).

Replaces JDAP latent delta with V-JEPA-conditioned decoder cross-attn KV.

Architecture (DAS1 + DAT clarifications):
  inputs:
    z_L: list of 4 (B, N_v=2048, 1024)        from frozen V-JEPA
    h_b: (B, 256, T_b=16, H_b=8, W_b=8)       from V2-style decoder.encode WITH kv_layers=None
                                              (DAT2: V-JEPA-clean h_b)

  internal:
    q_proj: Conv3d(256 → 1024, 1×1×1)
    q_resample: trilinear interp h_b spatial dims to V-JEPA grid (8, 16, 16)
    shared cross-attn body: 1 block, dim=1024, heads=8, ffn_mult=1.0
      Q = q_tokens
      K, V = z_L[ref_layer = 3]
    scale_heads: dict { scale_idx -> Linear(1024, 1024, zero-init) }
                 outputs (B, N_v, 1024) per active scale

  output:
    adapter_kv: dict { scale_idx -> tokens (B, N_v, 1024) }

Identity-at-init (DAT4):
  scale_heads zero-initialized → adapter output is exactly zeros at iter 0
  → cross-attn(Q=feat, K=zeros, V=zeros) produces zero attention output
  → adapter_zero_init pred ≈ no_prior pred  (max_abs ≤ 1e-5 smoke gate)

Spec: VJEPA_LLVE_CODE_REVIEW.md §DAS1 / §DAT2 / §DAU.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spatial_to_grid(x: torch.Tensor) -> torch.Tensor:
    """(B, D, T, H, W) -> (B, T*H*W, D)."""
    B, D, T, H, W = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, D).contiguous()


class _CrossAttnBlock(nn.Module):
    """Single cross-attn block: Q attends K,V; residual + tiny FFN.
    to_out and ffn final layer zero-initialized so block output is zero at init
    (combined with scale_heads zero-init → identity-at-init for adapter)."""

    def __init__(self, dim: int, heads: int = 8, ffn_mult: float = 1.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)
        ffn_h = int(dim * ffn_mult)
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_h), nn.GELU(), nn.Linear(ffn_h, dim))
        nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, D = q.shape
        Nk = kv.size(1)
        H = self.heads
        Dh = D // H
        qn = self.norm_q(q)
        kn = self.norm_kv(kv)
        Q = self.to_q(qn).view(B, Nq, H, Dh).transpose(1, 2)
        K = self.to_k(kn).view(B, Nk, H, Dh).transpose(1, 2)
        V = self.to_v(kn).view(B, Nk, H, Dh).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)
        attn = attn.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        out = self.to_out(out)
        q = q + out
        q = q + self.ffn(self.norm_f(q))
        return q


class ContentAttnAdapter(nn.Module):
    """C-2-Direct adapter producing decoder cross-attn KV from (z_L, h_b).

    Args:
        bottleneck_channels: 256 (DeltaModulatedDecoder dims[-1])
        vjepa_grid: (T_v, H_v, W_v) — V-JEPA token grid; default (8, 16, 16) for v11
        vjepa_dim: 1024 (V-JEPA ViT-L dim)
        active_scales: tuple of decoder scale indices to produce KV for; default (3, 4)
        ref_layer: which adapter index in z_L to attend to (default 3 = layer 23)
        attn_heads: 8
        ffn_mult: 1.0 (DAR7 small-first; bump to 2.0 if Stage 1 underfits)
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

        self.q_proj = nn.Conv3d(bottleneck_channels, vjepa_dim, kernel_size=1)
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
                          align_corners=False)                            # (B, D, T_v, H_v, W_v)
        return _spatial_to_grid(x)                                        # (B, N_v, D)

    def forward(
        self,
        z_L_layers: List[torch.Tensor],
        h_b: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Return dict {scale -> KV tokens (B, N_v, vjepa_dim)} for active_scales."""
        assert len(z_L_layers) > self.ref_layer, \
            f"need at least {self.ref_layer+1} z_L layers, got {len(z_L_layers)}"
        q = self._q_tokens(h_b)                                           # (B, N_v, D)
        kv = z_L_layers[self.ref_layer]                                   # (B, N_v, D)
        body_out = self.body(q, kv)                                       # (B, N_v, D)
        return {s: self.scale_heads[str(s)](body_out) for s in self.active_scales}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
