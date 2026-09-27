"""C-2-Direct ContentAttnAdapter v2 — drop_h_b variant (DBB5.B).

Tests whether h_b is the leakage source. Adapter sees ONLY z_L; queries are
learnable parameters (analogous to BLIP-2 Q-Former / Perceiver Resampler).

Spec: VJEPA_LLVE_CODE_REVIEW.md §DBB5.B.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossAttnBlock(nn.Module):
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


class ContentAttnAdapterNoHb(nn.Module):
    """DBB5.B variant — adapter sees ONLY z_L; queries are learnable parameters.

    Identity-at-init via zero-init scale_heads (same as v1).
    """

    def __init__(
        self,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_dim: int = 1024,
        active_scales: Tuple[int, ...] = (3, 4),
        ref_layer: int = 3,
        attn_heads: int = 8,
        ffn_mult: float = 1.0,
        n_query_tokens: int = 2048,   # same as N_v
    ):
        super().__init__()
        self.vjepa_grid = vjepa_grid
        self.vjepa_dim = vjepa_dim
        self.active_scales = tuple(int(s) for s in active_scales)
        self.ref_layer = int(ref_layer)
        self.n_query_tokens = n_query_tokens

        # Learnable query tokens (Q-Former / Perceiver-style)
        self.q_tokens = nn.Parameter(torch.zeros(1, n_query_tokens, vjepa_dim))
        nn.init.trunc_normal_(self.q_tokens, std=0.02)

        self.body = _CrossAttnBlock(vjepa_dim, heads=attn_heads, ffn_mult=ffn_mult)
        self.scale_heads = nn.ModuleDict({
            str(s): nn.Linear(vjepa_dim, vjepa_dim) for s in self.active_scales
        })
        for s in self.active_scales:
            nn.init.zeros_(self.scale_heads[str(s)].weight)
            nn.init.zeros_(self.scale_heads[str(s)].bias)

    def forward(self, z_L_layers: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        """No h_b. Returns {scale -> KV tokens (B, N_v, vjepa_dim)}."""
        assert len(z_L_layers) > self.ref_layer
        B = z_L_layers[0].shape[0]
        q = self.q_tokens.expand(B, -1, -1)              # (B, N_q, D)
        kv = z_L_layers[self.ref_layer]                   # (B, N_v, D)
        body_out = self.body(q, kv)                       # (B, N_q, D)
        return {s: self.scale_heads[str(s)](body_out) for s in self.active_scales}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
