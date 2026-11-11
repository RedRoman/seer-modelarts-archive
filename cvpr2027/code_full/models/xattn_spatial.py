"""Spatially grounded cross-attention to V-JEPA tokens (E11 / H10).

The default injection in `pixel_branch_v8.VJepaCrossAttnBF` flattens the token
grid into an unordered set: attention is permutation-invariant over the 256
tokens, so a query at pixel (i, j) has no mechanism to prefer the token covering
its own location and must recover correspondence from content alone.

Adding an explicit pixel-to-token distance bias is the obvious fix and is not
affordable: at the finest scale the bias is (H*W, 256) = (65536, 256) per frame,
which both costs memory and forces scaled_dot_product_attention onto the math
backend, the same failure that cost 8 GB in a single allocation during
evaluation.

This instead gives both sides a *shared* positional basis. One learnable grid is
interpolated to the query's resolution and to the token grid's, and added after
each side is projected into the model dimension. Position then enters through the
existing dot product, so correspondence is rewarded without materialising a bias
matrix and without changing the attention kernel.

Identical in structure to the global variant otherwise, including the zero-init
output projection, so the two differ in the positional term alone.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.pixel_branch_v8 import BiasFreeLayerNorm2d


class VJepaCrossAttnSpatial(nn.Module):
    """Cross-attn from decoder feature (Q) to V-JEPA tokens (K/V), position-aware.

    grid_hw is the token grid (H_v, W_v) the keys arrive on; the positional basis
    is stored at that resolution and interpolated to the query's H x W per call,
    so the module is agnostic to the scale it sits at and to a tiled encoder pass
    that changes the token count.
    """

    def __init__(self, dim: int, vjepa_dim: int, num_heads: int,
                 grid_hw: Tuple[int, int] = (16, 16)) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.grid_hw = tuple(int(g) for g in grid_hw)
        self.norm_q = BiasFreeLayerNorm2d(dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=True)
        self.kv = nn.Linear(vjepa_dim, dim * 2, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # the shared basis: small std so the module starts close to the global
        # variant and has to earn any reliance on position
        self.pos = nn.Parameter(torch.zeros(1, dim, *self.grid_hw))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _pos_for(self, h: int, w: int) -> torch.Tensor:
        if (h, w) == self.grid_hw:
            return self.pos
        return F.interpolate(self.pos, size=(h, w), mode="bicubic",
                             align_corners=False)

    def forward(self, x: torch.Tensor, vjepa_kv: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        M = vjepa_kv.shape[1]

        Q = self.q(self.norm_q(x))
        Q = Q + self._pos_for(H, W).to(Q.dtype)
        Q = rearrange(Q, "b (h d) y w -> b h (y w) d", h=self.h)

        kv = self.kv(vjepa_kv)                                  # (BT, M, 2*dim)
        K, V = kv.chunk(2, dim=-1)
        # tokens may arrive on a grid other than the stored one (a tiled encoder
        # pass multiplies the count); derive the side length from M when square
        side = int(round(M ** 0.5))
        gh, gw = (side, side) if side * side == M else self.grid_hw
        if gh * gw == M:
            pos_k = self._pos_for(gh, gw).flatten(2).transpose(1, 2)
            K = K + pos_k.to(K.dtype)

        K = rearrange(K, "b m (h d) -> b h m d", h=self.h)
        V = rearrange(V, "b m (h d) -> b h m d", h=self.h)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = rearrange(out, "b h (y w) d -> b (h d) y w", y=H, w=W)
        return x + self.proj(out)
