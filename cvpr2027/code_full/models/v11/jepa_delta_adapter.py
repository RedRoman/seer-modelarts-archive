"""JEPA Delta Adapter (CZ4 / Probe B). Compact compress-attend-expand pattern.

Per V-JEPA layer:
  z_L (B, N, 1024)  -> down (1024->256) -> [transformer x depth at dim=256] -> up (256->1024) -> delta_hat
  z_hat_H = z_L + delta_hat

Compact: ~14M per layer head, 4 heads = ~55M total.
Identity-at-init: out_proj zero-initialized so delta_hat = 0 -> z_hat_H = z_L.

Spec: VJEPA_LLVE_CODE_REVIEW.md §CZ4.A / §CZ8.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn


class _AdapterBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, ffn_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class _PerLayerHead(nn.Module):
    """Compact bottleneck adapter: 1024 -> down_dim -> N transformer blocks -> 1024."""
    def __init__(self, in_dim: int = 1024, bot_dim: int = 256, depth: int = 4, heads: int = 4):
        super().__init__()
        self.down = nn.Linear(in_dim, bot_dim)
        self.blocks = nn.ModuleList([_AdapterBlock(bot_dim, heads) for _ in range(depth)])
        self.up_norm = nn.LayerNorm(bot_dim)
        self.up = nn.Linear(bot_dim, in_dim)
        # zero-init the up-projection so the initial delta_hat = 0 (identity at iter 0).
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.down(z)
        for b in self.blocks: h = b(h)
        return self.up(self.up_norm(h))


class JEPADeltaAdapter(nn.Module):
    """Multi-layer V-JEPA delta adapter (4 V-JEPA layers, independent heads)."""
    def __init__(self, n_layers: int = 4, in_dim: int = 1024,
                 bot_dim: int = 256, depth: int = 4, heads: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            _PerLayerHead(in_dim, bot_dim, depth, heads) for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.in_dim = in_dim

    def forward(self, z_L_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(z_L_layers) == self.n_layers
        return [z + head(z) for z, head in zip(z_L_layers, self.heads)]

    @torch.no_grad()
    def predict_delta(self, z_L_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(z) for z, head in zip(z_L_layers, self.heads)]
