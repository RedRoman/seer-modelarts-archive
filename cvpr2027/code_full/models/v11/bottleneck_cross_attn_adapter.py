"""C-1a Bottleneck-Conditioned JDAP Adapter (DAI3/DAI4/DAI5).

Decoder-conditioned cross-attn adapter:
  inputs:
    h_b: (B, C_b=256, T_b=16, H_b=8, W_b=8)   decoder bottleneck feature
    z_L: list of 4 V-JEPA layer features, each (B, N_v=2048, D=1024)
  flow:
    Q tokens = h_b reshaped + linear projection to (B, N_v, D=1024)
              (resampled to V-JEPA grid 8x16x16 via trilinear interpolation)
    For each target adapter idx l in {2, 3} (V-JEPA layers 17, 23):
      one cross-attn block: Q=q_tokens, K,V=z_L[l]
      output delta_l: (B, N_v, D)
  output:
    delta_layers: list of 4
      idx 0, 1 = zeros_like(z_L[i])  (DAI5: shallow layers fixed at zero)
      idx 2, 3 = predicted delta from cross-attn

Spec: VJEPA_LLVE_CODE_REVIEW.md §DAI3-DAI6.
"""
from __future__ import annotations
from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _grid_to_spatial(tokens: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
    """(B, N=T*H*W, D) -> (B, D, T, H, W)."""
    T_v, H_v, W_v = grid
    B, N, D = tokens.shape
    assert N == T_v * H_v * W_v
    return tokens.view(B, T_v, H_v, W_v, D).permute(0, 4, 1, 2, 3).contiguous()


def _spatial_to_grid(x: torch.Tensor) -> torch.Tensor:
    """(B, D, T, H, W) -> (B, T*H*W, D)."""
    B, D, T, H, W = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, D).contiguous()


class _CrossAttnBlock(nn.Module):
    """Single cross-attn block: Q from query stream attends K,V from z_L stream.
    Output is residual added to Q (so init-zero output → identity)."""

    def __init__(self, dim: int, heads: int = 8, ffn_mult: float = 2.0):
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
        # tiny FFN
        ffn_h = int(dim * ffn_mult)
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_h), nn.GELU(), nn.Linear(ffn_h, dim))
        nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Nq, D), kv: (B, Nk, D)
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


class BottleneckCrossAttnAdapter(nn.Module):
    """C-1a adapter (DAI).

    Args:
        bottleneck_channels: decoder h_b channel dim (default 256 from DeltaModulatedDecoder).
        vjepa_grid: V-JEPA token grid (T_v, H_v, W_v); fixed (8, 16, 16) for v11.
        vjepa_dim: V-JEPA feature dim (default 1024 for ViT-L).
        n_layers: number of adapter output layers (matches V-JEPA layer subset; 4 = layers 0/11/17/23).
        active_layer_idx: which adapter indices receive predicted delta (DAI5: 2/3 only).
        attn_heads: cross-attn heads (default 8).
        attn_depth: number of cross-attn blocks per active layer (DAI4: 1).
    """

    def __init__(
        self,
        bottleneck_channels: int = 256,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_dim: int = 1024,
        n_layers: int = 4,
        active_layer_idx: Tuple[int, ...] = (2, 3),
        attn_heads: int = 8,
        attn_depth: int = 1,
    ):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.vjepa_grid = vjepa_grid
        self.vjepa_dim = vjepa_dim
        self.n_layers = n_layers
        self.active_layer_idx = tuple(active_layer_idx)
        self.attn_depth = attn_depth

        # Project h_b channels (256) → V-JEPA dim (1024) BEFORE resampling so
        # cross-attn shares the same dim as z_L tokens.
        self.q_proj = nn.Conv3d(bottleneck_channels, vjepa_dim, kernel_size=1)
        # Per-active-layer attn stacks
        self.attn_blocks = nn.ModuleDict({
            str(l): nn.ModuleList([_CrossAttnBlock(vjepa_dim, heads=attn_heads)
                                   for _ in range(attn_depth)])
            for l in self.active_layer_idx
        })
        # Per-active-layer output projection (zero init → identity-at-init)
        self.out_proj = nn.ModuleDict({
            str(l): nn.Linear(vjepa_dim, vjepa_dim) for l in self.active_layer_idx
        })
        for l in self.active_layer_idx:
            nn.init.zeros_(self.out_proj[str(l)].weight)
            nn.init.zeros_(self.out_proj[str(l)].bias)

    def _q_tokens(self, h_b: torch.Tensor) -> torch.Tensor:
        """Resample h_b (B, 256, T_b, H_b, W_b) → (B, N_v, vjepa_dim).
        Matches V-JEPA grid via trilinear interp + project to vjepa_dim."""
        # project channels first
        x = self.q_proj(h_b)                                              # (B, D, T_b, H_b, W_b)
        # resample to V-JEPA grid (T_v, H_v, W_v)
        x = F.interpolate(x, size=self.vjepa_grid, mode="trilinear",
                          align_corners=False)                            # (B, D, T_v, H_v, W_v)
        return _spatial_to_grid(x)                                        # (B, N_v, D)

    def predict_delta(self, h_b: torch.Tensor, z_L_layers: List[torch.Tensor]
                      ) -> List[torch.Tensor]:
        """Return list of 4 delta tensors (B, N_v, D) aligned with z_L_layers."""
        assert len(z_L_layers) == self.n_layers, \
            f"expected {self.n_layers} z_L layers, got {len(z_L_layers)}"
        q_tokens = self._q_tokens(h_b)                                    # (B, N_v, D)
        out: List[torch.Tensor] = []
        for i in range(self.n_layers):
            if i not in self.active_layer_idx:
                out.append(torch.zeros_like(z_L_layers[i]))
                continue
            kv = z_L_layers[i]                                            # (B, N_v, D)
            x = q_tokens
            for blk in self.attn_blocks[str(i)]:
                x = blk(x, kv)
            delta_i = self.out_proj[str(i)](x)                            # (B, N_v, D), zero-init
            out.append(delta_i)
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
