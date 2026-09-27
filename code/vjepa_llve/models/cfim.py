"""Cross-Frame Interaction Module (CFIM) — adapted from LLVE-STCD (IJCAI 2025).

Two flavors share the same long+short-range concept:
  - TokenCFIM: operates on (B, T, M, C) DiT-style tokens (M = H_v * W_v).
  - VolumeCFIM: operates on (B, C, T, H, W) decoder feature volumes.

Long-range branch:
    Cross-frame attention. Each frame queries a global K/V pool aggregated from
    all other frames. Captures slow scene context across the clip.

Short-range branch:
    Adjacent-frame channel concat + 1x1 conv fusion. Captures fast local motion
    without quadratic cost.

Output: x + alpha · long + (1 - alpha) · short, alpha learned, init 0.
Both internal projections are zero-init so CFIM is identity at training start.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCFIM(nn.Module):
    """Cross-frame interaction over DiT tokens grouped per frame.

    Input:  x_btmc (B, T, M, C)   — M tokens per frame
    Output: x_btmc + residual     — same shape; residual ≈ 0 at init
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads

        # Long-range cross-frame attention
        self.norm_l = nn.LayerNorm(dim, eps=1e-6)
        self.q_l = nn.Linear(dim, dim, bias=True)
        self.kv_l = nn.Linear(dim, 2 * dim, bias=True)
        self.proj_l = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj_l.weight)
        nn.init.zeros_(self.proj_l.bias)

        # Short-range temporal channel concat (3-frame window)
        self.norm_s = nn.LayerNorm(dim, eps=1e-6)
        self.short = nn.Linear(3 * dim, dim, bias=True)
        nn.init.zeros_(self.short.weight)
        nn.init.zeros_(self.short.bias)

        # Mixing weight, init 0 → starts as identity
        self.alpha = nn.Parameter(torch.zeros(1))

    def _long_range(self, x_btmc: torch.Tensor) -> torch.Tensor:
        B, T, M, C = x_btmc.shape
        x_norm = self.norm_l(x_btmc)
        # Q from each frame, K/V from ALL frames
        q = self.q_l(x_norm)  # (B, T, M, C)
        kv = self.kv_l(x_norm)  # (B, T, M, 2C)
        # Reshape K/V to global pool (B, T*M, C)
        k, v = kv.chunk(2, dim=-1)
        k_g = k.reshape(B, T * M, C)
        v_g = v.reshape(B, T * M, C)
        # Multi-head
        q = q.reshape(B, T * M, self.h, self.d).transpose(1, 2)  # (B,h,T*M,d)
        k_g = k_g.reshape(B, T * M, self.h, self.d).transpose(1, 2)
        v_g = v_g.reshape(B, T * M, self.h, self.d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k_g, v_g)  # (B,h,T*M,d)
        out = out.transpose(1, 2).reshape(B, T, M, C)
        return self.proj_l(out)

    def _short_range(self, x_btmc: torch.Tensor) -> torch.Tensor:
        # Pad temporal axis with replicate, concat (t-1, t, t+1) along channel.
        # F.pad(replicate) needs 3D input for 1D padding → flatten B*M first.
        B, T, M, C = x_btmc.shape
        x_norm = self.norm_s(x_btmc)
        x_flat = x_norm.permute(0, 2, 3, 1).reshape(B * M, C, T)  # (BM, C, T)
        x_pad = F.pad(x_flat, (1, 1), mode="replicate")           # (BM, C, T+2)
        x_prev = x_pad[..., :-2].reshape(B, M, C, T)
        x_curr = x_pad[..., 1:-1].reshape(B, M, C, T)
        x_next = x_pad[..., 2:].reshape(B, M, C, T)
        # cat along channel → (B, M, 3C, T) → permute (B, T, M, 3C)
        cat = torch.cat([x_prev, x_curr, x_next], dim=2).permute(0, 3, 1, 2).contiguous()
        return self.short(cat)

    def forward(self, x_btmc: torch.Tensor) -> torch.Tensor:
        long = self._long_range(x_btmc)
        short = self._short_range(x_btmc)
        a = self.alpha.sigmoid()
        return x_btmc + a * long + (1 - a) * short


class VolumeCFIM(nn.Module):
    """Cross-frame interaction over decoder feature volumes.

    Input:  x (B, C, T, H, W)
    Output: x + residual
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads

        # Long-range: token-level cross-frame attention after spatial pool to keep cost down.
        # We pool to a coarse grid (k=4) per frame then attend across frames.
        self.pool_size = 4  # down-sampled spatial side per frame for long-range
        self.norm_l = nn.GroupNorm(min(32, dim), dim)
        self.q_l = nn.Conv2d(dim, dim, 1, bias=True)
        self.kv_l = nn.Conv2d(dim, 2 * dim, 1, bias=True)
        self.proj_l = nn.Conv2d(dim, dim, 1, bias=True)
        nn.init.zeros_(self.proj_l.weight)
        nn.init.zeros_(self.proj_l.bias)

        # Short-range: 3D conv with 3x1x1 kernel (zero-init) — cheap temporal fusion
        self.short = nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0), bias=True)
        nn.init.zeros_(self.short.weight)
        nn.init.zeros_(self.short.bias)

        self.alpha = nn.Parameter(torch.zeros(1))

    def _long_range(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x_bcthw.shape
        x_bt = x_bcthw.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x_bt = self.norm_l(x_bt)
        q = self.q_l(x_bt)
        kv = self.kv_l(x_bt)
        # Pool K/V spatially to (pool, pool) per frame to keep attn cost manageable
        ps = min(self.pool_size, H, W)
        k, v = kv.chunk(2, dim=1)
        k_p = F.adaptive_avg_pool2d(k, ps)  # (BT, C, ps, ps)
        v_p = F.adaptive_avg_pool2d(v, ps)
        # Reshape to global K/V (B, T*ps*ps, C)
        k_g = k_p.reshape(B, T, C, ps * ps).permute(0, 1, 3, 2).reshape(B, T * ps * ps, C)
        v_g = v_p.reshape(B, T, C, ps * ps).permute(0, 1, 3, 2).reshape(B, T * ps * ps, C)
        # Q at full resolution, reshape to (B, T*H*W, C)
        q_g = q.reshape(B, T, C, H * W).permute(0, 1, 3, 2).reshape(B, T * H * W, C)
        # Multi-head
        Hd = self.h; Dd = self.d
        q_g = q_g.reshape(B, T * H * W, Hd, Dd).transpose(1, 2)
        k_g = k_g.reshape(B, T * ps * ps, Hd, Dd).transpose(1, 2)
        v_g = v_g.reshape(B, T * ps * ps, Hd, Dd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q_g, k_g, v_g)
        out = out.transpose(1, 2).reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        # Project back at full res via conv (per frame)
        out_bt = out.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        return self.proj_l(out_bt).reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    def _short_range(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        return self.short(x_bcthw)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        long = self._long_range(x_bcthw)
        short = self._short_range(x_bcthw)
        a = self.alpha.sigmoid()
        return x_bcthw + a * long + (1 - a) * short
