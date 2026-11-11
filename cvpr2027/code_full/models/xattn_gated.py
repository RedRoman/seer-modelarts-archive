"""Cross-attention that estimates whether the prior is worth trusting.

Five injection variants -- positional grounding, cross-frame aggregation,
multi-layer keys, exposure decoupling -- all land within noise of the default at
convergence, so the lever is not in how tokens are delivered. What the
measurements do single out is a trust problem:

  * the prior's sign flips with the target domain: +2.80 dB on LOL, -1.43 dB on
    BVI under identical training
  * at convergence, random tokens cost 4.1 dB against no tokens at all, while
    correct tokens buy only 0.6 dB

The model therefore relies on the tokens unconditionally. In-domain that is
roughly free; out of domain the tokens are out of distribution and the same
reliance turns them into a noise source. Every injection variant shares this
behaviour, which is consistent with them all performing alike.

This module gives the model a way to tell. A scalar gate per query location is
predicted from two signals that are available without supervision:

  agreement   how consistent the attended token summary is with the query's own
              content, as a cosine similarity; a prior that describes a
              different scene than the pixels show scores low
  magnitude   the norm of the attended summary relative to its running scale,
              which separates confident tokens from near-empty ones

The gate multiplies the residual, so a value of zero reproduces the ungated
model exactly rather than removing the path. It is initialised to pass
everything through, and the model has to learn to close it.

Falsifiable without training anything: feed random tokens to a trained gated
model and the gate should close, recovering the no-token result rather than the
4.1 dB loss. That check does not depend on the gate helping in-domain.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.pixel_branch_v8 import BiasFreeLayerNorm2d


class VJepaCrossAttnGated(nn.Module):
    """Cross-attn to V-JEPA tokens with a learned per-location reliability gate."""

    def __init__(self, dim: int, vjepa_dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.norm_q = BiasFreeLayerNorm2d(dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=True)
        self.kv = nn.Linear(vjepa_dim, dim * 2, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # two scalar statistics per location -> one gate value
        self.gate = nn.Sequential(
            nn.Conv2d(2, 16, 1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 4.0)   # sigmoid(4) ~ 0.982: open
        self.register_buffer("scale_ema", torch.ones(1), persistent=True)

    def forward(self, x: torch.Tensor, vjepa_kv: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        xn = self.norm_q(x)
        Q = rearrange(self.q(xn), "b (h d) y w -> b h (y w) d", h=self.h)
        kv = self.kv(vjepa_kv).reshape(BT, vjepa_kv.shape[1], 2, self.h, self.d)
        K, V = kv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = rearrange(out, "b h (y w) d -> b (h d) y w", y=H, w=W)

        # agreement between the attended prior and the query's own content
        agree = F.cosine_similarity(out.float(), xn.float(), dim=1, eps=1e-6)
        # magnitude of the attended summary against a slowly tracked scale
        mag = out.float().pow(2).mean(dim=1).clamp_min(1e-12).sqrt()
        if self.training:
            with torch.no_grad():
                self.scale_ema.mul_(0.99).add_(0.01 * mag.mean().detach())
        mag = mag / self.scale_ema.clamp_min(1e-6)

        g = torch.sigmoid(self.gate(torch.stack([agree, mag], dim=1)))
        return x + g.to(out.dtype) * self.proj(out)
