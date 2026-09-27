"""Illumination Condition Prompt Bank (VLU-Net inspired).

VLU-Net (CVPR 2025) introduced a "degradation prompt bank" mechanism:
    - Maintain K learnable prompt tensors per UNet scale: [K, C_s, h_s, w_s]
    - Use a global semantic feature (CLIP image embed in original) to softmax-select
      a mixture weight w over K prompts
    - Inject mixture * prompt as cross-attention K into each scale of decoder

For LLVE we adapt this to **illumination condition**:
    - K=8 illumination conditions (extreme dark, dim, low normal, normal, etc)
    - Use V-JEPA pooled latent (1024-dim global semantics) → Linear(1024→K) →
      softmax → mixture weight
    - Per-scale prompt bank, injected as cross-attn key
    - Channel-wise MDTA cross-attention (O(C²), efficient)

This adds a *task-level discrete prior* on top of v6/v7's spatial-token
cross-attn. Conceptually orthogonal: V-JEPA cross-attn provides scene-content
grounding; this bank provides illumination/noise condition grounding.

Trainable parameters: ~1-2M total across 4 scales (per-scale bank + Linear + cross-attn proj)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class IlluminationConditionRouter(nn.Module):
    """V-JEPA pooled feat (1024,) → softmax mixture over K conditions."""

    def __init__(self, vjepa_dim: int = 1024, num_conditions: int = 8) -> None:
        super().__init__()
        self.K = num_conditions
        self.proj = nn.Linear(vjepa_dim, num_conditions, bias=True)
        # Init bias to encourage uniform mixture early in training
        nn.init.zeros_(self.proj.bias)

    def forward(self, vjepa_pooled: torch.Tensor) -> torch.Tensor:
        """vjepa_pooled: (B, vjepa_dim). Returns mixture weights (B, K)."""
        return F.softmax(self.proj(vjepa_pooled), dim=-1)


class _PromptCrossAttn(nn.Module):
    """MDTA-style channel-wise cross-attention.

    Q from decoder feature (B, C, H, W); K, V from prompt feature (B, C_p, H, W).
    Channel attention is O(C²) regardless of spatial — cheap at all scales.
    Zero-init final projection ⇒ contributes 0 at init.
    """

    def __init__(self, dim: int, prompt_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q from decoder feat
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # K, V from prompt feat (different input channel)
        self.kv_proj = nn.Conv2d(prompt_dim, dim * 2, 1, bias=False)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=False)
        # Output proj zero-init for safe identity-at-init
        self.out = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """x: (BT, dim, H, W); prompt: (BT, prompt_dim, H, W). Returns (BT, dim, H, W)."""
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q_proj(x))
        kv = self.kv_dwconv(self.kv_proj(prompt))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (H c) y x -> b H c (y x)", H=self.h)
        k = rearrange(k, "b (H c) y x -> b H c (y x)", H=self.h)
        v = rearrange(v, "b (H c) y x -> b H c (y x)", H=self.h)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b H c (y x) -> b (H c) y x", y=h, x=w)
        return x + self.out(out)


class IlluminationPromptBank(nn.Module):
    """Per-scale prompt bank with K conditions.

    Stored as nn.Parameter of shape (1, K, prompt_dim, T_p, H_p, W_p).
    Forward: given mixture weights (B, K), produces (B, prompt_dim, T_p, H_p, W_p),
    then bilinear-resample to target (B, prompt_dim, T, H, W) and feed to decoder
    via per-frame cross-attention.

    Total trainable per scale ≈ K * prompt_dim * T_p * H_p * W_p, e.g.
    8 * 96 * 8 * 8 * 8 = 393K. Across 4 scales ≈ 1.5M.
    """

    def __init__(
        self,
        decoder_dim: int,
        num_conditions: int = 8,
        prompt_dim: int = 96,
        prompt_grid_thw: Tuple[int, int, int] = (8, 8, 8),
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.K = num_conditions
        self.prompt_dim = prompt_dim
        self.grid = prompt_grid_thw
        T_p, H_p, W_p = prompt_grid_thw
        self.bank = nn.Parameter(
            torch.randn(1, num_conditions, prompt_dim, T_p, H_p, W_p) * 0.02,
        )
        self.cross_attn = _PromptCrossAttn(decoder_dim, prompt_dim, num_heads=num_heads)

    def forward(
        self,
        decoder_feat: torch.Tensor,            # (BT, dim, H, W)
        mixture_weights: torch.Tensor,         # (B, K)
        target_T: int,
    ) -> torch.Tensor:
        """Compute prompt at target spatial-temporal extent and inject into decoder feat."""
        B = mixture_weights.shape[0]
        K = self.K
        T_p, H_p, W_p = self.grid
        # Weighted mixture: (B, prompt_dim, T_p, H_p, W_p)
        bank = self.bank.expand(B, -1, -1, -1, -1, -1)  # (B, K, C_p, T_p, H_p, W_p)
        w = mixture_weights.view(B, K, 1, 1, 1, 1)
        prompt_5d = (w * bank).sum(dim=1)  # (B, C_p, T_p, H_p, W_p)

        # Resample to (B, C_p, target_T, H, W). decoder_feat is (BT, dim, H, W).
        BT, _, H, W = decoder_feat.shape
        assert BT == B * target_T, f"BT={BT} != B*T={B}*{target_T}"
        # 3D interpolate temporal+spatial
        prompt_5d = F.interpolate(
            prompt_5d, size=(target_T, H, W), mode="trilinear", align_corners=False,
        )
        # Reshape to per-frame (BT, C_p, H, W)
        prompt_bt = prompt_5d.permute(0, 2, 1, 3, 4).reshape(B * target_T, self.prompt_dim, H, W)
        return self.cross_attn(decoder_feat, prompt_bt)
