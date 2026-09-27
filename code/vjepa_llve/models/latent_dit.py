"""DiT velocity field operating in V-JEPA's pretrained latent space (v6).

Key design decision (vs v5):
    v5 used SD VAE latent space → external pretrained pixel model in our pipeline,
    diluting the V-JEPA narrative. v6 operates on V-JEPA's token grid directly:
    no external pixel model, no SD VAE. The DiT learns rectified flow in V-JEPA's
    own pretrained 1024-dim feature space, where illumination invariance and
    temporal dynamics are already encoded.

Input:
    z_t: (B, N, 1024) — V-JEPA tokens at the deepest layer (l23)
         where N = T_jepa * H_jepa * W_jepa = 8 * 16 * 16 = 2048
    t:   (B,) in [0, 1] — flow time

Output:
    v_pred: (B, N, 1024) — predicted velocity z_HQ - z_LQ in V-JEPA latent
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------
class TimeEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t, self.freq_dim))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------
class _SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------------------
# DiT block with factorized spatial-temporal attention
# ---------------------------------------------------------------------------
class LatentDiTBlock(nn.Module):
    """One DiT block: AdaLN-Zero + factorized spatial / temporal self-attn + MLP.

    Operates on tokens of shape (B, T*H*W, dim) with known T, H, W.
    Spatial attention runs per frame (H*W tokens), temporal attention per
    spatial location (T tokens). Total attention cost is O((HW)^2 + T^2)
    per block — much cheaper than full O((THW)^2).
    """

    def __init__(
        self, dim: int, num_heads: int,
        T: int, H: int, W: int,
        ffn_mult: float = 4.0, use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.T = T; self.H = H; self.W = W
        self.use_checkpoint = use_checkpoint

        # AdaLN-Zero: 9 modulation values (3 per sub-module: spatial, temporal, MLP)
        self.adaln = nn.Linear(dim, 9 * dim, bias=True)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_s = _SelfAttn(dim, num_heads)

        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_t = _SelfAttn(dim, num_heads)

        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * ffn_mult)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim),
        )

    def _spatial(self, x, sh, sc, ga):
        B = x.shape[0]
        T, H, W, D = self.T, self.H, self.W, self.dim
        h = _modulate(self.norm_s(x), sh, sc)
        h = h.reshape(B, T, H * W, D).reshape(B * T, H * W, D)
        h = self.attn_s(h)
        h = h.reshape(B, T * H * W, D)
        return x + ga.unsqueeze(1) * h

    def _temporal(self, x, sh, sc, ga):
        B = x.shape[0]
        T, H, W, D = self.T, self.H, self.W, self.dim
        h = _modulate(self.norm_t(x), sh, sc)
        h = h.reshape(B, T, H * W, D).transpose(1, 2).reshape(B * H * W, T, D)
        h = self.attn_t(h)
        h = h.reshape(B, H * W, T, D).transpose(1, 2).reshape(B, T * H * W, D)
        return x + ga.unsqueeze(1) * h

    def _mlp(self, x, sh, sc, ga):
        h = _modulate(self.norm_f(x), sh, sc)
        return x + ga.unsqueeze(1) * self.mlp(h)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        sh_s, sc_s, ga_s, sh_t, sc_t, ga_t, sh_f, sc_f, ga_f = self.adaln(t_emb).chunk(9, dim=1)
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self._spatial, x, sh_s, sc_s, ga_s, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._temporal, x, sh_t, sc_t, ga_t, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._mlp, x, sh_f, sc_f, ga_f, use_reentrant=False)
        else:
            x = self._spatial(x, sh_s, sc_s, ga_s)
            x = self._temporal(x, sh_t, sc_t, ga_t)
            x = self._mlp(x, sh_f, sc_f, ga_f)
        return x


# ---------------------------------------------------------------------------
# Main: latent DiT in V-JEPA token grid
# ---------------------------------------------------------------------------
class VJepaLatentDiT(nn.Module):
    """DiT velocity field in V-JEPA latent space.

    Args
    ----
    vjepa_dim : int
        V-JEPA encoder hidden dim (1024 for ViT-L). Input/output is in this dim.
    grid : tuple
        V-JEPA token grid (T, H, W). For frames_per_clip=16 / tubelet=2 / patch=16
        on 256x256: (8, 16, 16).
    dim : int
        DiT internal dim (default 768).
    depth : int
        Number of DiT blocks.
    num_heads : int
        Attention heads.
    ffn_mult : float
    use_gradient_checkpoint : bool

    Forward:
        z_t (B, T*H*W, vjepa_dim), t (B,) -> v_pred (B, T*H*W, vjepa_dim)

    Identity-at-init: final_proj is zero-init, so v_pred = 0 initially. The ODE
    z(1) = z(0) reduces to identity, meaning refined ≡ E(LQ) at step 0.
    """

    def __init__(
        self,
        vjepa_dim: int = 1024,
        grid: Tuple[int, int, int] = (8, 16, 16),
        dim: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        ffn_mult: float = 4.0,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.vjepa_dim = vjepa_dim
        self.dim = dim
        self.T, self.H, self.W = grid
        self.n_tokens = self.T * self.H * self.W

        # Project V-JEPA dim down to DiT working dim, and a learned positional embed
        self.in_proj = nn.Linear(vjepa_dim, dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = TimeEmbedder(dim)

        self.blocks = nn.ModuleList([
            LatentDiTBlock(
                dim=dim, num_heads=num_heads,
                T=self.T, H=self.H, W=self.W,
                ffn_mult=ffn_mult,
                use_checkpoint=use_gradient_checkpoint,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Linear(dim, 2 * dim, bias=True)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.final_proj = nn.Linear(dim, vjepa_dim, bias=True)
        # Zero-init: identity-at-init for the OVERALL v6 model
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, D = z_t.shape
        if N != self.n_tokens:
            raise ValueError(
                f"z_t N={N} != grid product {self.n_tokens} "
                f"(T*H*W = {self.T}*{self.H}*{self.W})"
            )
        x = self.in_proj(z_t) + self.pos_embed
        t_emb = self.t_embedder(t)
        for blk in self.blocks:
            x = blk(x, t_emb)
        shift, scale = self.final_adaln(t_emb).chunk(2, dim=1)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.final_proj(x)

    @torch.no_grad()
    def sample(
        self, z_lq: torch.Tensor, num_steps: int = 8,
    ) -> torch.Tensor:
        """Solve Euler ODE from z_LQ to get refined tokens ≈ z_HQ.

        z_lq: (B, N, vjepa_dim). Returns refined tokens of same shape.
        """
        z = z_lq.clone()
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_k = torch.full(
                (z.shape[0],), k * dt + 0.5 * dt,
                device=z.device, dtype=torch.float32,
            )
            v = self(z, t_k)
            z = z + dt * v
        return z
