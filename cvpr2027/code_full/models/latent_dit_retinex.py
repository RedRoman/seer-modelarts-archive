"""Retinex-Decomposed Latent DiT for v6.D.

Architectural change vs v6.0 (NOT just a training trick):
    v6.0 LatentDiT: single 768-dim DiT block sequence; opaque to LLVE structure.
    v6.D LatentDiT: dim is split into two 384-dim branches that explicitly model
                    log-Retinex decomposition of V-JEPA latent:
                        z_t ≈ z_R(t) + z_L(t)
                    where z_R is reflectance content and z_L is illumination.

Why log-Retinex in latent (not pixel) domain:
    Pixel-domain Retinex: I = R · L  (multiplicative)
    log-domain:           log I = log R + log L  (additive, easier to model)
    V-JEPA latent has linear-ish behavior wrt log-luminance under self-supervision,
    so additive decomposition in latent ≈ multiplicative Retinex in pixel.

Block structure (RetinexDiTBlock):
    [z_R, z_L] in (B, N, 384) each
        AdaLN_R modulate                    AdaLN_L modulate
        ↓                                   ↓
        spatial+temporal self-attn (R)      spatial+temporal self-attn (L)
        ↓                                   ↓
        cross-attn R←L  (R queries from L)  cross-attn L←R  (L queries from R)
        ↓                                   ↓
        FFN_R                               FFN_L

Periodic CFIM injection: every cfim_every blocks, run a TokenCFIM on R tokens
(reflectance benefits more from cross-frame context; illumination is already
locally smooth).

Output:
    v_R = final_proj_R(z_R) → (B, N, 1024)   [zero-init]
    v_L = final_proj_L(z_L) → (B, N, 1024)   [zero-init]
    v_pred = v_R + v_L      → (B, N, 1024)   identity-at-init since both zero

Aux outputs (for losses):
    z_R_out, z_L_out at full DiT depth → used for L-smoothness and R-temporal regs.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfim import TokenCFIM


# ---------------------------------------------------------------------------
# Time embedding (same as v6.0 LatentDiT)
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
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t, self.freq_dim))


def _modulate(x, sh, sc):
    return x * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention
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


class _CrossAttn(nn.Module):
    """Q from one branch, K/V from the other. Lightweight cross-branch talk."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        # Zero-init so cross-talk contributes 0 at training start.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        B, N, D = x_q.shape
        q = self.q(x_q).reshape(B, N, self.h, self.d).transpose(1, 2)
        kv = self.kv(x_kv).reshape(B, N, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------------------
# Retinex DiT block (factorized spatial-temporal, dual-branch)
# ---------------------------------------------------------------------------
class RetinexDiTBlock(nn.Module):
    """Dual-branch DiT block: separate R and L pathways with sparse cross-talk."""

    def __init__(
        self, dim_branch: int, num_heads: int,
        T: int, H: int, W: int,
        ffn_mult: float = 4.0, use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim_branch
        self.T, self.H, self.W = T, H, W
        self.use_checkpoint = use_checkpoint

        # AdaLN-Zero per branch: 9 mod values (spatial / temporal / mlp x shift,scale,gate)
        self.adaln_R = nn.Linear(dim_branch * 2, 9 * dim_branch, bias=True)
        self.adaln_L = nn.Linear(dim_branch * 2, 9 * dim_branch, bias=True)
        nn.init.zeros_(self.adaln_R.weight); nn.init.zeros_(self.adaln_R.bias)
        nn.init.zeros_(self.adaln_L.weight); nn.init.zeros_(self.adaln_L.bias)

        self.norm_sR = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.attn_sR = _SelfAttn(dim_branch, num_heads)
        self.norm_tR = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.attn_tR = _SelfAttn(dim_branch, num_heads)

        self.norm_sL = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.attn_sL = _SelfAttn(dim_branch, num_heads)
        self.norm_tL = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.attn_tL = _SelfAttn(dim_branch, num_heads)

        # Cross-branch sparse talk (R queries from L, and vice versa)
        self.norm_xR = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.norm_xL = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.cross_RL = _CrossAttn(dim_branch, num_heads)
        self.cross_LR = _CrossAttn(dim_branch, num_heads)

        # FFN per branch
        self.norm_fR = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        self.norm_fL = nn.LayerNorm(dim_branch, elementwise_affine=False, eps=1e-6)
        h = int(dim_branch * ffn_mult)
        self.mlp_R = nn.Sequential(
            nn.Linear(dim_branch, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim_branch),
        )
        self.mlp_L = nn.Sequential(
            nn.Linear(dim_branch, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim_branch),
        )

    def _spatial(self, x, attn, norm, sh, sc, ga):
        B = x.shape[0]
        T, H, W, D = self.T, self.H, self.W, self.dim
        h = _modulate(norm(x), sh, sc)
        h = h.reshape(B, T, H * W, D).reshape(B * T, H * W, D)
        h = attn(h)
        h = h.reshape(B, T * H * W, D)
        return x + ga.unsqueeze(1) * h

    def _temporal(self, x, attn, norm, sh, sc, ga):
        B = x.shape[0]
        T, H, W, D = self.T, self.H, self.W, self.dim
        h = _modulate(norm(x), sh, sc)
        h = h.reshape(B, T, H * W, D).transpose(1, 2).reshape(B * H * W, T, D)
        h = attn(h)
        h = h.reshape(B, H * W, T, D).transpose(1, 2).reshape(B, T * H * W, D)
        return x + ga.unsqueeze(1) * h

    def _mlp_branch(self, x, mlp, norm, sh, sc, ga):
        h = _modulate(norm(x), sh, sc)
        return x + ga.unsqueeze(1) * mlp(h)

    def _block_inner(self, z_R, z_L, t_emb_concat):
        # AdaLN params per branch from concat([t_emb_R, t_emb_L]) — both branches see joint timestep
        modR = self.adaln_R(t_emb_concat).chunk(9, dim=1)
        modL = self.adaln_L(t_emb_concat).chunk(9, dim=1)
        sR_sh, sR_sc, sR_ga, tR_sh, tR_sc, tR_ga, fR_sh, fR_sc, fR_ga = modR
        sL_sh, sL_sc, sL_ga, tL_sh, tL_sc, tL_ga, fL_sh, fL_sc, fL_ga = modL

        # Self-attn within each branch (spatial + temporal, factorized)
        z_R = self._spatial(z_R, self.attn_sR, self.norm_sR, sR_sh, sR_sc, sR_ga)
        z_L = self._spatial(z_L, self.attn_sL, self.norm_sL, sL_sh, sL_sc, sL_ga)
        z_R = self._temporal(z_R, self.attn_tR, self.norm_tR, tR_sh, tR_sc, tR_ga)
        z_L = self._temporal(z_L, self.attn_tL, self.norm_tL, tL_sh, tL_sc, tL_ga)

        # Cross-branch sparse talk (zero-init, won't disturb identity-at-init)
        z_R = z_R + self.cross_RL(self.norm_xR(z_R), self.norm_xL(z_L))
        z_L = z_L + self.cross_LR(self.norm_xL(z_L), self.norm_xR(z_R))

        # FFN per branch
        z_R = self._mlp_branch(z_R, self.mlp_R, self.norm_fR, fR_sh, fR_sc, fR_ga)
        z_L = self._mlp_branch(z_L, self.mlp_L, self.norm_fL, fL_sh, fL_sc, fL_ga)
        return z_R, z_L

    def forward(self, z_R, z_L, t_emb_R, t_emb_L):
        t_emb_concat = torch.cat([t_emb_R, t_emb_L], dim=-1)
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._block_inner, z_R, z_L, t_emb_concat, use_reentrant=False,
            )
        return self._block_inner(z_R, z_L, t_emb_concat)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class RetinexLatentDiT(nn.Module):
    """v6.D Retinex Latent DiT.

    Forward:
        z_t (B, N, vjepa_dim), t (B,) ->
            v_pred (B, N, vjepa_dim), aux dict with z_R_out / z_L_out
    """

    def __init__(
        self,
        vjepa_dim: int = 1024,
        grid: Tuple[int, int, int] = (8, 16, 16),
        dim: int = 768,             # total dim, split into 2 branches
        depth: int = 16,
        num_heads: int = 12,        # split per branch (heads // 2 each)
        ffn_mult: float = 4.0,
        cfim_every: int = 4,        # insert TokenCFIM on R tokens every K blocks
        cfim_heads: int = 8,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0 and num_heads % 2 == 0
        self.vjepa_dim = vjepa_dim
        self.dim = dim
        self.dim_branch = dim // 2
        self.heads_branch = num_heads // 2
        self.T, self.H, self.W = grid
        self.n_tokens = self.T * self.H * self.W
        self.cfim_every = max(0, int(cfim_every))

        # Project V-JEPA dim -> (R, L) split. Init balance so R = L = z/2 at start.
        self.in_proj_R = nn.Linear(vjepa_dim, self.dim_branch, bias=True)
        self.in_proj_L = nn.Linear(vjepa_dim, self.dim_branch, bias=True)

        # Positional embeds per branch
        self.pos_embed_R = nn.Parameter(torch.zeros(1, self.n_tokens, self.dim_branch))
        self.pos_embed_L = nn.Parameter(torch.zeros(1, self.n_tokens, self.dim_branch))
        nn.init.trunc_normal_(self.pos_embed_R, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_L, std=0.02)

        # Time embedders (separate for R and L; identity-equivalent if needed)
        self.t_embedder_R = TimeEmbedder(self.dim_branch)
        self.t_embedder_L = TimeEmbedder(self.dim_branch)

        # Stack of Retinex blocks
        self.blocks = nn.ModuleList([
            RetinexDiTBlock(
                dim_branch=self.dim_branch, num_heads=self.heads_branch,
                T=self.T, H=self.H, W=self.W,
                ffn_mult=ffn_mult,
                use_checkpoint=use_gradient_checkpoint,
            )
            for _ in range(depth)
        ])

        # CFIM at periodic positions (only on R branch, where cross-frame matters most)
        self.cfim_R = nn.ModuleList()
        self.cfim_positions = []
        if self.cfim_every > 0:
            for i in range(depth):
                if (i + 1) % self.cfim_every == 0 and (i + 1) < depth:
                    self.cfim_positions.append(i)
                    self.cfim_R.append(TokenCFIM(self.dim_branch, num_heads=cfim_heads))

        # Final modulation + projection per branch
        self.final_norm_R = nn.LayerNorm(self.dim_branch, elementwise_affine=False, eps=1e-6)
        self.final_norm_L = nn.LayerNorm(self.dim_branch, elementwise_affine=False, eps=1e-6)
        self.final_adaln_R = nn.Linear(self.dim_branch * 2, 2 * self.dim_branch, bias=True)
        self.final_adaln_L = nn.Linear(self.dim_branch * 2, 2 * self.dim_branch, bias=True)
        nn.init.zeros_(self.final_adaln_R.weight); nn.init.zeros_(self.final_adaln_R.bias)
        nn.init.zeros_(self.final_adaln_L.weight); nn.init.zeros_(self.final_adaln_L.bias)

        self.final_proj_R = nn.Linear(self.dim_branch, vjepa_dim, bias=True)
        self.final_proj_L = nn.Linear(self.dim_branch, vjepa_dim, bias=True)
        nn.init.zeros_(self.final_proj_R.weight); nn.init.zeros_(self.final_proj_R.bias)
        nn.init.zeros_(self.final_proj_L.weight); nn.init.zeros_(self.final_proj_L.bias)

    def _apply_cfim(self, z_R, cfim_idx):
        # z_R shape (B, T*H*W, C_branch) -> (B, T, H*W, C) for TokenCFIM
        B, N, C = z_R.shape
        z_btmc = z_R.reshape(B, self.T, self.H * self.W, C)
        z_btmc = self.cfim_R[cfim_idx](z_btmc)
        return z_btmc.reshape(B, N, C)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor):
        B, N, D = z_t.shape
        if N != self.n_tokens:
            raise ValueError(
                f"z_t N={N} != grid product {self.n_tokens} "
                f"(T*H*W = {self.T}*{self.H}*{self.W})"
            )
        z_R = self.in_proj_R(z_t) + self.pos_embed_R
        z_L = self.in_proj_L(z_t) + self.pos_embed_L
        t_emb_R = self.t_embedder_R(t)
        t_emb_L = self.t_embedder_L(t)

        cfim_iter = iter(range(len(self.cfim_R)))
        for i, blk in enumerate(self.blocks):
            z_R, z_L = blk(z_R, z_L, t_emb_R, t_emb_L)
            if i in self.cfim_positions:
                z_R = self._apply_cfim(z_R, next(cfim_iter))

        # Final modulate + project, per branch
        t_emb_concat = torch.cat([t_emb_R, t_emb_L], dim=-1)
        sh_R, sc_R = self.final_adaln_R(t_emb_concat).chunk(2, dim=1)
        sh_L, sc_L = self.final_adaln_L(t_emb_concat).chunk(2, dim=1)
        z_R_out = _modulate(self.final_norm_R(z_R), sh_R, sc_R)
        z_L_out = _modulate(self.final_norm_L(z_L), sh_L, sc_L)
        v_R = self.final_proj_R(z_R_out)
        v_L = self.final_proj_L(z_L_out)
        v_pred = v_R + v_L

        return {
            "v_pred": v_pred,
            "v_R": v_R,
            "v_L": v_L,
            "z_R_out": z_R_out,
            "z_L_out": z_L_out,
        }

    @torch.no_grad()
    def sample(self, z_lq: torch.Tensor, num_steps: int = 8) -> torch.Tensor:
        z = z_lq.clone()
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_k = torch.full(
                (z.shape[0],), k * dt + 0.5 * dt,
                device=z.device, dtype=torch.float32,
            )
            v = self.forward(z, t_k)["v_pred"]
            z = z + dt * v
        return z
