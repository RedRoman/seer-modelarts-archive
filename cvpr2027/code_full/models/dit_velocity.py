"""
DiT-based velocity field for v5.0 latent flow matching.

Operates in SD-VAE latent space (4-ch, T x 32 x 32 per video) with patch=2,
producing a velocity prediction v(z_t, t, c) for rectified flow LQ -> HQ.

Architecture:
    z_t (B, 4, T, 32, 32)
       │
       ├─ patchify (Conv3d kernel=(1,2,2), stride=(1,2,2)) ──► (B, dim, T, 16, 16)
       ▼
    tokens (B, T*16*16, dim) + learned 3D pos embed
       │
       ├─ time t → sinusoidal embed → MLP → t_emb (B, dim)
       │
       ├─ N DiT blocks:
       │     - AdaLN-Zero modulation from t_emb
       │     - Spatial self-attn (per frame, 256 tokens)
       │     - Temporal self-attn (per spatial loc, T tokens)
       │     - Cross-attn to V-JEPA features (every 4 blocks)
       │     - GDFN-style MLP
       │
       └─ output proj (zero-init)
              │
              ▼
         velocity (B, 4, T, 32, 32)

Identity-at-init: output proj is zero-init, so v ≡ 0 initially. The ODE
z_{t+dt} = z_t + dt * v(z_t, t) reduces to z(1) = z(0) at init, meaning
no flow happens — pred output is just VAE.decode(z_LQ), which is
approximately LQ (high-quality VAE reconstruction). Training pushes v to
encode the LQ -> HQ trajectory.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------
class TimeEmbedder(nn.Module):
    """Sinusoidal time embedding -> 2-layer MLP."""

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
        # t: (B,) in [0, 1]
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
        emb = self._sinusoidal(t, self.freq_dim)
        return self.mlp(emb)  # (B, dim)


# ---------------------------------------------------------------------------
# AdaLN-Zero modulation (DiT)
# ---------------------------------------------------------------------------
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: (B, N, D), shift/scale: (B, D)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Attention primitives
# ---------------------------------------------------------------------------
class _SDPSelfAttn(nn.Module):
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


class _SDPCrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, ctx_dim: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        _, M, _ = ctx.shape
        q = self.q(x).reshape(B, N, self.h, self.d).transpose(1, 2)
        kv = self.kv(ctx).reshape(B, M, 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------------------
# DiT block with AdaLN-Zero + factorized spatial/temporal + optional cross-attn
# ---------------------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        T: int, H_p: int, W_p: int,
        ffn_mult: float = 4.0,
        ctx_dim: Optional[int] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.T = T; self.H_p = H_p; self.W_p = W_p
        self.use_checkpoint = use_checkpoint

        # Each modulation produces (shift_s, scale_s, gate_s, shift_t, scale_t,
        # gate_t, shift_x, scale_x, gate_x, shift_f, scale_f, gate_f)
        # 4 sub-modules × 3 = 12 modulation values per block.
        # If ctx_dim is None we drop the cross-attn modulations (9 instead).
        self.has_cross = ctx_dim is not None
        n_mods = 12 if self.has_cross else 9
        self.adaln = nn.Linear(dim, n_mods * dim, bias=True)
        # Zero-init the AdaLN: gates start at 0 → block is identity at init,
        # critical for stable diffusion training (DiT paper).
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_s = _SDPSelfAttn(dim, num_heads)

        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_t = _SDPSelfAttn(dim, num_heads)

        if self.has_cross:
            self.norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.attn_c = _SDPCrossAttn(dim, num_heads, ctx_dim)

        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * ffn_mult)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim),
        )

    def _spatial_attn(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        T, H_p, W_p, D = self.T, self.H_p, self.W_p, self.dim
        h = _modulate(self.norm_s(x), shift, scale)
        h = h.reshape(B, T, H_p * W_p, D).reshape(B * T, H_p * W_p, D)
        h = self.attn_s(h)
        h = h.reshape(B, T * H_p * W_p, D)
        return x + gate.unsqueeze(1) * h

    def _temporal_attn(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        T, H_p, W_p, D = self.T, self.H_p, self.W_p, self.dim
        h = _modulate(self.norm_t(x), shift, scale)
        h = h.reshape(B, T, H_p * W_p, D).transpose(1, 2).reshape(B * H_p * W_p, T, D)
        h = self.attn_t(h)
        h = h.reshape(B, H_p * W_p, T, D).transpose(1, 2).reshape(B, T * H_p * W_p, D)
        return x + gate.unsqueeze(1) * h

    def _cross_attn(self, x: torch.Tensor, ctx: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = _modulate(self.norm_c(x), shift, scale)
        h = self.attn_c(h, ctx)
        return x + gate.unsqueeze(1) * h

    def _mlp(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = _modulate(self.norm_f(x), shift, scale)
        h = self.mlp(h)
        return x + gate.unsqueeze(1) * h

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute modulation values from t_emb
        mods = self.adaln(t_emb)
        if self.has_cross and ctx is not None:
            (sh_s, sc_s, ga_s, sh_t, sc_t, ga_t, sh_c, sc_c, ga_c, sh_f, sc_f, ga_f) = mods.chunk(12, dim=1)
        else:
            (sh_s, sc_s, ga_s, sh_t, sc_t, ga_t, sh_f, sc_f, ga_f) = mods.chunk(9, dim=1)

        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self._spatial_attn, x, sh_s, sc_s, ga_s, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._temporal_attn, x, sh_t, sc_t, ga_t, use_reentrant=False)
            if self.has_cross and ctx is not None:
                x = torch.utils.checkpoint.checkpoint(self._cross_attn, x, ctx, sh_c, sc_c, ga_c, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._mlp, x, sh_f, sc_f, ga_f, use_reentrant=False)
        else:
            x = self._spatial_attn(x, sh_s, sc_s, ga_s)
            x = self._temporal_attn(x, sh_t, sc_t, ga_t)
            if self.has_cross and ctx is not None:
                x = self._cross_attn(x, ctx, sh_c, sc_c, ga_c)
            x = self._mlp(x, sh_f, sc_f, ga_f)
        return x


# ---------------------------------------------------------------------------
# Main DiT velocity field
# ---------------------------------------------------------------------------
class DiTVelocityField(nn.Module):
    """DiT operating in SD-VAE latent space, predicting flow velocity.

    Args
    ----
    latent_channels : int
        VAE latent channels (4 for SD VAE).
    latent_h, latent_w : int
        VAE latent spatial size (H/8 = 32 for 256-px input).
    num_frames : int
        Video length T.
    patch : int
        Latent patch size (2 → 16x16 spatial tokens per frame for 32x32 latent).
    dim : int
        DiT hidden dim.
    depth : int
        Number of DiT blocks.
    num_heads : int
        Attention heads.
    ffn_mult : float
        MLP expansion.
    ctx_dim : int
        V-JEPA hidden dim (1024 for ViT-L). Set 0 for no cross-attn.
    cross_attn_blocks : tuple
        Indices of blocks where cross-attention is inserted.
    cross_attn_layer_map : tuple
        Maps each cross-attn block to which V-JEPA layer (0..3).
    use_gradient_checkpoint : bool
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_h: int = 32,
        latent_w: int = 32,
        num_frames: int = 16,
        patch: int = 2,
        dim: int = 384,
        depth: int = 16,
        num_heads: int = 6,
        ffn_mult: float = 4.0,
        ctx_dim: int = 1024,
        cross_attn_blocks: Tuple[int, ...] = (3, 7, 11, 15),
        cross_attn_layer_map: Optional[Tuple[int, ...]] = (0, 1, 2, 3),
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert latent_h % patch == 0 and latent_w % patch == 0
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.T = num_frames
        self.patch = patch
        self.H_p = latent_h // patch
        self.W_p = latent_w // patch
        self.n_tokens = self.T * self.H_p * self.W_p
        self.dim = dim
        self.has_cross = ctx_dim > 0

        # Patchify
        self.patch_embed = nn.Conv3d(
            latent_channels, dim,
            kernel_size=(1, patch, patch), stride=(1, patch, patch), bias=True,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding
        self.t_embedder = TimeEmbedder(dim)

        # Cross-attn schedule
        cross_set = set(int(i) for i in cross_attn_blocks)
        if self.has_cross:
            if cross_attn_layer_map is None:
                cross_attn_layer_map = tuple(i % 4 for i in range(len(cross_attn_blocks)))
            self._cross_layer_for_block = {
                int(b): int(l) for b, l in zip(cross_attn_blocks, cross_attn_layer_map)
            }
        else:
            self._cross_layer_for_block = {}

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, num_heads=num_heads,
                T=self.T, H_p=self.H_p, W_p=self.W_p,
                ffn_mult=ffn_mult,
                ctx_dim=(ctx_dim if i in cross_set else None),
                use_checkpoint=use_gradient_checkpoint,
            )
            for i in range(depth)
        ])

        # Output: AdaLN-Zero + Linear to predict velocity per token, then unpatchify.
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Linear(dim, 2 * dim, bias=True)
        nn.init.zeros_(self.final_adaln.weight)
        nn.init.zeros_(self.final_adaln.bias)
        self.final_proj = nn.Linear(dim, patch * patch * latent_channels, bias=True)
        # Zero-init the final projection so initial velocity prediction is 0.
        # Combined with AdaLN-Zero gates in each block, this gives perfect
        # identity-at-init: v(z_t, t, c) = 0 → z(1) = z(0), so the inference
        # ODE returns z_LQ unchanged, which decodes to ~LQ pixels.
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    @staticmethod
    def _token_to_5d(
        x: torch.Tensor, T: int, H_p: int, W_p: int, patch: int, c: int,
    ) -> torch.Tensor:
        """(B, T*H_p*W_p, patch*patch*c) → (B, c, T, H_p*patch, W_p*patch)."""
        B = x.shape[0]
        x = x.reshape(B, T, H_p, W_p, patch, patch, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5).contiguous()  # (B, c, T, H_p, p, W_p, p)
        x = x.reshape(B, c, T, H_p * patch, W_p * patch)
        return x

    def forward(
        self,
        z_t: torch.Tensor,                  # (B, 4, T, 32, 32)
        t: torch.Tensor,                    # (B,) in [0, 1]
        ctx_layers: Optional[List[torch.Tensor]] = None,  # 4 V-JEPA features (B, N, ctx_dim)
    ) -> torch.Tensor:
        """Returns predicted velocity v of shape (B, 4, T, 32, 32)."""
        B = z_t.shape[0]
        # Patchify
        x = self.patch_embed(z_t)                       # (B, dim, T, H_p, W_p)
        x = rearrange(x, "b d t h w -> b (t h w) d")
        x = x + self.pos_embed

        t_emb = self.t_embedder(t)                      # (B, dim)

        for i, blk in enumerate(self.blocks):
            ctx = None
            if self.has_cross and i in self._cross_layer_for_block and ctx_layers is not None:
                layer_idx = self._cross_layer_for_block[i]
                ctx = ctx_layers[layer_idx]
            x = blk(x, t_emb, ctx)

        # Final AdaLN + projection
        shift, scale = self.final_adaln(t_emb).chunk(2, dim=1)
        x = _modulate(self.final_norm(x), shift, scale)
        x = self.final_proj(x)                          # (B, N, patch*patch*c)

        # Unpatchify back to (B, 4, T, 32, 32)
        v = self._token_to_5d(
            x, self.T, self.H_p, self.W_p, self.patch, self.latent_channels,
        )
        return v
