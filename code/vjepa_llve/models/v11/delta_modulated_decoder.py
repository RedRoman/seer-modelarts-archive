"""Delta-Modulated Decoder for v11 Probe C (CZ4 / DH5,DH6).

Compact 3D U-Net (~30-50M trainable) that accepts:
  LQ video                 — always present
  z_L (optional)           — V-JEPA features of LQ, B,N,1024 per layer
  delta (optional)         — per-token modulation source (B,N,1024) per layer
  z_H (optional)           — oracle target latent (substitutes z_L for cross-attn KV)

Same arch handles ALL 6 Probe C conditions:
  no_prior        : decoder(LQ)
  passive_zL      : decoder(LQ, z_L)
  identity_delta  : decoder(LQ, z_L, delta=0)
  shuffled_delta  : decoder(LQ, z_L, delta=delta_from_other_seq)
  oracle_zH       : decoder(LQ, z_H)             [z_H replaces z_L for cross-attn]
  oracle_delta    : decoder(LQ, z_L, delta=z_H-z_L)
  predicted_delta : decoder(LQ, z_L, delta=A(z_L))

Delta interface (per DH6): AdaLN modulation at each decoder scale.
  delta tokens → spatial map → per-channel scale,shift → modulate features
NOT plain cross-attn.

Spec: V11_PROBE_C_SPEC.md §3.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- shared utilities ----------------------------------------------
def _grid_to_spatial(tokens: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
    """(B, N=T*H*W, D) -> (B, D, T, H, W). Token order: t-major, then y, then x."""
    T_v, H_v, W_v = grid
    B, N, D = tokens.shape
    assert N == T_v * H_v * W_v, f"N={N} != T*H*W={T_v*H_v*W_v}"
    return tokens.view(B, T_v, H_v, W_v, D).permute(0, 4, 1, 2, 3).contiguous()


# ----- AdaLN modulator -----------------------------------------------
class AdaLNHead(nn.Module):
    """Project token-grid delta to per-(C,T,H,W) scale + shift, then resize-add to features.
    Zero-initialized so delta=None or delta=0 leaves features unchanged."""
    def __init__(self, vjepa_dim: int = 1024, feat_channels: int = 64):
        super().__init__()
        self.proj = nn.Conv3d(vjepa_dim, 2 * feat_channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor, delta_grid: Optional[torch.Tensor]) -> torch.Tensor:
        # feat: (B, C, T, H, W); delta_grid: (B, vjepa_dim, T_v, H_v, W_v) or None
        if delta_grid is None: return feat
        sh = self.proj(delta_grid)                      # (B, 2C, T_v, H_v, W_v)
        sh = F.interpolate(sh, size=feat.shape[-3:], mode="trilinear", align_corners=False)
        scale, shift = sh.chunk(2, dim=1)
        return feat * (1.0 + scale) + shift


# ----- compact cross-attn block (decoder side) ----------------------
class CrossAttnBlock(nn.Module):
    """At a given decoder scale, pixel queries attend over V-JEPA tokens.
    KV source is z_L (passive) or z_H (oracle_zH)."""
    def __init__(self, channels: int, vjepa_dim: int = 1024, heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(vjepa_dim)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(vjepa_dim, channels)
        self.to_v = nn.Linear(vjepa_dim, channels)
        self.heads = heads
        self.to_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.to_out.weight); nn.init.zeros_(self.to_out.bias)

    def forward(self, feat: torch.Tensor, kv_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        # feat: (B, C, T, H, W); kv_tokens: (B, N, vjepa_dim) or None.
        if kv_tokens is None: return feat
        B, C, T, H, W = feat.shape
        x = feat.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, C)
        q = self.to_q(self.norm_q(x))
        kv_n = self.norm_kv(kv_tokens)
        k = self.to_k(kv_n); v = self.to_v(kv_n)
        # multihead
        def split(t): return t.view(B, t.shape[1], self.heads, C // self.heads).transpose(1, 2)
        q_, k_, v_ = split(q), split(k), split(v)
        attn = (q_ @ k_.transpose(-2, -1)) / math.sqrt(C // self.heads)
        attn = attn.softmax(dim=-1)
        out = (attn @ v_).transpose(1, 2).reshape(B, T*H*W, C)
        out = self.to_out(out)
        out = out.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return feat + out


# ----- residual 3D conv block ----------------------------------------
class ResBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.skip  = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class _Down(nn.Module):
    def __init__(self, c): super().__init__(); self.op = nn.Conv3d(c, c, (1,3,3), (1,2,2), (0,1,1))
    def forward(self, x): return self.op(x)

class _Up(nn.Module):
    def __init__(self, c): super().__init__()
    def forward(self, x): return F.interpolate(x, scale_factor=(1,2,2), mode="nearest")


# ----- decoder -------------------------------------------------------
class DeltaModulatedDecoder(nn.Module):
    """5-scale 3D U-Net with cross-attn (over z_L or z_H) AND AdaLN modulation
    (over delta) at each scale. All optional inputs handled by the same arch
    (passing None or zero is well-defined)."""
    def __init__(
        self,
        in_channels: int = 3,
        dims: Tuple[int, ...] = (32, 48, 96, 192, 256),
        num_blocks: Tuple[int, ...] = (1, 2, 2, 3, 4),
        vjepa_dim: int = 1024,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_layer_per_scale: Tuple[int, ...] = (0, 0, 1, 2, 3),
        attn_at_scales: Tuple[int, ...] = (3, 4),                   # cross-attn ONLY at deepest 2 scales (spatial<=32)
        delta_in_encoder: bool = True,                                # DAH2.C/DAI1: V2 = False (delta only in decoder half)
    ):
        super().__init__()
        assert len(dims) == 5 and len(num_blocks) == 5 and len(vjepa_layer_per_scale) == 5
        self.dims = dims
        self.vjepa_dim = vjepa_dim
        self.vjepa_grid = vjepa_grid
        self.vjepa_layer_per_scale = list(vjepa_layer_per_scale)
        self.attn_at_scales = set(int(s) for s in attn_at_scales)
        self.delta_in_encoder = bool(delta_in_encoder)

        # stem: downsample 2x spatially so top scale is at half-res (saves >4x memory)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, dims[0]), nn.SiLU(),
        )

        # encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_attn   = nn.ModuleList()
        self.enc_adaln  = nn.ModuleList()
        self.downs      = nn.ModuleList()
        c_prev = dims[0]
        for i, (d, nb) in enumerate(zip(dims, num_blocks)):
            stage = nn.Sequential(*[ResBlock3D(c_prev if k==0 else d, d) for k in range(nb)])
            self.enc_blocks.append(stage)
            self.enc_attn.append(CrossAttnBlock(d, vjepa_dim) if i in self.attn_at_scales else nn.Identity())
            self.enc_adaln.append(AdaLNHead(vjepa_dim, d))
            self.downs.append(_Down(d) if i < len(dims) - 1 else nn.Identity())
            c_prev = d

        # decoder (mirror of encoder)
        self.dec_blocks = nn.ModuleList()
        self.dec_attn   = nn.ModuleList()
        self.dec_adaln  = nn.ModuleList()
        self.ups        = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            in_c = dims[i+1] + dims[i]   # skip concat
            out_c = dims[i]
            nb = num_blocks[i]
            stage = nn.Sequential(*[ResBlock3D(in_c if k==0 else out_c, out_c) for k in range(nb)])
            self.dec_blocks.append(stage)
            self.dec_attn.append(CrossAttnBlock(out_c, vjepa_dim) if i in self.attn_at_scales else nn.Identity())
            self.dec_adaln.append(AdaLNHead(vjepa_dim, out_c))
            self.ups.append(_Up(dims[i+1]))

        # head: upsample 2x to recover full res, then conv to RGB
        self.head = nn.Sequential(
            nn.ConvTranspose3d(dims[0], dims[0], (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, dims[0]), nn.SiLU(),
            nn.Conv3d(dims[0], in_channels, 3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # residual prediction

    def _select_kv_for_scale(self, i: int, kv_layers: Optional[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        if kv_layers is None: return None
        l = self.vjepa_layer_per_scale[i]
        return kv_layers[l] if l < len(kv_layers) else None

    def _select_delta_grid(self, i: int, delta_layers: Optional[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        if delta_layers is None: return None
        l = self.vjepa_layer_per_scale[i]
        if l >= len(delta_layers): return None
        return _grid_to_spatial(delta_layers[l], self.vjepa_grid)

    def encode(
        self,
        video_lq: torch.Tensor,
        kv_layers: Optional[List[torch.Tensor]] = None,
        delta_layers: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encoder pass; returns (h_b, skips). When self.delta_in_encoder=False
        the delta is forced to None in the encoder regardless of input.
        h_b is the encoder bottleneck output BEFORE the no-op down at scale 4."""
        x = self.stem(video_lq)
        skips: List[torch.Tensor] = []
        delta_for_enc = delta_layers if self.delta_in_encoder else None
        for i, (blk, attn, adaln, down) in enumerate(zip(
                self.enc_blocks, self.enc_attn, self.enc_adaln, self.downs)):
            x = blk(x)
            kv = self._select_kv_for_scale(i, kv_layers)
            d  = self._select_delta_grid(i, delta_for_enc)
            if isinstance(attn, CrossAttnBlock): x = attn(x, kv)
            x = adaln(x, d)
            skips.append(x)
            if not isinstance(down, nn.Identity): x = down(x)
        return x, skips

    def decode(
        self,
        h_b: torch.Tensor,
        skips: List[torch.Tensor],
        video_lq: torch.Tensor,
        kv_layers: Optional[List[torch.Tensor]] = None,
        delta_layers: Optional[List[torch.Tensor]] = None,
        kv_override: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decoder pass; consumes h_b/skips from encode() and applies delta in dec_adaln.

        DAS2/DAT3: kv_override is a dict {scale_index -> token tensor (B, N, D)}.
        When kv_override[scale] is provided, it REPLACES the kv_layers source at
        that scale (no fallback to raw z_L). Other scales fall back to kv_layers.
        """
        x = h_b
        for j, (blk, attn, adaln, up) in enumerate(zip(
                self.dec_blocks, self.dec_attn, self.dec_adaln, self.ups)):
            x = up(x)
            i = len(self.dims) - 2 - j
            sk = skips[i]
            if x.shape[-3:] != sk.shape[-3:]:
                x = F.interpolate(x, size=sk.shape[-3:], mode="nearest")
            x = torch.cat([x, sk], dim=1)
            x = blk(x)
            if kv_override is not None and i in kv_override:
                kv = kv_override[i]
            else:
                kv = self._select_kv_for_scale(i, kv_layers)
            d  = self._select_delta_grid(i, delta_layers)
            if isinstance(attn, CrossAttnBlock): x = attn(x, kv)
            x = adaln(x, d)
        residual = self.head(x)
        return (video_lq + residual).clamp(0, 1)

    def forward(
        self,
        video_lq: torch.Tensor,
        kv_layers: Optional[List[torch.Tensor]] = None,
        delta_layers: Optional[List[torch.Tensor]] = None,
        return_bottleneck: bool = False,
        kv_override: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """video_lq: (B, 3, T, H, W) in [0,1].
           kv_layers: list of 4 tensors (B, N=T_v*H_v*W_v, vjepa_dim) or None.
           delta_layers: list of 4 tensors same shape, or None.
           kv_override: dict {scale -> tokens} (DAS2/DAT3); replaces kv_layers at given scale.
           return_bottleneck: if True, returns (pred, h_b) for adapter pipelines.
        Returns RGB pred residual added to LQ (residual prediction)."""
        h_b, skips = self.encode(video_lq, kv_layers, delta_layers)
        pred = self.decode(h_b, skips, video_lq, kv_layers, delta_layers,
                           kv_override=kv_override)
        if return_bottleneck:
            return pred, h_b
        return pred
