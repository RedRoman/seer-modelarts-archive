"""V-JEPA-grounded pixel decoder (v6).

A 4-scale UNet that takes LQ pixels as input and uses V-JEPA's 4 hierarchical
features as multi-scale cross-attention conditioning. Output is the HQ pixel
prediction (residual + LQ).

Design intent (vs v3 V-DUN): V-DUN had 8 GDM blocks doing PGD-style depth
unfolding. v6 keeps the 4-scale Restormer backbone but drops the PGD blocks
(they were a v3-specific bet). Cross-attention to V-JEPA features at every
scale bridges the semantic prior (V-JEPA) with the pixel reconstruction
(decoder). LQ pixels enter as the U-Net input — this preserves textures the
V-JEPA token grid (16x16 spatial) cannot represent.

Architecture:
    LQ pixels (B, 3, T, H, W)
        │ stem Conv3d
        ▼
    enc0  scale=H,         dim=base                 ─┐ skip0
        │ down (PixelUnshuffle 2)                    │
    enc1  scale=H/2,       dim=2*base               ─┐ skip1
        │ down                                        │
    enc2  scale=H/4,       dim=4*base               ─┐ skip2
        │ down                                        │
    enc3  scale=H/8,       dim=8*base               ─┐ skip3
        │ down                                        │
    bot   scale=H/16,      dim=16*base
        │ TemporalMixer + Restormer + cross-attn(layer=3)
        │ up
    dec3  cat skip3 + cross-attn(layer=2 or 3)
        │ up
    dec2  cat skip2 + cross-attn(layer=1 or 2)
        │ up
    dec1  cat skip1 + cross-attn(layer=0 or 1)
        │ up
    dec0  cat skip0 (no cross-attn at finest scale; too costly)
        │ output Conv2d (zero-init)
        ▼
    residual → out = LQ + residual

V-JEPA features at layers [5, 11, 17, 23] all share the same token grid
(T_v, H_v, W_v) = (T/2, H/16, W/16). For 256-px H/W and T=16: (8, 16, 16).
At each decoder scale, V-JEPA tokens are reshaped to (B, vjepa_dim, T_v, H_v, W_v)
then trilinear-resampled to match decoder spatial-temporal extents.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Restormer primitives
# ---------------------------------------------------------------------------
class _LN2d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-5)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class GDFN(nn.Module):
    """Gated DConv FFN (Restormer)."""

    def __init__(self, dim: int, ffn_exp: float = 2.66) -> None:
        super().__init__()
        h = int(dim * ffn_exp)
        self.proj_in = nn.Conv2d(dim, h * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(h * 2, h * 2, 3, padding=1, groups=h * 2, bias=False)
        self.proj_out = nn.Conv2d(h, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        a, b = self.dwconv(x).chunk(2, dim=1)
        return self.proj_out(F.gelu(a) * b)


class MDTA(nn.Module):
    """Multi-DConv Transposed Self-Attn (Restormer): channel attention."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hh, ww = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (H c) y x -> b H c (y x)", H=self.h)
        k = rearrange(k, "b (H c) y x -> b H c (y x)", H=self.h)
        v = rearrange(v, "b (H c) y x -> b H c (y x)", H=self.h)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b H c (y x) -> b (H c) y x", y=hh, x=ww)
        return self.proj(out)


class RestormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, ffn_exp: float = 2.66) -> None:
        super().__init__()
        self.norm1 = _LN2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = _LN2d(dim)
        self.ffn = GDFN(dim, ffn_exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# V-JEPA cross-attention block
# ---------------------------------------------------------------------------
class VJepaCrossAttn(nn.Module):
    """Cross-attention from decoder feature (Q) to V-JEPA tokens (KV)."""

    def __init__(self, dim: int, vjepa_dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.d = dim // num_heads
        self.norm_q = _LN2d(dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.kv = nn.Linear(vjepa_dim, dim * 2, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        # Zero-init the output projection: at init, cross-attn contributes 0,
        # so decoder reduces to plain V-DUN-style UNet without V-JEPA influence.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, vjepa_kv: torch.Tensor,
    ) -> torch.Tensor:
        """x: (BT, C, H, W) decoder feature; vjepa_kv: (BT, M, vjepa_dim) KV tokens.

        Returns x + cross_attn_residual.
        """
        BT, C, H, W = x.shape
        Q = self.q(self.norm_q(x))
        Q = rearrange(Q, "b (h d) y w -> b h (y w) d", h=self.h)  # (BT, h, HW, d)
        kv = self.kv(vjepa_kv).reshape(BT, vjepa_kv.shape[1], 2, self.h, self.d)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, BT, h, M, d)
        K, V = kv.unbind(0)
        out = F.scaled_dot_product_attention(Q, K, V)  # (BT, h, HW, d)
        out = rearrange(out, "b h (y w) d -> b (h d) y w", y=H, w=W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# V-JEPA token spatial alignment helper
# ---------------------------------------------------------------------------
def _vjepa_to_kv_at_scale(
    vjepa_tokens: torch.Tensor,  # (B, N=T_v*H_v*W_v, vjepa_dim)
    grid: Tuple[int, int, int],  # (T_v, H_v, W_v)
    target_T: int,               # decoder temporal extent (T pixel frames)
) -> torch.Tensor:
    """Reshape V-JEPA tokens to per-frame KV: (B*T, M=H_v*W_v, vjepa_dim).

    Aligns V-JEPA's T_v temporal tokens to T decoder frames via nearest
    repeat (each pair of frames shares one V-JEPA token if tubelet=2).
    """
    B, N, D = vjepa_tokens.shape
    T_v, H_v, W_v = grid
    assert N == T_v * H_v * W_v, f"N={N} != T_v*H_v*W_v={T_v*H_v*W_v}"
    x = vjepa_tokens.reshape(B, T_v, H_v * W_v, D)
    if T_v != target_T:
        # Repeat per V-JEPA temporal token to cover (target_T // T_v) decoder frames.
        if target_T % T_v == 0:
            rep = target_T // T_v
            x = x.repeat_interleave(rep, dim=1)
        else:
            # Fallback: linear interpolation along temporal axis on the spatial-flattened tensor
            x_perm = x.permute(0, 2, 3, 1)  # (B, M, D, T_v)
            x_perm = F.interpolate(x_perm, size=target_T, mode="linear", align_corners=False)
            x = x_perm.permute(0, 3, 1, 2)  # (B, target_T, M, D)
    return x.reshape(B * target_T, H_v * W_v, D)


# ---------------------------------------------------------------------------
# Encoder / decoder stages
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    """PixelUnshuffle 2x: keeps total spatial-element count, halves spatial dims, doubles channels."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),  # output channels: (dim/2)*4 = 2*dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """PixelShuffle 2x: doubles spatial dims, halves channels."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),  # output channels: (dim*2)/4 = dim/2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class TemporalMixer3D(nn.Module):
    """3x1x1 Conv3d for cross-frame mixing at the bottleneck."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(min(32, dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(min(32, dim), dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        return x_bcthw + self.body(x_bcthw)


# ---------------------------------------------------------------------------
# Main pixel decoder
# ---------------------------------------------------------------------------
class VJepaPixelDecoder(nn.Module):
    """4-scale UNet: LQ pixels in, HQ pixels out, V-JEPA cross-attn at every scale.

    Args
    ----
    in_channels : 3 (RGB).
    base_dim    : channel width at finest scale (default 48).
                  Per-scale dims are [base, 2*base, 4*base, 8*base, 16*base].
    num_blocks  : list of 5 ints — Restormer block counts at scales
                  [enc0, enc1, enc2, enc3, bottleneck]. Default [2, 4, 4, 6, 8].
    heads       : list of 5 ints — attention heads per scale.
    ffn_exp     : MLP expansion in GDFN.
    vjepa_dim   : V-JEPA encoder hidden dim (1024 for ViT-L).
    vjepa_grid  : V-JEPA token grid (T_v, H_v, W_v) e.g. (8, 16, 16).
    vjepa_layer_per_scale :
        For each decoder scale [enc0..enc3, bottleneck], which V-JEPA layer
        index (0..3) to cross-attn with. Default: [0, 0, 1, 2, 3] — shallow
        V-JEPA at fine spatial decoder scales, deep V-JEPA at coarse scales.
    cross_attn_at_scales :
        Set of decoder scales where cross-attn is applied. Default {2, 3, 4}
        (skip the two finest scales 0,1 to save compute — Q is too large).
    use_gradient_checkpoint : bool
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 48,
        num_blocks: Tuple[int, ...] = (2, 4, 4, 6, 8),
        heads: Tuple[int, ...] = (2, 4, 8, 8, 8),
        ffn_exp: float = 2.66,
        vjepa_dim: int = 1024,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_layer_per_scale: Tuple[int, int, int, int, int] = (0, 0, 1, 2, 3),
        cross_attn_at_scales: Tuple[int, ...] = (2, 3, 4),
        use_gradient_checkpoint: bool = True,
        use_cfim_bottleneck: bool = False,   # v6.D: add VolumeCFIM at bottleneck (zero-init)
    ) -> None:
        super().__init__()
        assert len(num_blocks) == 5 and len(heads) == 5

        self.dims = [base_dim * (2 ** i) for i in range(5)]  # [b, 2b, 4b, 8b, 16b]
        dims = self.dims
        self.vjepa_grid = vjepa_grid
        self.vjepa_layer_per_scale = list(vjepa_layer_per_scale)
        self.cross_attn_scales = set(cross_attn_at_scales)
        ck = use_gradient_checkpoint
        self.use_cfim_bottleneck = use_cfim_bottleneck

        # Stem
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=True)

        # Build encoder stages
        self.enc_blocks = nn.ModuleList()
        self.enc_xattn = nn.ModuleList()  # one per scale (None if not in cross_attn_scales)
        self.enc_down = nn.ModuleList()   # 4 downsamples
        for s in range(5):
            stage = nn.ModuleList([
                RestormerBlock(dims[s], heads[s], ffn_exp) for _ in range(num_blocks[s])
            ])
            self.enc_blocks.append(stage)
            self.enc_xattn.append(
                VJepaCrossAttn(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )
            if s < 4:
                self.enc_down.append(Downsample(dims[s]))

        # Bottleneck temporal mixer (only at deepest scale, full grid match)
        self.temporal_mixer = TemporalMixer3D(dims[4])
        # v6.D: optional VolumeCFIM in parallel with temporal_mixer at bottleneck.
        # Zero-init residual so it does NOT change v6.0 behavior when disabled.
        if self.use_cfim_bottleneck:
            from .cfim import VolumeCFIM
            self.cfim_bottleneck = VolumeCFIM(dims[4], num_heads=8)
        else:
            self.cfim_bottleneck = None

        # Decoder stages: 4 up steps, with skip from enc[3..0]
        self.dec_blocks = nn.ModuleList()
        self.dec_xattn = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        self.dec_reduce = nn.ModuleList()
        for s in range(3, -1, -1):  # s: 3, 2, 1, 0
            self.dec_up.append(Upsample(dims[s + 1]))
            # After up, channels = dims[s+1]/2 = dims[s]. After cat with skip dims[s]: 2*dims[s].
            # Reduce back to dims[s] with 1x1 conv
            self.dec_reduce.append(nn.Conv2d(2 * dims[s], dims[s], 1, bias=False))
            stage = nn.ModuleList([
                RestormerBlock(dims[s], heads[s], ffn_exp)
                for _ in range(num_blocks[s])
            ])
            self.dec_blocks.append(stage)
            # Cross-attn at decoder scale uses original encoder-side mapping (mirror)
            self.dec_xattn.append(
                VJepaCrossAttn(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )

        # Output residual head — zero-init: at step 0 the model is identity (out = LQ).
        self.output_conv = nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.output_conv.weight)

        self.use_checkpoint = use_gradient_checkpoint

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bcthw_to_btchw(x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def _btchw_to_bcthw(x_bt: torch.Tensor, T: int) -> torch.Tensor:
        BT, C, H, W = x_bt.shape
        B = BT // T
        return x_bt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    def _kv_for_scale(
        self, vjepa_layers: List[torch.Tensor], scale_idx: int, target_T: int,
    ) -> Optional[torch.Tensor]:
        """Get per-frame KV for cross-attn at the given decoder scale, if active."""
        if scale_idx not in self.cross_attn_scales:
            return None
        layer_idx = self.vjepa_layer_per_scale[scale_idx]
        return _vjepa_to_kv_at_scale(
            vjepa_layers[layer_idx], self.vjepa_grid, target_T,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        video_lq: torch.Tensor,             # (B, 3, T, H, W) in [0, 1]
        vjepa_layers: List[torch.Tensor],   # 4 layers, each (B, N, vjepa_dim)
    ) -> torch.Tensor:
        if len(vjepa_layers) != 4:
            raise ValueError(
                f"v6 PixelDecoder expects 4 V-JEPA feature layers; got {len(vjepa_layers)}"
            )
        B, _, T, H, W = video_lq.shape
        x_bt = self._bcthw_to_btchw(video_lq)
        x_bt = self.stem(x_bt)

        # ---- encoder ----
        skips = []
        for s in range(5):
            for blk in self.enc_blocks[s]:
                x_bt = blk(x_bt)
            if self.enc_xattn[s] is not None:
                kv = self._kv_for_scale(vjepa_layers, s, target_T=T)
                x_bt = self.enc_xattn[s](x_bt, kv)
            if s < 4:
                skips.append(x_bt)
                x_bt = self.enc_down[s](x_bt)
            else:
                # bottleneck: temporal mixer on (B, C, T, H, W); optional CFIM in parallel
                x5 = self._btchw_to_bcthw(x_bt, T)
                x5 = self.temporal_mixer(x5)
                if self.cfim_bottleneck is not None:
                    x5 = self.cfim_bottleneck(x5)
                x_bt = self._bcthw_to_btchw(x5)

        # ---- decoder ----
        # dec_blocks indexed 0..3 in coarse→fine order corresponds to scales 3, 2, 1, 0
        for i, scale_idx in enumerate([3, 2, 1, 0]):
            x_bt = self.dec_up[i](x_bt)
            x_bt = torch.cat([x_bt, skips[scale_idx]], dim=1)
            x_bt = self.dec_reduce[i](x_bt)
            for blk in self.dec_blocks[i]:
                x_bt = blk(x_bt)
            if self.dec_xattn[i] is not None:
                kv = self._kv_for_scale(vjepa_layers, scale_idx, target_T=T)
                x_bt = self.dec_xattn[i](x_bt, kv)

        # ---- output residual ----
        residual_bt = self.output_conv(x_bt)
        out = self._btchw_to_bcthw(residual_bt, T) + video_lq
        if not self.training:
            out = out.clamp(0.0, 1.0)
        return out
