"""V-JEPA-LLVE v8 Pixel Branch — big UNet decoder with:
  - BiasFree LayerNorm (VLU-Net style)
  - V-JEPA multi-scale cross-attn (existing)
  - Illumination Prompt Bank cross-attn (VLU-Net inspired, K=8 conditions)
  - Pixel parallel CNN encoder path (preserves high-frequency detail)
  - Bottleneck VolumeCFIM (cross-frame, from v6.D)
  - Retinex dual-head (R̂, L̂) + residual head

Architectural break from v7:
    v7 was "single LQ pixel input → big UNet → output". Information bottleneck
    was V-JEPA cross-attn alone.
    v8 adds:
      (1) Pixel parallel CNN path that bypasses V-JEPA — high-freq guarantee
      (2) Illumination prompt bank — task-level discrete prior, complements spatial cross-attn
      (3) BiasFree LN — better dark-region stability
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .biasfree_ln import BiasFreeLayerNorm2d
from .cfim import VolumeCFIM
from .illumination_prompt_bank import IlluminationPromptBank


# ---------------------------------------------------------------------------
# Restormer primitives with BiasFree LN
# ---------------------------------------------------------------------------
class GDFN(nn.Module):
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


class RestormerBlockBF(nn.Module):
    """Restormer block with BiasFree LN."""

    def __init__(self, dim: int, num_heads: int = 4, ffn_exp: float = 2.66) -> None:
        super().__init__()
        self.norm1 = BiasFreeLayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = BiasFreeLayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# V-JEPA cross-attention (decoder Q ← V-JEPA K/V)
# ---------------------------------------------------------------------------
def _vjepa_to_kv_at_scale(
    vjepa_tokens: torch.Tensor,
    grid: Tuple[int, int, int],
    target_T: int,
) -> torch.Tensor:
    """(B, N, D) → (B*T, M, D) per-frame KV for cross-attn."""
    B, N, D = vjepa_tokens.shape
    T_v, H_v, W_v = grid
    assert N == T_v * H_v * W_v
    x = vjepa_tokens.reshape(B, T_v, H_v * W_v, D)
    if T_v != target_T:
        if target_T % T_v == 0:
            rep = target_T // T_v
            x = x.repeat_interleave(rep, dim=1)
        else:
            x_perm = x.permute(0, 2, 3, 1)
            x_perm = F.interpolate(x_perm, size=target_T, mode="linear", align_corners=False)
            x = x_perm.permute(0, 3, 1, 2)
    return x.reshape(B * target_T, H_v * W_v, D)


class VJepaCrossAttnBF(nn.Module):
    """Cross-attn from decoder feature (Q) to V-JEPA tokens (K/V), BiasFree."""

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

    def forward(self, x: torch.Tensor, vjepa_kv: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        Q = self.q(self.norm_q(x))
        Q = rearrange(Q, "b (h d) y w -> b h (y w) d", h=self.h)
        kv = self.kv(vjepa_kv).reshape(BT, vjepa_kv.shape[1], 2, self.h, self.d)
        kv = kv.permute(2, 0, 3, 1, 4)
        K, V = kv.unbind(0)
        out = F.scaled_dot_product_attention(Q, K, V)
        out = rearrange(out, "b h (y w) d -> b (h d) y w", y=H, w=W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Up/Down sampling (PixelShuffle style)
# ---------------------------------------------------------------------------
class FlexDown(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 4, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FlexUp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Pixel parallel CNN encoder (high-freq bypass)
# ---------------------------------------------------------------------------
class PixelParallelEncoder(nn.Module):
    """Lightweight CNN encoder of LQ video, produces 4-scale features matching decoder.

    Returns features at scales [256, 128, 64, 32] resolutions, dims tied to decoder dims.
    Operates per-frame (B, T) collapsed for stage processing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Tuple[int, int, int, int] = (32, 64, 128, 192),
    ) -> None:
        super().__init__()
        self.dims = list(dims)
        # Stage 0 at 256x256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=False),
            BiasFreeLayerNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=False),
        )
        # Down 0→1 at 128x128
        self.down0 = FlexDown(dims[0], dims[1])
        self.stage1 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], 3, padding=1, bias=False),
            BiasFreeLayerNorm2d(dims[1]),
            nn.GELU(),
            nn.Conv2d(dims[1], dims[1], 3, padding=1, bias=False),
        )
        # Down 1→2 at 64
        self.down1 = FlexDown(dims[1], dims[2])
        self.stage2 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], 3, padding=1, bias=False),
            BiasFreeLayerNorm2d(dims[2]),
            nn.GELU(),
            nn.Conv2d(dims[2], dims[2], 3, padding=1, bias=False),
        )
        # Down 2→3 at 32
        self.down2 = FlexDown(dims[2], dims[3])
        self.stage3 = nn.Sequential(
            nn.Conv2d(dims[3], dims[3], 3, padding=1, bias=False),
            BiasFreeLayerNorm2d(dims[3]),
            nn.GELU(),
            nn.Conv2d(dims[3], dims[3], 3, padding=1, bias=False),
        )

    def forward(self, x_bt: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.stem(x_bt)         # (BT, dims[0], 256, 256)
        f1 = self.stage1(self.down0(f0))  # (BT, dims[1], 128, 128)
        f2 = self.stage2(self.down1(f1))  # (BT, dims[2], 64, 64)
        f3 = self.stage3(self.down2(f2))  # (BT, dims[3], 32, 32)
        return [f0, f1, f2, f3]


# ---------------------------------------------------------------------------
# Main: big v8 pixel decoder
# ---------------------------------------------------------------------------
class TemporalMixer3D(nn.Module):
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


class VJepaPixelBranch_v8(nn.Module):
    """v8 big pixel UNet with all enhancements.

    Inputs:
      video_lq (B, 3, T, H, W)
      vjepa_layers: List[4] of (B, N, vjepa_dim)
      vjepa_pooled (B, vjepa_dim) — global semantic for prompt bank routing

    Output dict:
      pred_pixels (B, 3, T, H, W) — residual head: LQ + residual
      R_hat (B, 3, T, H, W), L_hat (B, 1, T, H, W) — Retinex aux
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Tuple[int, int, int, int, int] = (64, 128, 256, 512, 1024),
        num_blocks: Tuple[int, ...] = (4, 8, 8, 12, 16),
        heads: Tuple[int, ...] = (4, 8, 8, 16, 16),
        ffn_exp: float = 2.66,
        vjepa_dim: int = 1024,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_layer_per_scale: Tuple[int, int, int, int, int] = (0, 0, 1, 2, 3),
        cross_attn_at_scales: Tuple[int, ...] = (1, 2, 3, 4),
        prompt_bank_at_scales: Tuple[int, ...] = (1, 2, 3, 4),
        prompt_num_conditions: int = 8,
        prompt_dim: int = 96,
        prompt_grid: Tuple[int, int, int] = (8, 8, 8),
        pixel_parallel_dims: Tuple[int, int, int, int] = (32, 64, 128, 192),
        use_cfim_bottleneck: bool = True,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert len(dims) == 5 and len(num_blocks) == 5

        self.dims = list(dims)
        self.vjepa_grid = vjepa_grid
        self.vjepa_layer_per_scale = list(vjepa_layer_per_scale)
        self.cross_attn_scales = set(cross_attn_at_scales)
        self.prompt_bank_scales = set(prompt_bank_at_scales)
        self.use_checkpoint = use_gradient_checkpoint

        # Pixel parallel encoder
        self.pixel_parallel = PixelParallelEncoder(
            in_channels=in_channels, dims=pixel_parallel_dims,
        )
        # Reduce conv to combine pixel-parallel features into decoder enc paths
        self.parallel_reduce = nn.ModuleList([
            nn.Conv2d(dims[s] + pixel_parallel_dims[s], dims[s], 1, bias=False)
            for s in range(4)
        ])

        # Stem
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=True)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_xattn = nn.ModuleList()
        self.enc_prompt = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        for s in range(5):
            self.enc_blocks.append(nn.ModuleList([
                RestormerBlockBF(dims[s], heads[s], ffn_exp) for _ in range(num_blocks[s])
            ]))
            self.enc_xattn.append(
                VJepaCrossAttnBF(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )
            self.enc_prompt.append(
                IlluminationPromptBank(
                    decoder_dim=dims[s],
                    num_conditions=prompt_num_conditions,
                    prompt_dim=prompt_dim,
                    prompt_grid_thw=prompt_grid,
                    num_heads=min(heads[s], 4),
                ) if s in self.prompt_bank_scales else None
            )
            if s < 4:
                self.enc_down.append(FlexDown(dims[s], dims[s + 1]))

        # Bottleneck
        self.temporal_mixer = TemporalMixer3D(dims[4])
        self.cfim_bottleneck = VolumeCFIM(dims[4], num_heads=8) if use_cfim_bottleneck else None

        # Decoder (mirror)
        self.dec_blocks = nn.ModuleList()
        self.dec_xattn = nn.ModuleList()
        self.dec_prompt = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        self.dec_reduce = nn.ModuleList()
        for s in range(3, -1, -1):
            self.dec_up.append(FlexUp(dims[s + 1], dims[s]))
            self.dec_reduce.append(nn.Conv2d(2 * dims[s], dims[s], 1, bias=False))
            self.dec_blocks.append(nn.ModuleList([
                RestormerBlockBF(dims[s], heads[s], ffn_exp) for _ in range(num_blocks[s])
            ]))
            self.dec_xattn.append(
                VJepaCrossAttnBF(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )
            self.dec_prompt.append(
                IlluminationPromptBank(
                    decoder_dim=dims[s],
                    num_conditions=prompt_num_conditions,
                    prompt_dim=prompt_dim,
                    prompt_grid_thw=prompt_grid,
                    num_heads=min(heads[s], 4),
                ) if s in self.prompt_bank_scales else None
            )

        # Output heads
        self.residual_head = nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

        self.retinex_head_R = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.retinex_head_L = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(dims[0], 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _bt(x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def _back(x_bt: torch.Tensor, T: int) -> torch.Tensor:
        BT, C, H, W = x_bt.shape
        B = BT // T
        return x_bt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    def _kv_for_scale(self, vjepa_layers, scale_idx, target_T):
        if scale_idx not in self.cross_attn_scales:
            return None
        layer_idx = self.vjepa_layer_per_scale[scale_idx]
        return _vjepa_to_kv_at_scale(vjepa_layers[layer_idx], self.vjepa_grid, target_T)

    def forward(
        self,
        video_lq: torch.Tensor,
        vjepa_layers: List[torch.Tensor],
        mixture_weights: torch.Tensor,
    ) -> dict:
        B, _, T, H, W = video_lq.shape

        # Pixel parallel encoder
        lq_bt = self._bt(video_lq)
        pp_feats = self.pixel_parallel(lq_bt)  # 4 scales

        # Main UNet
        x_bt = self.stem(lq_bt)

        skips = []
        for s in range(5):
            for blk in self.enc_blocks[s]:
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(blk, x_bt, use_reentrant=False)
                else:
                    x_bt = blk(x_bt)
            # Combine pixel parallel feature into encoder path at scales 0..3
            if s < 4:
                pp = pp_feats[s]
                if pp.shape[-2:] != x_bt.shape[-2:]:
                    pp = F.interpolate(pp, size=x_bt.shape[-2:], mode="bilinear", align_corners=False)
                x_bt = self.parallel_reduce[s](torch.cat([x_bt, pp], dim=1))
            # V-JEPA cross-attn
            if self.enc_xattn[s] is not None:
                kv = self._kv_for_scale(vjepa_layers, s, target_T=T)
                x_bt = self.enc_xattn[s](x_bt, kv)
            # Illumination prompt bank
            if self.enc_prompt[s] is not None:
                x_bt = self.enc_prompt[s](x_bt, mixture_weights, target_T=T)
            if s < 4:
                skips.append(x_bt)
                x_bt = self.enc_down[s](x_bt)
            else:
                x5 = self._back(x_bt, T)
                x5 = self.temporal_mixer(x5)
                if self.cfim_bottleneck is not None:
                    x5 = self.cfim_bottleneck(x5)
                x_bt = self._bt(x5)

        for i, scale_idx in enumerate([3, 2, 1, 0]):
            x_bt = self.dec_up[i](x_bt)
            x_bt = torch.cat([x_bt, skips[scale_idx]], dim=1)
            x_bt = self.dec_reduce[i](x_bt)
            for blk in self.dec_blocks[i]:
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(blk, x_bt, use_reentrant=False)
                else:
                    x_bt = blk(x_bt)
            if self.dec_xattn[i] is not None:
                kv = self._kv_for_scale(vjepa_layers, scale_idx, target_T=T)
                x_bt = self.dec_xattn[i](x_bt, kv)
            if self.dec_prompt[i] is not None:
                x_bt = self.dec_prompt[i](x_bt, mixture_weights, target_T=T)

        residual_bt = self.residual_head(x_bt)
        R_bt = self.retinex_head_R(x_bt)
        L_bt = self.retinex_head_L(x_bt)

        residual = self._back(residual_bt, T)
        R_hat = self._back(R_bt, T)
        L_hat = self._back(L_bt, T)

        out_main = video_lq + residual
        if not self.training:
            out_main = out_main.clamp(0.0, 1.0)
            R_hat = R_hat.clamp(0.0, 1.0)
            L_hat = L_hat.clamp(0.0, 1.0)
        return {
            "pred_pixels": out_main,
            "residual": residual,
            "R_hat": R_hat,
            "L_hat": L_hat,
        }
