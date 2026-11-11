"""V-JEPA-LLVE v7: Big Pixel Branch (no LatentDiT).

Architectural break vs v6.D/v6.E:
    * No latent flow / DiT. Pure supervised pixel reconstruction.
    * Decoder capacity scaled up: dims [64, 128, 256, 384, 512] (vs v6's [48,96,192,384,768]).
    * Block depth ramped: [4, 6, 6, 8, 10] (vs v6's [2, 4, 4, 6, 8]).
    * V-JEPA cross-attn at ALL 5 scales (vs v6's {2, 3, 4}).
    * VolumeCFIM at bottleneck for cross-frame interaction.
    * Retinex dual output head: R̂ (3-ch) + L̂ (1-ch) at full resolution.
       Final = LQ + residual; auxiliary supervision: HQ ≈ R̂ ⊙ L̂.
    * Identity-at-init: residual head zero-init ⇒ output = LQ at start.

Why this should break the v6 ceiling:
    v6's 60% trainable params went to LatentDiT processing 16x16 V-JEPA tokens.
    That's information-theoretically capped because V-JEPA was trained for semantic
    invariance, not pixel reconstruction. Even perfect latent learning gives ~25.5 dB.
    v7 puts all 300M on pixel reconstruction, with V-JEPA only as multi-scale guidance.
    PSNR ceiling now depends on PixelBranch capacity, which scales with dims/blocks.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Reuse Restormer primitives + cross-attn from existing code
from .pixel_decoder import (
    RestormerBlock, VJepaCrossAttn, Downsample, Upsample, TemporalMixer3D,
    _vjepa_to_kv_at_scale,
)
from .cfim import VolumeCFIM


class VJepaPixelBranch_v7(nn.Module):
    """High-capacity 5-scale UNet with V-JEPA cross-attn at every scale,
    bottleneck CFIM, and Retinex pixel-domain dual head.

    Args
    ----
    in_channels : 3 (RGB).
    dims : list of 5 ints — channel widths at scales [enc0..enc3, bottleneck].
           Default: [64, 128, 256, 384, 512].
    num_blocks : list of 5 ints — Restormer block counts at each scale.
           Default: [4, 6, 6, 8, 10].
    heads : list of 5 ints — attention heads.
    ffn_exp : MDTA/GDFN expansion.
    vjepa_dim, vjepa_grid : V-JEPA encoder spec.
    vjepa_layer_per_scale : V-JEPA layer index used for cross-attn at each scale.
    cross_attn_at_scales : decoder scales where cross-attn is active.
        Default {0,1,2,3,4} (all scales — major change vs v6).
    use_cfim_bottleneck : VolumeCFIM at bot scale (default True).
    use_gradient_checkpoint : bool.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Tuple[int, int, int, int, int] = (64, 128, 256, 384, 512),
        num_blocks: Tuple[int, ...] = (4, 6, 6, 8, 10),
        heads: Tuple[int, ...] = (4, 8, 8, 8, 8),
        ffn_exp: float = 2.66,
        vjepa_dim: int = 1024,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        vjepa_layer_per_scale: Tuple[int, int, int, int, int] = (0, 0, 1, 2, 3),
        cross_attn_at_scales: Tuple[int, ...] = (0, 1, 2, 3, 4),
        use_cfim_bottleneck: bool = True,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert len(dims) == 5 and len(num_blocks) == 5 and len(heads) == 5

        self.dims = list(dims)
        self.vjepa_grid = vjepa_grid
        self.vjepa_layer_per_scale = list(vjepa_layer_per_scale)
        self.cross_attn_scales = set(cross_attn_at_scales)
        self.use_checkpoint = use_gradient_checkpoint

        # Stem: 3 → dims[0]
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=True)

        # Encoder: 5 stages with custom-width transitions
        self.enc_blocks = nn.ModuleList()
        self.enc_xattn = nn.ModuleList()
        self.enc_down = nn.ModuleList()
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
                # Custom downsample: dims[s] -> dims[s+1] at half spatial
                self.enc_down.append(_FlexDownsample(dims[s], dims[s + 1]))

        # Bottleneck
        self.temporal_mixer = TemporalMixer3D(dims[4])
        self.cfim_bottleneck = VolumeCFIM(dims[4], num_heads=8) if use_cfim_bottleneck else None

        # Decoder: 4 up steps with skip from enc[3..0], custom widths
        self.dec_blocks = nn.ModuleList()
        self.dec_xattn = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        self.dec_reduce = nn.ModuleList()
        for s in range(3, -1, -1):  # s: 3, 2, 1, 0
            self.dec_up.append(_FlexUpsample(dims[s + 1], dims[s]))
            self.dec_reduce.append(nn.Conv2d(2 * dims[s], dims[s], 1, bias=False))
            stage = nn.ModuleList([
                RestormerBlock(dims[s], heads[s], ffn_exp)
                for _ in range(num_blocks[s])
            ])
            self.dec_blocks.append(stage)
            self.dec_xattn.append(
                VJepaCrossAttn(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )

        # --- Output heads ---
        # Main residual: 3-ch, zero-init (identity-at-init: out = LQ)
        self.residual_head = nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

        # Retinex pixel-domain auxiliary heads:
        #   R̂ (3-ch reflectance) — learns object color/material
        #   L̂ (1-ch illumination) — learns lighting envelope, single channel for smooth recovery
        # Both small init (not zero) so loss can flow through them from start.
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

    # --------------------------------------------------------------
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
        if scale_idx not in self.cross_attn_scales:
            return None
        layer_idx = self.vjepa_layer_per_scale[scale_idx]
        return _vjepa_to_kv_at_scale(
            vjepa_layers[layer_idx], self.vjepa_grid, target_T,
        )

    # --------------------------------------------------------------
    def forward(
        self,
        video_lq: torch.Tensor,             # (B, 3, T, H, W) in [0, 1]
        vjepa_layers: List[torch.Tensor],   # 4 layers, each (B, N, vjepa_dim)
    ):
        if len(vjepa_layers) != 4:
            raise ValueError(f"v7 expects 4 V-JEPA layers; got {len(vjepa_layers)}")
        B, _, T, H, W = video_lq.shape
        x_bt = self._bcthw_to_btchw(video_lq)
        x_bt = self.stem(x_bt)

        # ---- encoder ----
        skips = []
        for s in range(5):
            for blk in self.enc_blocks[s]:
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(blk, x_bt, use_reentrant=False)
                else:
                    x_bt = blk(x_bt)
            if self.enc_xattn[s] is not None:
                kv = self._kv_for_scale(vjepa_layers, s, target_T=T)
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(self.enc_xattn[s], x_bt, kv, use_reentrant=False)
                else:
                    x_bt = self.enc_xattn[s](x_bt, kv)
            if s < 4:
                skips.append(x_bt)
                x_bt = self.enc_down[s](x_bt)
            else:
                # bottleneck: temporal mixer + CFIM (if enabled)
                x5 = self._btchw_to_bcthw(x_bt, T)
                x5 = self.temporal_mixer(x5)
                if self.cfim_bottleneck is not None:
                    x5 = self.cfim_bottleneck(x5)
                x_bt = self._bcthw_to_btchw(x5)

        # ---- decoder ----
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
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(self.dec_xattn[i], x_bt, kv, use_reentrant=False)
                else:
                    x_bt = self.dec_xattn[i](x_bt, kv)

        # ---- output heads ----
        residual_bt = self.residual_head(x_bt)
        R_bt = self.retinex_head_R(x_bt)
        L_bt = self.retinex_head_L(x_bt)

        residual = self._btchw_to_bcthw(residual_bt, T)
        R_hat = self._btchw_to_bcthw(R_bt, T)
        L_hat = self._btchw_to_bcthw(L_bt, T)

        out_main = video_lq + residual
        if not self.training:
            out_main = out_main.clamp(0.0, 1.0)
            R_hat = R_hat.clamp(0.0, 1.0)
            L_hat = L_hat.clamp(0.0, 1.0)
        return {
            "pred_pixels": out_main,
            "residual":    residual,
            "R_hat":       R_hat,
            "L_hat":       L_hat,  # single channel; broadcast for compose
        }


class _FlexDownsample(nn.Module):
    """Downsample 2x with custom output channel count (vs PixelUnshuffle which is 4x)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # Stride-2 conv 3x3 — flexible channel adapter
        self.body = nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _FlexUpsample(nn.Module):
    """Upsample 2x with custom output channel count."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.body = nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
