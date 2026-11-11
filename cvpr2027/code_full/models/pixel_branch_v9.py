"""V-JEPA-LLVE v9.0 Pixel Branch.

Per locked v9.0 spec (G4 review criteria):
    - 4-scale UNet with BiasFree LN + PixelShuffle + Restormer blocks
    - V-JEPA cross-attn at scales {1, 2, 3, 4} (kept from v8)
    - Local illumination cue injection at scales {0, 1, 2, 3, 4} (NEW per F1.4)
    - Bottleneck VolumeCFIM (kept from v8)
    - Output: residual head + R̂ + L̂ aux heads
    - DROPPED: Pixel Parallel CNN path (per F1.5)
    - DROPPED: Illumination Prompt Bank (replaced by local cues per F2.4)
    - DROPPED: V-JEPA token affinity (deferred to v9.2)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .biasfree_ln import BiasFreeLayerNorm2d
from .cfim import VolumeCFIM
from .local_illumination_cues import IlluminationCueInjection, extract_illumination_cues
from .pixel_branch_v8 import (
    RestormerBlockBF, VJepaCrossAttnBF, FlexDown, FlexUp, TemporalMixer3D,
    _vjepa_to_kv_at_scale,
)


class VJepaPixelBranch_v9(nn.Module):
    """v9.0 big pixel UNet — clean Retinex baseline (no Predictor, no affinity).

    Inputs:
      video_lq (B, 3, T, H, W) in [0, 1]
      vjepa_layers: List[4] of (B, N, vjepa_dim) — V-JEPA hierarchical features

    Output dict:
      pred_pixels (B, 3, T, H, W)  — main residual output: LQ + residual
      R_hat (B, 3, T, H, W), L_hat (B, 1, T, H, W) — Retinex aux supervision
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
        use_cfim_bottleneck: bool = True,
        use_illumination_cues: bool = True,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert len(dims) == 5 and len(num_blocks) == 5

        self.dims = list(dims)
        self.vjepa_grid = vjepa_grid
        self.vjepa_layer_per_scale = list(vjepa_layer_per_scale)
        self.cross_attn_scales = set(cross_attn_at_scales)
        self.use_illumination_cues = use_illumination_cues
        self.use_checkpoint = use_gradient_checkpoint

        # Stem: LQ pixels (3) → dims[0]
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=True)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.enc_xattn = nn.ModuleList()
        self.enc_illum = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        for s in range(5):
            self.enc_blocks.append(nn.ModuleList([
                RestormerBlockBF(dims[s], heads[s], ffn_exp) for _ in range(num_blocks[s])
            ]))
            self.enc_xattn.append(
                VJepaCrossAttnBF(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )
            self.enc_illum.append(
                IlluminationCueInjection(dims[s]) if use_illumination_cues else None
            )
            if s < 4:
                self.enc_down.append(FlexDown(dims[s], dims[s + 1]))

        # Bottleneck
        self.temporal_mixer = TemporalMixer3D(dims[4])
        self.cfim_bottleneck = VolumeCFIM(dims[4], num_heads=8) if use_cfim_bottleneck else None

        # Decoder (mirror)
        self.dec_blocks = nn.ModuleList()
        self.dec_xattn = nn.ModuleList()
        self.dec_illum = nn.ModuleList()
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
            self.dec_illum.append(
                IlluminationCueInjection(dims[s]) if use_illumination_cues else None
            )

        # Output heads
        # Main: residual head (zero-init for identity-at-init, pred = LQ at start)
        self.residual_head = nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

        # Retinex auxiliary heads (R̂ ∈ [0,1]^3, L̂ ∈ [0,1]^1)
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
    ) -> dict:
        B, _, T, H, W = video_lq.shape
        lq_bt = self._bt(video_lq)

        # Pre-compute illumination cues at full resolution (5 channels)
        cues_full = None
        if self.use_illumination_cues:
            cues_full = extract_illumination_cues(video_lq)  # (B, 5, T, H, W)

        x_bt = self.stem(lq_bt)

        skips = []
        for s in range(5):
            for blk in self.enc_blocks[s]:
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(blk, x_bt, use_reentrant=False)
                else:
                    x_bt = blk(x_bt)
            # V-JEPA cross-attn
            if self.enc_xattn[s] is not None:
                kv = self._kv_for_scale(vjepa_layers, s, target_T=T)
                x_bt = self.enc_xattn[s](x_bt, kv)
            # Local illumination cue injection
            if self.enc_illum[s] is not None and cues_full is not None:
                x_bt = self.enc_illum[s](x_bt, cues_full, target_T=T)
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
            if self.dec_illum[i] is not None and cues_full is not None:
                x_bt = self.dec_illum[i](x_bt, cues_full, target_T=T)

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
