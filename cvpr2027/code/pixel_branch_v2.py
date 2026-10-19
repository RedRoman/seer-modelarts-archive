"""Seer-v2 pixel branch: the minimal architecture our ablations support.

Relative to pixel_branch_v10.py (the AAAI/ICLR architecture) this removes the
two components that measurably did not help, and keeps everything that did:

  removed  IGAB (Retinexformer illumination-guided attention in the decoder)
           -- dropping it gained +0.67 dB on L1 and +0.39 on L2 (750-it ablation)
  removed  the learnable cross-layer router (M3): its softmax mass never left
           the initial 0.948, and a softer init scored better, so the mapping is
           now a fixed layer-per-scale assignment with no parameters
  kept     V-JEPA cross-attention with zero-init output projection
  kept     LIC (local illumination cues) -- removing it cost L1/L2
  kept     bottleneck temporal mixer + CFIM -- removing it cost L2 and tLP
  kept     Retinex auxiliary heads (training-only, cost-free at inference)

The illumination estimator is also dropped, since it only fed IGAB.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .cfim import VolumeCFIM
from .local_illumination_cues import IlluminationCueInjection, extract_illumination_cues
from .pixel_branch_v8 import (FlexDown, FlexUp, RestormerBlockBF, TemporalMixer3D,
                              VJepaCrossAttnBF, _vjepa_to_kv_at_scale)


class SeerV2PixelBranch(nn.Module):
    """Five-scale Restormer U-Net conditioned on frozen V-JEPA tokens.

    Inputs
    ------
    video_lq     : (B, 3, T, H, W) in [0, 1]
    vjepa_layers : list of 4 tensors (B, N, vjepa_dim), shallow -> deep

    Output dict: pred_pixels (B,3,T,H,W) = LQ + residual, plus residual/R_hat/L_hat.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        num_blocks: Tuple[int, ...] = (4, 8, 8, 12, 16),
        heads: Tuple[int, ...] = (4, 8, 8, 16, 16),
        ffn_exp: float = 2.66,
        vjepa_dim: int = 1024,
        vjepa_grid: Tuple[int, int, int] = (8, 16, 16),
        num_vjepa_layers: int = 4,
        cross_attn_at_scales: Tuple[int, ...] = (1, 2, 3, 4),
        layer_per_scale: Tuple[int, ...] = (0, 0, 1, 2, 3),
        use_cfim_bottleneck: bool = True,
        use_illumination_cues: bool = True,
        use_gradient_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        assert len(dims) == 5 and len(num_blocks) == 5 and len(heads) == 5
        self.dims, self.num_blocks, self.heads = list(dims), list(num_blocks), list(heads)
        self.vjepa_grid = vjepa_grid
        self.num_vjepa_layers = int(num_vjepa_layers)
        self.cross_attn_scales = set(cross_attn_at_scales)
        self.layer_per_scale = list(layer_per_scale)
        self.use_illumination_cues = use_illumination_cues
        self.use_checkpoint = use_gradient_checkpoint

        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=True)

        self.enc_blocks, self.enc_xattn, self.enc_illum, self.enc_down = (
            nn.ModuleList() for _ in range(4))
        for s in range(5):
            self.enc_blocks.append(nn.ModuleList(
                [RestormerBlockBF(dims[s], heads[s], ffn_exp) for _ in range(num_blocks[s])]))
            self.enc_xattn.append(VJepaCrossAttnBF(dims[s], vjepa_dim, num_heads=heads[s])
                                  if s in self.cross_attn_scales else None)
            self.enc_illum.append(IlluminationCueInjection(dims[s])
                                  if use_illumination_cues else None)
            if s < 4:
                self.enc_down.append(FlexDown(dims[s], dims[s + 1]))

        self.temporal_mixer = TemporalMixer3D(dims[4])
        self.cfim_bottleneck = VolumeCFIM(dims[4], num_heads=8) if use_cfim_bottleneck else None

        self.dec_blocks, self.dec_xattn, self.dec_illum = (nn.ModuleList() for _ in range(3))
        self.dec_up, self.dec_reduce = nn.ModuleList(), nn.ModuleList()
        for s in range(3, -1, -1):
            self.dec_up.append(FlexUp(dims[s + 1], dims[s]))
            self.dec_reduce.append(nn.Conv2d(2 * dims[s], dims[s], 1, bias=False))
            self.dec_blocks.append(nn.ModuleList(
                [RestormerBlockBF(dims[s], heads[s], ffn_exp) for _ in range(num_blocks[s])]))
            self.dec_xattn.append(VJepaCrossAttnBF(dims[s], vjepa_dim, num_heads=heads[s])
                                  if s in self.cross_attn_scales else None)
            self.dec_illum.append(IlluminationCueInjection(dims[s])
                                  if use_illumination_cues else None)

        self.residual_head = nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=False)
        nn.init.zeros_(self.residual_head.weight)
        self.retinex_head_R = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(dims[0], in_channels, 3, padding=1, bias=True), nn.Sigmoid())
        self.retinex_head_L = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1, bias=True), nn.SiLU(inplace=True),
            nn.Conv2d(dims[0], 1, 3, padding=1, bias=True), nn.Sigmoid())

    @staticmethod
    def _bt(x):
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def _back(x, T):
        BT, C, H, W = x.shape
        return x.reshape(BT // T, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    def _kv(self, vjepa_layers: List[torch.Tensor], s: int, target_T: int):
        """Fixed layer-to-scale mapping (no learnable router)."""
        if s not in self.cross_attn_scales:
            return None
        li = min(self.layer_per_scale[s], len(vjepa_layers) - 1)
        return _vjepa_to_kv_at_scale(vjepa_layers[li], self.vjepa_grid, target_T)

    def forward(self, video_lq: torch.Tensor, vjepa_layers: List[torch.Tensor]) -> dict:
        B, _, T, H, W = video_lq.shape
        x_bt = self.stem(self._bt(video_lq))
        cues = extract_illumination_cues(video_lq) if self.use_illumination_cues else None

        skips: List[torch.Tensor] = []
        for s in range(5):
            for blk in self.enc_blocks[s]:
                x_bt = (torch.utils.checkpoint.checkpoint(blk, x_bt, use_reentrant=False)
                        if self.use_checkpoint and self.training else blk(x_bt))
            if self.enc_xattn[s] is not None:
                x_bt = self.enc_xattn[s](x_bt, self._kv(vjepa_layers, s, T))
            if self.enc_illum[s] is not None and cues is not None:
                x_bt = self.enc_illum[s](x_bt, cues, target_T=T)
            if s < 4:
                skips.append(x_bt)
                x_bt = self.enc_down[s](x_bt)
            else:
                x5 = self.temporal_mixer(self._back(x_bt, T))
                if self.cfim_bottleneck is not None:
                    x5 = self.cfim_bottleneck(x5)
                x_bt = self._bt(x5)

        for i, s in enumerate([3, 2, 1, 0]):
            x_bt = self.dec_up[i](x_bt)
            x_bt = self.dec_reduce[i](torch.cat([x_bt, skips[s]], dim=1))
            for blk in self.dec_blocks[i]:
                x_bt = (torch.utils.checkpoint.checkpoint(blk, x_bt, use_reentrant=False)
                        if self.use_checkpoint and self.training else blk(x_bt))
            if self.dec_xattn[i] is not None:
                x_bt = self.dec_xattn[i](x_bt, self._kv(vjepa_layers, s, T))
            if self.dec_illum[i] is not None and cues is not None:
                x_bt = self.dec_illum[i](x_bt, cues, target_T=T)

        residual = self._back(self.residual_head(x_bt), T)
        R_hat = self._back(self.retinex_head_R(x_bt), T)
        L_hat = self._back(self.retinex_head_L(x_bt), T)
        out = video_lq + residual
        if not self.training:
            out, R_hat, L_hat = (t.clamp(0.0, 1.0) for t in (out, R_hat, L_hat))
        return {"pred_pixels": out, "residual": residual, "R_hat": R_hat, "L_hat": L_hat}
