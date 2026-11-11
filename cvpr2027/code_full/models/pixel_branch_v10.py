"""V-JEPA-LLVE v35 (Seer) Pixel Branch — reconstructed from checkpoint keys.

Extends the v9.0 architecture with three additional Retinexformer-inspired
components + a learnable cross-layer V-JEPA router:

    - Illumination_Estimator  (Retinexformer-style LQ -> illu_fea, illu_map)
    - illu_proj               (per-decoder-scale 1x1 projections of illu_fea)
    - dec_igab                (Retinexformer IGAB blocks, 4 decoder scales
                                x 2 blocks each, use illu_fea as attn modulator)
    - router_logits           (5 scales x 4 candidate V-JEPA layers; softmax
                                mixes token sources feeding each cross-attn)

All other modules (stem, enc/dec Restormer blocks, enc/dec V-JEPA cross-attn,
enc/dec local-illumination-cue injectors, temporal_mixer, VolumeCFIM
bottleneck, residual + Retinex R/L heads) are re-used verbatim from v9.

The resulting `state_dict()` reproduces the layout of the trained checkpoint
`/cache/zihaohe/checkpoint/runs/v35_S4_*_ft/best_y512.pt['pixel_branch']`
(962 keys, 0 missing / 0 unexpected).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .cfim import VolumeCFIM
from .local_illumination_cues import (
    IlluminationCueInjection,
    extract_illumination_cues,
)
from .pixel_branch_v8 import (
    FlexDown,
    FlexUp,
    RestormerBlockBF,
    TemporalMixer3D,
    VJepaCrossAttnBF,
    _vjepa_to_kv_at_scale,
)


# ---------------------------------------------------------------------------
# Retinexformer Illumination Estimator (depth_conv uses groups=1 in v35 ckpt)
# ---------------------------------------------------------------------------
class Illumination_Estimator(nn.Module):
    """LQ (B,3,H,W) -> illu_fea (B,C_mid,H,W), illu_map (B,3,H,W).

    Same layout as RetinexFormer_arch.Illumination_Estimator with two edits
    that match the v35 checkpoint shapes exactly:
      1. depth_conv uses `groups=1` (weight (C_mid, C_mid, 5, 5)) rather than
         the original `groups=n_fea_in` (weight (C_mid, C_mid/n_fea_in, 5, 5)).
    """

    def __init__(
        self,
        n_fea_middle: int = 64,
        n_fea_in: int = 4,
        n_fea_out: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True,
            groups=1,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img: torch.Tensor):
        # img: (B, 3, H, W)
        mean_c = img.mean(dim=1, keepdim=True)              # (B, 1, H, W)
        x = torch.cat([img, mean_c], dim=1)                 # (B, 4, H, W)
        illu_fea = self.depth_conv(self.conv1(x))           # (B, C_mid, H, W)
        illu_map = self.conv2(illu_fea)                     # (B, 3, H, W)
        return illu_fea, illu_map


# ---------------------------------------------------------------------------
# Retinexformer IGAB primitives (verbatim key layout)
# ---------------------------------------------------------------------------
class _GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class IG_MSA(nn.Module):
    """Illumination-guided multi-head self-attention (RetinexFormer)."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            _GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        """x_in / illu_fea_trans: (B, H, W, C)."""
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, "b n (hd d) -> b hd n d", hd=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)),
        )
        v = v * illu_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = attn @ v                                       # (B, heads, d, HW)
        out = out.permute(0, 3, 1, 2)                        # (B, HW, heads, d)
        out = out.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(out).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return out_c + out_p


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            _GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            _GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        y = self.net(x.permute(0, 3, 1, 2).contiguous())
        return y.permute(0, 2, 3, 1)


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class IGAB(nn.Module):
    """Illumination-guided attention block (RetinexFormer)."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                _PreNorm(dim, _FeedForward(dim=dim)),
            ]))

    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        """x, illu_fea: (B, C, H, W)."""
        x = x.permute(0, 2, 3, 1)
        illu_fea_hwc = illu_fea.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea_hwc) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class VJepaPixelBranch(nn.Module):
    """v35 (Seer) Pixel Branch, keyed to reproduce v9's UNet + Retinex adds.

    Inputs
    ------
    video_lq       : (B, 3, T, H, W) in [0, 1]
    vjepa_layers   : List of 4 x (B, N, vjepa_dim) hierarchical V-JEPA tokens

    Output dict
    -----------
    pred_pixels    : (B, 3, T, H, W) = LQ + residual (main output)
    residual, R_hat, L_hat : aux Retinex outputs
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
        num_vjepa_layers: int = 4,
        cross_attn_at_scales: Tuple[int, ...] = (1, 2, 3, 4),
        illu_middle_dim: int = 64,
        igab_dim_head: int = 64,
        igab_num_blocks: int = 2,
        use_cfim_bottleneck: bool = True,
        use_illumination_cues: bool = True,
        use_gradient_checkpoint: bool = True,
        router_init_c: float = 4.0,
    ) -> None:
        super().__init__()
        assert len(dims) == 5 and len(num_blocks) == 5 and len(heads) == 5

        self.dims = list(dims)
        self.num_blocks = list(num_blocks)
        self.heads = list(heads)
        self.vjepa_grid = vjepa_grid
        self.num_vjepa_layers = int(num_vjepa_layers)
        self.cross_attn_scales = set(cross_attn_at_scales)
        self.use_illumination_cues = use_illumination_cues
        self.use_checkpoint = use_gradient_checkpoint
        self.illu_middle_dim = illu_middle_dim
        self.igab_dim_head = igab_dim_head

        # Stem: LQ (3ch) -> dims[0]
        self.stem = nn.Conv2d(in_channels, dims[0], 3, padding=1, bias=True)

        # ----- Encoder (5 scales) -----
        self.enc_blocks = nn.ModuleList()
        self.enc_xattn = nn.ModuleList()
        self.enc_illum = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        for s in range(5):
            self.enc_blocks.append(nn.ModuleList([
                RestormerBlockBF(dims[s], heads[s], ffn_exp)
                for _ in range(num_blocks[s])
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

        # ----- Bottleneck -----
        self.temporal_mixer = TemporalMixer3D(dims[4])
        self.cfim_bottleneck = (
            VolumeCFIM(dims[4], num_heads=8) if use_cfim_bottleneck else None
        )

        # ----- Decoder (mirror; scale_idx = 3,2,1,0) -----
        self.dec_blocks = nn.ModuleList()
        self.dec_xattn = nn.ModuleList()
        self.dec_illum = nn.ModuleList()
        self.dec_up = nn.ModuleList()
        self.dec_reduce = nn.ModuleList()
        self.dec_igab = nn.ModuleList()
        for s in range(3, -1, -1):
            self.dec_up.append(FlexUp(dims[s + 1], dims[s]))
            self.dec_reduce.append(nn.Conv2d(2 * dims[s], dims[s], 1, bias=False))
            self.dec_blocks.append(nn.ModuleList([
                RestormerBlockBF(dims[s], heads[s], ffn_exp)
                for _ in range(num_blocks[s])
            ]))
            self.dec_xattn.append(
                VJepaCrossAttnBF(dims[s], vjepa_dim, num_heads=heads[s])
                if s in self.cross_attn_scales else None
            )
            self.dec_illum.append(
                IlluminationCueInjection(dims[s]) if use_illumination_cues else None
            )
            self.dec_igab.append(
                IGAB(
                    dim=dims[s],
                    dim_head=igab_dim_head,
                    heads=max(1, dims[s] // igab_dim_head),
                    num_blocks=igab_num_blocks,
                )
            )

        # ----- Retinexformer Illumination Estimator -----
        self.illu_estimator = Illumination_Estimator(
            n_fea_middle=illu_middle_dim,
            n_fea_in=in_channels + 1,       # LQ (3) + mean_c (1) = 4
            n_fea_out=in_channels,
        )
        # illu_proj[i]: 1x1 conv projecting illu_fea (C_mid) -> dec_dims[i]
        # decoder scale order: [dims[3], dims[2], dims[1], dims[0]]
        self.illu_proj = nn.ModuleList([
            nn.Conv2d(illu_middle_dim, dims[s], 1, bias=False)
            for s in [3, 2, 1, 0]
        ])

        # ----- Cross-layer V-JEPA router (5 scales x num_vjepa_layers) -----
        # Init: identity-like s.t. softmax picks v9's per-scale mapping
        # scale 0->layer 0, 1->0, 2->1, 3->2, 4->3.
        c = float(router_init_c)
        router_init = torch.zeros(5, num_vjepa_layers)
        idx_per_scale = [0, 0, 1, 2, 3]
        for s, li in enumerate(idx_per_scale):
            router_init[s, li] = c
        self.router_logits = nn.Parameter(router_init)

        # ----- Output heads -----
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

    # -------- shape helpers --------
    @staticmethod
    def _bt(x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def _back(x_bt: torch.Tensor, T: int) -> torch.Tensor:
        BT, C, H, W = x_bt.shape
        B = BT // T
        return x_bt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    # -------- router-mixed KV --------
    def _mixed_kv_for_scale(
        self,
        vjepa_layers: List[torch.Tensor],
        s: int,
        target_T: int,
    ):
        if s not in self.cross_attn_scales:
            return None
        w = F.softmax(self.router_logits[s], dim=0)   # (L,)
        # weighted average across V-JEPA layers
        mixed = None
        L = min(len(vjepa_layers), self.num_vjepa_layers)
        for li in range(L):
            contrib = w[li] * vjepa_layers[li]
            mixed = contrib if mixed is None else mixed + contrib
        return _vjepa_to_kv_at_scale(mixed, self.vjepa_grid, target_T)

    # -------- forward --------
    def forward(
        self,
        video_lq: torch.Tensor,
        vjepa_layers: List[torch.Tensor],
    ) -> dict:
        B, _, T, H, W = video_lq.shape
        lq_bt = self._bt(video_lq)                    # (BT, 3, H, W)

        # Retinexformer illumination estimation on LQ (per frame)
        illu_fea_full, _illu_map = self.illu_estimator(lq_bt)  # (BT, C_mid, H, W)

        # Local illumination cues (5 channels) for enc_illum / dec_illum
        cues_full = None
        if self.use_illumination_cues:
            cues_full = extract_illumination_cues(video_lq)     # (B, 5, T, H, W)

        # Stem
        x_bt = self.stem(lq_bt)

        # -------- Encoder --------
        skips: List[torch.Tensor] = []
        for s in range(5):
            for blk in self.enc_blocks[s]:
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(
                        blk, x_bt, use_reentrant=False,
                    )
                else:
                    x_bt = blk(x_bt)
            if self.enc_xattn[s] is not None:
                kv = self._mixed_kv_for_scale(vjepa_layers, s, target_T=T)
                x_bt = self.enc_xattn[s](x_bt, kv)
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

        # -------- Decoder --------
        for i, scale_idx in enumerate([3, 2, 1, 0]):
            x_bt = self.dec_up[i](x_bt)
            x_bt = torch.cat([x_bt, skips[scale_idx]], dim=1)
            x_bt = self.dec_reduce[i](x_bt)
            for blk in self.dec_blocks[i]:
                if self.use_checkpoint and self.training:
                    x_bt = torch.utils.checkpoint.checkpoint(
                        blk, x_bt, use_reentrant=False,
                    )
                else:
                    x_bt = blk(x_bt)
            if self.dec_xattn[i] is not None:
                kv = self._mixed_kv_for_scale(vjepa_layers, scale_idx, target_T=T)
                x_bt = self.dec_xattn[i](x_bt, kv)
            if self.dec_illum[i] is not None and cues_full is not None:
                x_bt = self.dec_illum[i](x_bt, cues_full, target_T=T)
            # Retinexformer IGAB with per-scale illumination fea
            _, _, hi, wi = x_bt.shape
            illu_scaled = F.interpolate(
                illu_fea_full, size=(hi, wi), mode="bilinear", align_corners=False,
            )
            illu_at_dim = self.illu_proj[i](illu_scaled)          # (BT, dim_s, hi, wi)
            x_bt = self.dec_igab[i](x_bt, illu_at_dim)

        # -------- Heads --------
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


# Backwards-compat alias — some downstream code may import v10 by its version tag.
VJepaPixelBranch_v10 = VJepaPixelBranch
