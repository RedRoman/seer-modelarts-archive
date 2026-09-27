"""V-JEPA-LLVE v7 — pure pixel-domain reconstruction with V-JEPA grounding.

Architectural break vs v6 series:
    NO LatentDiT. NO flow matching. NO ODE inference.
    Just V-JEPA encoder (frozen) → big PixelBranch → output.

    All 300M trainable parameters now serve pixel reconstruction directly,
    with V-JEPA providing multi-scale semantic guidance via cross-attention.

Output format:
    pred_pixels (B,3,T,H,W): main residual output = LQ + residual
    R_hat (B,3,T,H,W):       Retinex reflectance head (auxiliary supervision)
    L_hat (B,1,T,H,W):       Retinex illumination head (auxiliary supervision)

Training: pred + R̂ × L̂ both supervised against HQ; L̂ smoothness regularizer.
Inference: 1 forward pass, no ODE. ~5x faster than v6.E.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .pixel_branch_v7 import VJepaPixelBranch_v7
from .vjepa_backbone import build_vjepa_backbone


class VJepaLLVE_v7(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # frozen V-JEPA backbone
        self.backbone: nn.Module = build_vjepa_backbone(cfg["backbone"])
        self.vjepa_dim = int(self.backbone.hidden_dim)

        # V-JEPA token grid
        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        # Big pixel branch — sole trainable component
        pbcfg = cfg["pixel_branch"]
        self.pixel_branch = VJepaPixelBranch_v7(
            in_channels = 3,
            dims        = tuple(pbcfg.get("dims", (64, 128, 256, 384, 512))),
            num_blocks  = tuple(pbcfg.get("num_blocks", (4, 6, 6, 8, 10))),
            heads       = tuple(pbcfg.get("heads", (4, 8, 8, 8, 8))),
            ffn_exp     = float(pbcfg.get("ffn_exp", 2.66)),
            vjepa_dim   = self.vjepa_dim,
            vjepa_grid  = vjepa_grid,
            vjepa_layer_per_scale = tuple(
                pbcfg.get("vjepa_layer_per_scale", (0, 0, 1, 2, 3))
            ),
            cross_attn_at_scales = tuple(pbcfg.get("cross_attn_at_scales", (0, 1, 2, 3, 4))),
            use_cfim_bottleneck = bool(pbcfg.get("use_cfim_bottleneck", True)),
            use_gradient_checkpoint = bool(pbcfg.get("use_gradient_checkpoint", True)),
        )

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.pixel_branch.parameters():
            if p.requires_grad:
                yield p

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(video_01)["feats"])

    def forward(
        self,
        video_lq: torch.Tensor,
        video_hq: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """v7 has no flow / no t. video_hq, t are accepted for API compat but ignored."""
        e_lq_layers = self.encode_video_features(video_lq)
        target_dtype = video_lq.dtype
        e_lq_layers = [f.to(target_dtype) for f in e_lq_layers]

        out = self.pixel_branch(video_lq, e_lq_layers)
        out["e_lq_layers"] = e_lq_layers
        return out

    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
