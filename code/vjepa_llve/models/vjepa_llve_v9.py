"""V-JEPA-LLVE v9.0 end-to-end model.

Locked spec (G4 review criteria):
    - Frozen V-JEPA 2.1 ViT-L encoder (304M)
    - V-JEPA cross-attn at 4 decoder scales (kept)
    - Big Pixel Branch v9 (~250M trainable, BiasFree LN, Restormer)
    - Local illumination cue injection (NEW, replaces global Prompt Bank)
    - Bottleneck VolumeCFIM (kept)
    - Retinex dual aux head (R̂, L̂)
    - DROPPED: Predictor / LoRA / counterfactual loss (per F3.3)
    - DROPPED: Pixel parallel CNN path (per F1.5)
    - DROPPED: Illumination Prompt Bank (replaced by local cues per F2.4)
    - DROPPED: Multi-iter rollout / monotonicity loss
    - NO TEACHER FORCING: decoder always conditioned on V-JEPA encoder LQ output
      (since Predictor is dropped, there's no z_hq_pred to use anyway).

Conceptual signature:
    pred_pixels = LQ + residual
    R̂, L̂ = aux heads (supervised by R̂ ⊙ L̂ ≈ HQ + temporal smoothness)
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .pixel_branch_v9 import VJepaPixelBranch_v9
from .vjepa_backbone import build_vjepa_backbone


class VJepaLLVE_v9(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        # AH4 ablation: skip V-JEPA backbone entirely (build + forward).
        # Pixel branch is then run with empty vjepa_layers; cross_attn_at_scales must be [].
        self.disable_vjepa = bool(cfg.get("disable_vjepa", False))

        # V-JEPA token grid (still derived from cfg even when disabled, for shape compat)
        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        if self.disable_vjepa:
            self.backbone = None
            self.vjepa_dim = int(bcfg.get("hidden_dim", 1024))  # nominal, never used
        else:
            self.backbone = build_vjepa_backbone(cfg["backbone"])
            self.vjepa_dim = int(self.backbone.hidden_dim)

        # Big Pixel Branch (only trainable component in v9.0)
        pbcfg = cfg["pixel_branch"]
        self.pixel_branch = VJepaPixelBranch_v9(
            in_channels=3,
            dims=tuple(pbcfg.get("dims", (64, 128, 256, 512, 1024))),
            num_blocks=tuple(pbcfg.get("num_blocks", (4, 8, 8, 12, 16))),
            heads=tuple(pbcfg.get("heads", (4, 8, 8, 16, 16))),
            ffn_exp=float(pbcfg.get("ffn_exp", 2.66)),
            vjepa_dim=self.vjepa_dim,
            vjepa_grid=vjepa_grid,
            vjepa_layer_per_scale=tuple(pbcfg.get("vjepa_layer_per_scale", (0, 0, 1, 2, 3))),
            cross_attn_at_scales=tuple(pbcfg.get("cross_attn_at_scales", (1, 2, 3, 4))),
            use_cfim_bottleneck=bool(pbcfg.get("use_cfim_bottleneck", True)),
            use_illumination_cues=bool(pbcfg.get("use_illumination_cues", True)),
            use_gradient_checkpoint=bool(pbcfg.get("use_gradient_checkpoint", True)),
        )

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.pixel_branch.parameters():
            if p.requires_grad:
                yield p

    def freeze_backbone(self) -> None:
        if self.backbone is None:
            return
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        if self.backbone is None:
            return []
        return list(self.backbone(video_01)["feats"])

    def forward(
        self,
        video_lq: torch.Tensor,
        video_hq: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,  # accepted for API compat, not used in v9.0
    ) -> Dict[str, torch.Tensor]:
        """v9.0 forward.

        Decoder always conditions on V-JEPA(LQ) hierarchical features —
        NO teacher forcing with E(HQ).
        When disable_vjepa=True, e_lq_layers is empty and pixel_branch must have
        cross_attn_at_scales=[].
        """
        target_dtype = video_lq.dtype
        e_lq_layers = self.encode_video_features(video_lq)
        e_lq_layers = [f.to(target_dtype) for f in e_lq_layers]

        out = self.pixel_branch(video_lq, e_lq_layers)
        out["e_lq_layers"] = e_lq_layers
        return out

    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
