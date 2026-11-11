"""V-JEPA-LLVE v10 — extends v9 with VJepaPosEmb (M1).

Architecture:
    LQ → V-JEPA(frozen, 304M) → e_lq_layers
                                        ↓
                                 + VJepaPosEmb (M1: gamma_pos * (P_t+P_y+P_x))
                                        ↓
                                 pixel_branch_v9 (361M trainable, UNCHANGED)
                                        ↓
                                       pred

Identity-at-init guarantee (CH3):
    With gamma_pos=0 and P_*=0, the pos-emb addition is exactly zero, so
    v10 forward is byte-identical to v9 forward.

Spec: V10_DESIGN_SPEC.md §3, §4.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .vjepa_pos_emb import VJepaPosEmb
from ..pixel_branch_v9 import VJepaPixelBranch_v9
from ..vjepa_backbone import build_vjepa_backbone


class VJepaLLVE_v10(nn.Module):
    """Same as VJepaLLVE_v9 but with VJepaPosEmb applied to V-JEPA tokens
    before pixel_branch. See spec §3 for design rationale."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.disable_vjepa = bool(cfg.get("disable_vjepa", False))

        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        if self.disable_vjepa:
            self.backbone = None
            self.vjepa_dim = int(bcfg.get("hidden_dim", 1024))
        else:
            self.backbone = build_vjepa_backbone(cfg["backbone"])
            self.vjepa_dim = int(self.backbone.hidden_dim)

        # NEW v10: positional embedding (identity-at-init).
        # Apply only when V-JEPA is active; if disabled, pos-emb is irrelevant.
        if not self.disable_vjepa:
            self.vjepa_pos_emb = VJepaPosEmb(token_grid=vjepa_grid, dim=self.vjepa_dim)
        else:
            self.vjepa_pos_emb = None

        # Pixel branch — UNCHANGED from v9.
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

    # ---- training utilities ------------------------------------------
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.pixel_branch.parameters():
            if p.requires_grad:
                yield p
        if self.vjepa_pos_emb is not None:
            for p in self.vjepa_pos_emb.parameters():
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

    # ---- v9.0b_mix weight loading helper -----------------------------
    def load_v9_weights(self, v9_state: Dict[str, torch.Tensor], strict: bool = False) -> None:
        """Load v9.0b_mix checkpoint weights into v10. New v10 keys
        (vjepa_pos_emb.*) start at zero per identity-at-init."""
        # v9 had no vjepa_pos_emb; just load whatever matches.
        own = self.state_dict()
        # Map: copy any v9 key that exists in own.
        compat = {k: v for k, v in v9_state.items() if k in own and own[k].shape == v.shape}
        own.update(compat)
        self.load_state_dict(own, strict=False)
        n_own = len(own); n_loaded = len(compat); n_skipped = len(v9_state) - n_loaded
        print(f"[VJepaLLVE_v10] loaded {n_loaded}/{n_own} keys from v9 ckpt; "
              f"{n_skipped} v9 keys not used; pos-emb keys are at identity-init")

    # ---- forward -----------------------------------------------------
    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        if self.backbone is None:
            return []
        return list(self.backbone(video_01)["feats"])

    def forward(
        self,
        video_lq: torch.Tensor,
        video_hq: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        target_dtype = video_lq.dtype
        e_lq_layers = self.encode_video_features(video_lq)
        e_lq_layers = [f.to(target_dtype) for f in e_lq_layers]

        # NEW v10: apply positional embedding (identity-at-init).
        if self.vjepa_pos_emb is not None and len(e_lq_layers) > 0:
            e_lq_layers = self.vjepa_pos_emb(e_lq_layers)

        out = self.pixel_branch(video_lq, e_lq_layers)
        out["e_lq_layers"] = e_lq_layers
        return out

    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
