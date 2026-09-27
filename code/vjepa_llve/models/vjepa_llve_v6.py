"""V-JEPA-LLVE v6 end-to-end model.

Pipeline:
    LQ video
       │
       ├──► V-JEPA 2.1 Encoder (frozen) ──► E(LQ) at 4 layers
       │                                       │
       │                ┌──────────────────────┘
       │                │  At training: also encode HQ to get E(HQ) for flow target
       │                │
       │                ▼
       │   【Stage A】 V-JEPA Latent DiT (rectified flow in V-JEPA space)
       │   ─────────────────────────────────────────────────────
       │   Trains: predict v(z_t, t) where z_t = (1-t)·E(LQ)[-1] + t·E(HQ)[-1]
       │   Loss: MSE(v_pred, E(HQ)[-1] - E(LQ)[-1])
       │
       │                ▼ refined V-JEPA tokens (at deepest layer)
       │
       ▼
    【Stage B】 V-JEPA-Grounded Pixel Decoder
    ──────────────────────────────────────────────
    Input:  raw LQ pixels (B,3,T,256,256)
    Cond:   4 V-JEPA layers (E(LQ) at l5,l11,l17 + refined at l23 [or E(HQ) at training time
            for stable teacher-forcing])
    Output: HQ pixels = LQ + residual
    Loss:   Charbonnier + LPIPS + temporal_DoD

The teacher-forcing trick at training time: when computing the pixel loss
during training, the pixel decoder's deepest cross-attn condition is the
TRUE E(HQ)[-1] (computed from HQ through frozen V-JEPA), not the DiT's
prediction. This decouples decoder learning from DiT convergence speed.
At inference, the DiT-solved refined tokens replace E(HQ)[-1] (since we
have no HQ at test time).

Identity-at-init:
  - DiT.final_proj zero-init → v_pred = 0 → ODE returns z_LQ
  - VJepaCrossAttn.proj zero-init → cross-attn contributes 0 at init
  - PixelDecoder.output_conv zero-init → residual = 0
  Together: out = LQ at step 0.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .latent_dit import VJepaLatentDiT
from .pixel_decoder import VJepaPixelDecoder
from .vjepa_backbone import build_vjepa_backbone


class VJepaLLVE_v6(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # --- frozen backbone ---
        self.backbone: nn.Module = build_vjepa_backbone(cfg["backbone"])
        self.vjepa_dim = int(self.backbone.hidden_dim)
        self.target_layer = int(cfg.get("predictor", {}).get("target_layer_idx", 3))

        # --- V-JEPA token grid ---
        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        # --- Stage A: latent DiT ---
        ldcfg = cfg["latent_dit"]
        self.latent_dit = VJepaLatentDiT(
            vjepa_dim = self.vjepa_dim,
            grid      = vjepa_grid,
            dim       = int(ldcfg.get("dim", 768)),
            depth     = int(ldcfg.get("depth", 16)),
            num_heads = int(ldcfg.get("num_heads", 12)),
            ffn_mult  = float(ldcfg.get("ffn_mult", 4.0)),
            use_gradient_checkpoint = bool(ldcfg.get("use_gradient_checkpoint", True)),
        )

        # --- Stage B: pixel decoder ---
        pdcfg = cfg["pixel_decoder"]
        self.pixel_decoder = VJepaPixelDecoder(
            in_channels = 3,
            base_dim    = int(pdcfg.get("base_dim", 48)),
            num_blocks  = tuple(pdcfg.get("num_blocks", (2, 4, 4, 6, 8))),
            heads       = tuple(pdcfg.get("heads", (2, 4, 8, 8, 8))),
            ffn_exp     = float(pdcfg.get("ffn_exp", 2.66)),
            vjepa_dim   = self.vjepa_dim,
            vjepa_grid  = vjepa_grid,
            vjepa_layer_per_scale = tuple(
                pdcfg.get("vjepa_layer_per_scale", (0, 0, 1, 2, 3))
            ),
            cross_attn_at_scales = tuple(pdcfg.get("cross_attn_at_scales", (2, 3, 4))),
            use_gradient_checkpoint = bool(pdcfg.get("use_gradient_checkpoint", True)),
        )

        self.num_inference_steps = int(cfg.get("inference_steps", 8))
        # Optional refiner is treated as an engineering trick (not core to v6).
        # It can be added later via cfg.refiner.enabled = true; we keep the
        # class import out for now to keep the v6 main path clean.

    # ------------------------------------------------------------------
    # Param accessors
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.latent_dit.parameters():
            if p.requires_grad:
                yield p
        for p in self.pixel_decoder.parameters():
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
    # Frozen encoders
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(video_01)["feats"])

    # ------------------------------------------------------------------
    # Training-time forward
    # ------------------------------------------------------------------
    def forward(
        self,
        video_lq: torch.Tensor,
        video_hq: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """If video_hq is provided (training), returns flow + pixel outputs.

        Returns:
            pred_pixels   : (B, 3, T, H, W) — pixel decoder output
            v_pred        : (B, N, D) — DiT velocity prediction (training only)
            v_target      : (B, N, D) — flow target = E(HQ)[-1] - E(LQ)[-1]
            t_used        : (B,) — sampled flow time
            e_lq_layers   : the raw V-JEPA layers (for diagnostics)
        """
        e_lq_layers = self.encode_video_features(video_lq)
        target_dtype = video_lq.dtype
        e_lq_layers = [f.to(target_dtype) for f in e_lq_layers]

        out: Dict[str, torch.Tensor] = {"e_lq_layers": e_lq_layers}

        if video_hq is not None:
            # Training: compute flow target and teacher-force decoder with E(HQ)[-1].
            e_hq_layers = self.encode_video_features(video_hq)
            e_hq_layers = [f.to(target_dtype) for f in e_hq_layers]

            B = video_lq.shape[0]
            if t is None:
                t = torch.rand(B, device=video_lq.device, dtype=torch.float32)
            t_b = t.view(B, 1, 1).to(target_dtype)
            z_t = (1.0 - t_b) * e_lq_layers[self.target_layer] + t_b * e_hq_layers[self.target_layer]
            v_target = (e_hq_layers[self.target_layer] - e_lq_layers[self.target_layer])

            v_pred = self.latent_dit(z_t, t)
            out["v_pred"] = v_pred
            out["v_target"] = v_target
            out["t_used"] = t

            # Decoder conditioning: teacher-force E(HQ) at target_layer; raw E(LQ) elsewhere.
            cond_layers = list(e_lq_layers)
            cond_layers[self.target_layer] = e_hq_layers[self.target_layer]
            pred_pixels = self.pixel_decoder(video_lq, cond_layers)
            out["pred_pixels"] = pred_pixels
        else:
            # Inference: DiT solves ODE to refine target_layer; decoder uses refined.
            refined = self.latent_dit.sample(
                e_lq_layers[self.target_layer],
                num_steps=self.num_inference_steps,
            )
            cond_layers = list(e_lq_layers)
            cond_layers[self.target_layer] = refined
            out["pred_pixels"] = self.pixel_decoder(video_lq, cond_layers)
            out["refined"] = refined

        return out

    # ------------------------------------------------------------------
    # Inference convenience
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
