"""V-JEPA-LLVE v6.D end-to-end model.

Architectural changes vs v6.0:
    1. LatentDiT → RetinexLatentDiT (dual-branch R/L decomposition in latent
       domain; sparse cross-talk; periodic TokenCFIM on R branch).
    2. PixelDecoder bottleneck adds VolumeCFIM in parallel (zero-init residual).
    3. Forward returns auxiliary R/L outputs so the loss can apply Retinex
       regularization (illumination smoothness, reflectance temporal stability).

Identity-at-init preserved:
    - RetinexLatentDiT.final_proj_R/L zero-init → v_pred = 0
    - VJepaCrossAttn.proj zero-init (unchanged)
    - PixelDecoder.output_conv zero-init (unchanged)
    - VolumeCFIM zero-init projections + alpha=0 (sigmoid 0 → 0.5; both branches zero-init though)
    - TokenCFIM same
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .latent_dit_retinex import RetinexLatentDiT
from .pixel_decoder import VJepaPixelDecoder
from .vjepa_backbone import build_vjepa_backbone


class VJepaLLVE_v6D(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # frozen backbone
        self.backbone: nn.Module = build_vjepa_backbone(cfg["backbone"])
        self.vjepa_dim = int(self.backbone.hidden_dim)
        self.target_layer = int(cfg.get("predictor", {}).get("target_layer_idx", 3))

        # V-JEPA token grid
        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        # Stage A: Retinex Latent DiT
        ldcfg = cfg["latent_dit"]
        self.latent_dit = RetinexLatentDiT(
            vjepa_dim = self.vjepa_dim,
            grid      = vjepa_grid,
            dim       = int(ldcfg.get("dim", 768)),
            depth     = int(ldcfg.get("depth", 16)),
            num_heads = int(ldcfg.get("num_heads", 12)),
            ffn_mult  = float(ldcfg.get("ffn_mult", 4.0)),
            cfim_every = int(ldcfg.get("cfim_every", 4)),
            cfim_heads = int(ldcfg.get("cfim_heads", 8)),
            use_gradient_checkpoint = bool(ldcfg.get("use_gradient_checkpoint", True)),
        )

        # Stage B: Pixel Decoder + optional bottleneck CFIM
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
            use_cfim_bottleneck = bool(pdcfg.get("use_cfim_bottleneck", True)),
        )

        self.num_inference_steps = int(cfg.get("inference_steps", 8))

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
    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(video_01)["feats"])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        video_lq: torch.Tensor,
        video_hq: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        e_lq_layers = self.encode_video_features(video_lq)
        target_dtype = video_lq.dtype
        e_lq_layers = [f.to(target_dtype) for f in e_lq_layers]

        out: Dict[str, torch.Tensor] = {"e_lq_layers": e_lq_layers}

        if video_hq is not None:
            e_hq_layers = self.encode_video_features(video_hq)
            e_hq_layers = [f.to(target_dtype) for f in e_hq_layers]

            B = video_lq.shape[0]
            if t is None:
                t = torch.rand(B, device=video_lq.device, dtype=torch.float32)
            t_b = t.view(B, 1, 1).to(target_dtype)
            z_lq = e_lq_layers[self.target_layer]
            z_hq = e_hq_layers[self.target_layer]
            z_t = (1.0 - t_b) * z_lq + t_b * z_hq
            v_target = z_hq - z_lq

            dit_out = self.latent_dit(z_t, t)
            out["v_pred"]   = dit_out["v_pred"]
            out["v_target"] = v_target
            out["v_R"]      = dit_out["v_R"]
            out["v_L"]      = dit_out["v_L"]
            out["z_R_out"]  = dit_out["z_R_out"]   # for L-smooth / R-temporal regularizers
            out["z_L_out"]  = dit_out["z_L_out"]
            out["t_used"]   = t

            cond_layers = list(e_lq_layers)
            cond_layers[self.target_layer] = e_hq_layers[self.target_layer]
            out["pred_pixels"] = self.pixel_decoder(video_lq, cond_layers)
        else:
            refined = self.latent_dit.sample(
                e_lq_layers[self.target_layer],
                num_steps=self.num_inference_steps,
            )
            cond_layers = list(e_lq_layers)
            cond_layers[self.target_layer] = refined
            out["pred_pixels"] = self.pixel_decoder(video_lq, cond_layers)
            out["refined"] = refined

        return out

    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
