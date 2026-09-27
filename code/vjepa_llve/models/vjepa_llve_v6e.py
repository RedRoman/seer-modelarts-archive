"""V-JEPA-LLVE v6.E end-to-end model — PMRF (Posterior-Mean Rectified Flow).

Architecture vs v6.D:
    Same Retinex DiT + CFIM. ONLY the flow trajectory and decoder input change:
        v6.D: V-JEPA(LQ) -- rectified flow --> V-JEPA(HQ)
        v6.E: V-JEPA(v6.0(LQ)) -- rectified flow --> V-JEPA(HQ)
              ^^^^^^^^^^^^^^^^^
              MMSE estimate (frozen v6.0 best.pt, PSNR 25.32 dB)

Why PMRF (Posterior-Mean Rectified Flow, ICLR 2025):
    Theoretical guarantee: starting flow from MMSE estimate gives optimal MSE
    *and* perfect perception. Practically:
      - Flow trajectory is much shorter (residual transport instead of full).
      - Lower-bound at inference is the MMSE itself (~25.32 dB),
        not zero. Identity-at-init ⇒ v6.E starts at v6.0 baseline already.

Identity-at-init guaranteed by:
    1. DiT.final_proj_R/L zero-init  ⇒ v_pred = 0  ⇒ refined = E(v6.0_out)
    2. Decoder.output_conv zero-init ⇒ residual = 0
    3. Decoder input = v6.0(LQ) (NOT raw LQ as in v6.D)
    Total: out = v6.0(LQ) ⇒ initial val PSNR ~= 25.32 dB.

Components:
    self.mmse:           VJepaLLVE_v6 (frozen, weights from v6.0 best.pt)
    self.backbone:       V-JEPA 2.1 Encoder (frozen, shared with v6.0)
    self.latent_dit:     RetinexLatentDiT (300M, trainable)
    self.pixel_decoder:  VJepaPixelDecoder + VolumeCFIM bottleneck (110M, trainable)

Forward (training):
    y_mmse = mmse.sample(LQ)["pred_pixels"]    # frozen, no grad
    e_y    = backbone(y_mmse)                  # 4 layers
    e_hq   = backbone(HQ)                      # for flow target
    z_y    = e_y[target_layer]; z_hq = e_hq[target_layer]
    z_t    = (1-t)·z_y + t·z_hq
    v_target = z_hq - z_y                       # short residual
    v_pred = latent_dit(z_t, t)
    decoder input = y_mmse pixels (NOT LQ)
    decoder cond  = e_y (LQ-side); deepest layer teacher-forced with e_hq

Forward (inference):
    y_mmse = mmse.sample(LQ)["pred_pixels"]
    e_y    = backbone(y_mmse)
    refined = latent_dit.sample(e_y[target_layer], num_steps)
    decoder input = y_mmse; cond[target] = refined
    output = y_mmse + decoder_residual ≈ y_mmse at init.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .latent_dit_retinex import RetinexLatentDiT
from .pixel_decoder import VJepaPixelDecoder
from .vjepa_backbone import build_vjepa_backbone
from .vjepa_llve_v6 import VJepaLLVE_v6


def _load_v6_mmse(cfg: Dict[str, Any], ckpt_path: str, device) -> VJepaLLVE_v6:
    """Build a frozen v6.0 model and load best.pt weights for use as MMSE predictor."""
    mmse = VJepaLLVE_v6(cfg).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mmse.latent_dit.load_state_dict(state["latent_dit"], strict=True)
    mmse.pixel_decoder.load_state_dict(state["pixel_decoder"], strict=True)
    for p in mmse.parameters():
        p.requires_grad_(False)
    mmse.eval()
    return mmse


class VJepaLLVE_v6E(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # --- shared frozen V-JEPA backbone (used for both MMSE encoding and target encoding) ---
        self.backbone: nn.Module = build_vjepa_backbone(cfg["backbone"])
        self.vjepa_dim = int(self.backbone.hidden_dim)
        self.target_layer = int(cfg.get("predictor", {}).get("target_layer_idx", 3))

        # --- frozen MMSE predictor: v6.0 model with best.pt loaded ---
        # Note: v6.0 needs its own backbone reference. To save memory we let v6.0 SHARE
        # the same backbone instance; v6.0's backbone won't get gradient anyway.
        mmse_cfg = cfg["mmse"]
        ckpt_path = mmse_cfg["ckpt_path"]
        # Build v6.0 with cfg's full model spec (it expects the same backbone cfg).
        v6_cfg_for_mmse = {
            "backbone":      cfg["backbone"],
            "predictor":     mmse_cfg.get("predictor", {"enabled": False}),
            "latent_dit":    mmse_cfg["latent_dit"],
            "pixel_decoder": mmse_cfg["pixel_decoder"],
            "inference_steps": int(mmse_cfg.get("inference_steps", 8)),
        }
        # Build the v6 model — its backbone will be a fresh instance. This costs ~600MB
        # but keeps the load_state_dict logic clean.
        self.mmse = VJepaLLVE_v6(v6_cfg_for_mmse)
        # Load weights
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.mmse.latent_dit.load_state_dict(state["latent_dit"], strict=True)
        self.mmse.pixel_decoder.load_state_dict(state["pixel_decoder"], strict=True)
        for p in self.mmse.parameters():
            p.requires_grad_(False)
        self.mmse.eval()
        # Override num_inference_steps for the MMSE predictor (defaults to 8, keep that).

        # --- V-JEPA token grid ---
        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        # --- Stage A: Retinex Latent DiT (same as v6.D) ---
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

        # --- Stage B: PixelDecoder + CFIM bottleneck (same as v6.D) ---
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
        # Also re-freeze MMSE just in case
        for p in self.mmse.parameters():
            p.requires_grad_(False)
        self.mmse.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep backbone + mmse in eval mode
        self.backbone.eval()
        self.mmse.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        return list(self.backbone(video_01)["feats"])

    @torch.no_grad()
    def predict_mmse(self, video_lq: torch.Tensor) -> torch.Tensor:
        """Run frozen v6.0 model on LQ to get MMSE estimate (~25.3 dB on SDSD)."""
        out = self.mmse.sample(video_lq)
        return out["pred_pixels"].clamp(0, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        video_lq: torch.Tensor,
        video_hq: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        target_dtype = video_lq.dtype

        # 1. Run frozen v6.0 → MMSE estimate (in pixel space)
        y_mmse = self.predict_mmse(video_lq).to(target_dtype)

        # 2. Encode y_mmse via shared V-JEPA backbone
        e_y_layers = self.encode_video_features(y_mmse)
        e_y_layers = [f.to(target_dtype) for f in e_y_layers]

        out: Dict[str, torch.Tensor] = {
            "e_y_layers": e_y_layers,
            "y_mmse":     y_mmse,
        }

        if video_hq is not None:
            # Training: encode HQ, set up flow target between z_y and z_hq.
            e_hq_layers = self.encode_video_features(video_hq)
            e_hq_layers = [f.to(target_dtype) for f in e_hq_layers]

            B = video_lq.shape[0]
            if t is None:
                t = torch.rand(B, device=video_lq.device, dtype=torch.float32)
            t_b = t.view(B, 1, 1).to(target_dtype)

            z_y = e_y_layers[self.target_layer]
            z_hq = e_hq_layers[self.target_layer]
            z_t = (1.0 - t_b) * z_y + t_b * z_hq
            v_target = z_hq - z_y      # short residual: smaller magnitude than v6.D's

            dit_out = self.latent_dit(z_t, t)
            out["v_pred"]   = dit_out["v_pred"]
            out["v_target"] = v_target
            out["v_R"]      = dit_out["v_R"]
            out["v_L"]      = dit_out["v_L"]
            out["z_R_out"]  = dit_out["z_R_out"]
            out["z_L_out"]  = dit_out["z_L_out"]
            out["t_used"]   = t

            cond_layers = list(e_y_layers)
            cond_layers[self.target_layer] = e_hq_layers[self.target_layer]
            # Decoder input is y_mmse, NOT raw LQ. Decoder learns to refine y_mmse.
            out["pred_pixels"] = self.pixel_decoder(y_mmse, cond_layers)
        else:
            # Inference: solve flow ODE from z_y → refined ≈ z_hq.
            refined = self.latent_dit.sample(
                e_y_layers[self.target_layer],
                num_steps=self.num_inference_steps,
            )
            cond_layers = list(e_y_layers)
            cond_layers[self.target_layer] = refined
            out["refined"] = refined
            out["pred_pixels"] = self.pixel_decoder(y_mmse, cond_layers)

        return out

    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
