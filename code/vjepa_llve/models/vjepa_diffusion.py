"""V-JEPA-LLVE v5 end-to-end model: VAE + V-JEPA backbone + predictor refiner + DiT.

Pipeline (training):
    LQ video                         HQ video
       │                                │
       │ VAE.encode (frozen)            │ VAE.encode (frozen)
       ▼                                ▼
     z_LQ                              z_HQ
       │                                │
       │   ───── sample t ~ U(0,1) ────►│
       │   z_t = (1-t)·z_LQ + t·z_HQ
       │   v_target = z_HQ - z_LQ
       │
       │ V-JEPA encoder (frozen) ──► E(LQ) at 4 layers
       │                                │
       │                                ▼
       │     CounterfactualRefiner (LoRA + identity-gate)
       │                                │
       │                                ▼
       │     refined V-JEPA features   ── + (optionally) E(HQ) target for loss
       │                                │
       ▼                                ▼
    DiT Velocity Field (cross-attn KV from refined features)
       v_pred = DiT(z_t, t, refined)
       Loss: MSE(v_pred, v_target) + L1(refined, E(HQ).detach())

Pipeline (inference):
    LQ video → z_LQ, refined features
    Solve Euler ODE: z(0)=z_LQ; for k=0..N-1: z+=dt·DiT(z, t_k, refined)
    VAE.decode(z(1)) → HQ pixels

Identity-at-init:
    - DiT.final_proj zero-init → v ≡ 0 → ODE returns z_LQ unchanged → output ≈ LQ
    - CounterfactualRefiner.gate=0 → refined ≡ raw E(LQ)
    Together: total model output ≈ LQ at step 0.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from .dit_velocity import DiTVelocityField
from .predictor_lora import CounterfactualRefiner
from .vae_wrapper import SDVAEWrapper
from .vjepa_backbone import build_vjepa_backbone


class VJepaDiffusion(nn.Module):

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # --- frozen encoders ---
        self.vae = SDVAEWrapper(
            repo=cfg.get("vae", {}).get("repo", "stabilityai/sd-vae-ft-mse"),
            dtype=torch.float16,
            deterministic_encode=bool(cfg.get("vae", {}).get("deterministic_encode", False)),
        )
        self.backbone: nn.Module = build_vjepa_backbone(cfg["backbone"])
        ctx_dim: int = int(self.backbone.hidden_dim)

        # --- counterfactual refiner (V-JEPA 2.1 predictor + LoRA) ---
        pcfg = cfg.get("predictor", {}) or {}
        self.predictor_enabled = bool(pcfg.get("enabled", True))
        self.predictor_target_layer = int(pcfg.get("target_layer_idx", 3))
        if self.predictor_enabled:
            bcfg = cfg["backbone"]
            self.refiner = CounterfactualRefiner(
                local_repo        = pcfg.get("local_repo", bcfg.get("local_repo")),
                ckpt_path         = pcfg.get("ckpt_path", bcfg.get("local_ckpt", "")),
                encoder_embed_dim = ctx_dim,
                image_size        = int(bcfg.get("image_size", 256)),
                num_frames        = int(bcfg.get("frames_per_clip", 16)),
                patch_size        = int(bcfg.get("patch_size", 16)),
                tubelet_size      = int(bcfg.get("tubelet_size", 2)),
                predictor_embed_dim = int(pcfg.get("predictor_embed_dim", 384)),
                depth             = int(pcfg.get("depth", 12)),
                num_heads         = int(pcfg.get("num_heads", 12)),
                lora_rank         = int(pcfg.get("lora_rank", 16)),
                freeze_body       = bool(pcfg.get("freeze_body", True)),
            )

        # --- DiT velocity field ---
        dcfg = cfg["dit"]
        image_size = int(cfg["backbone"].get("image_size", 256))
        latent_h = latent_w = image_size // self.vae.downsample
        self.dit = DiTVelocityField(
            latent_channels = self.vae.latent_channels,
            latent_h = latent_h,
            latent_w = latent_w,
            num_frames = int(cfg["backbone"].get("frames_per_clip", 16)),
            patch = int(dcfg.get("patch", 2)),
            dim = int(dcfg.get("dim", 1024)),
            depth = int(dcfg.get("depth", 28)),
            num_heads = int(dcfg.get("num_heads", 16)),
            ffn_mult = float(dcfg.get("ffn_mult", 4.0)),
            ctx_dim = ctx_dim,
            cross_attn_blocks = tuple(dcfg.get("cross_attn_blocks", (3, 7, 11, 15, 19, 23, 27))),
            cross_attn_layer_map = tuple(dcfg.get("cross_attn_layer_map", (0, 1, 2, 3, 2, 3, 3))),
            use_gradient_checkpoint = bool(dcfg.get("use_gradient_checkpoint", True)),
        )

        self.num_inference_steps = int(cfg.get("inference_steps", 8))

    # ------------------------------------------------------------------
    # Param accessors
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.dit.parameters():
            if p.requires_grad:
                yield p
        if self.predictor_enabled:
            for p in self.refiner.parameters():
                if p.requires_grad:
                    yield p

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.vae.eval()
        return self

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_pixels_to_latent(self, pixels_01: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixels_01)

    @torch.no_grad()
    def decode_latent_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)

    @torch.no_grad()
    def encode_video_features(self, video_01: torch.Tensor) -> List[torch.Tensor]:
        out = self.backbone(video_01)
        return [f for f in out["feats"]]

    # ------------------------------------------------------------------
    # Build cross-attn context layers for DiT, refining the deepest one
    # ------------------------------------------------------------------
    def make_ctx_layers(
        self, raw_layers: List[torch.Tensor], target_dtype: torch.dtype,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Cast layers to target dtype; if predictor enabled, replace the
        target layer with refined output. Returns (ctx_for_dit, refined_or_None)."""
        ctx = [c.to(target_dtype) for c in raw_layers]
        refined = None
        if self.predictor_enabled:
            tl = self.predictor_target_layer
            refined = self.refiner(ctx[tl], mod="video")
            ctx[tl] = refined
        return ctx, refined

    # ------------------------------------------------------------------
    # Training-time forward: predict velocity at sampled t
    # ------------------------------------------------------------------
    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor,
        ctx_layers: List[torch.Tensor],
    ) -> torch.Tensor:
        return self.dit(z_t, t, ctx_layers)

    # ------------------------------------------------------------------
    # Inference: ODE Euler from z_LQ
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self, video_lq_01: torch.Tensor, num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        N = num_steps if num_steps is not None else self.num_inference_steps
        z_LQ = self.encode_pixels_to_latent(video_lq_01)
        raw_feats = self.encode_video_features(video_lq_01)

        dit_dtype = next(self.dit.parameters()).dtype
        ctx, _ = self.make_ctx_layers(raw_feats, target_dtype=dit_dtype)
        z = z_LQ.to(dit_dtype)

        dt = 1.0 / N
        for k in range(N):
            t_k = torch.full(
                (z.shape[0],), k * dt + 0.5 * dt,
                device=z.device, dtype=torch.float32,
            )
            v = self.dit(z, t_k, ctx)
            z = z + dt * v

        pred = self.decode_latent_to_pixels(z.to(self.vae.dtype))
        return {"pred": pred, "z_LQ": z_LQ, "z_pred": z}
