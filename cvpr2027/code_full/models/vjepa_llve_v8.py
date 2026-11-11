"""V-JEPA-LLVE v8.B+ end-to-end model.

Architecture:
  V-JEPA Encoder (frozen, 304M)
    ├─→ Hierarchical layers (4): condition Pixel Decoder via cross-attn
    ├─→ Pooled feature (1024,): route Illumination Prompt Bank via softmax
    └─→ Deepest layer (z_LQ): input to V-JEPA Predictor

  V-JEPA Predictor (frozen body + LoRA, ~150M body + 8M trainable)
    └─→ K=3 step rollout: z_k = Predictor(z_{k-1} + α·action_token)
        Returns z_HQ_pred + intermediates for monotonicity loss

  Pixel Branch (~280M trainable)
    Input: video_lq + V-JEPA layers + mixture_weights (from Router)
    Pixel parallel CNN encoder (high-freq bypass)
    Big UNet (BiasFree LN, PixelShuffle, Restormer)
    V-JEPA cross-attn at scales {1,2,3,4}
    Illumination Prompt Bank cross-attn at scales {1,2,3,4}
    Bottleneck VolumeCFIM
    Output: residual + Retinex (R̂, L̂)

  Final: pred_pixels = LQ + residual
         + auxiliary supervision: HQ ≈ R̂ ⊙ L̂
         + counterfactual: z_HQ_pred ≈ V-JEPA(HQ).detach()
         + monotonicity: cos(z_k, z_HQ) increasing in k
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from .illumination_prompt_bank import IlluminationConditionRouter
from .pixel_branch_v8 import VJepaPixelBranch_v8
from .vjepa_backbone import build_vjepa_backbone
from .vjepa_predictor_v8 import VJepaPredictorRefinerV8


class VJepaLLVE_v8(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # frozen backbone
        self.backbone: nn.Module = build_vjepa_backbone(cfg["backbone"])
        self.vjepa_dim = int(self.backbone.hidden_dim)
        self.target_layer = int(cfg.get("predictor", {}).get("target_layer_idx", 3))

        # V-JEPA grid
        bcfg = cfg["backbone"]
        T_v = int(bcfg.get("frames_per_clip", 16)) // int(bcfg.get("tubelet_size", 2))
        H_v = int(bcfg.get("image_size", 256)) // int(bcfg.get("patch_size", 16))
        W_v = H_v
        vjepa_grid = (T_v, H_v, W_v)

        # V-JEPA Predictor refiner with K-step rollout + action token
        pcfg = cfg["predictor"]
        self.predictor_refiner = VJepaPredictorRefinerV8(
            local_repo=pcfg["local_repo"],
            ckpt_path=pcfg["ckpt_path"],
            encoder_embed_dim=self.vjepa_dim,
            image_size=int(bcfg.get("image_size", 256)),
            num_frames=int(bcfg.get("frames_per_clip", 16)),
            patch_size=int(bcfg.get("patch_size", 16)),
            tubelet_size=int(bcfg.get("tubelet_size", 2)),
            predictor_embed_dim=int(pcfg.get("predictor_embed_dim", 384)),
            depth=int(pcfg.get("depth", 12)),
            num_heads=int(pcfg.get("num_heads", 12)),
            lora_rank=int(pcfg.get("lora_rank", 16)),
            num_iters=int(pcfg.get("num_iters", 3)),
        )

        # Illumination condition router (V-JEPA pooled → softmax over K)
        rcfg = cfg.get("illumination_router", {})
        self.illumination_router = IlluminationConditionRouter(
            vjepa_dim=self.vjepa_dim,
            num_conditions=int(rcfg.get("num_conditions", 8)),
        )

        # Pixel branch (big UNet with all enhancements)
        pbcfg = cfg["pixel_branch"]
        self.pixel_branch = VJepaPixelBranch_v8(
            in_channels=3,
            dims=tuple(pbcfg.get("dims", (64, 128, 256, 512, 1024))),
            num_blocks=tuple(pbcfg.get("num_blocks", (4, 8, 8, 12, 16))),
            heads=tuple(pbcfg.get("heads", (4, 8, 8, 16, 16))),
            ffn_exp=float(pbcfg.get("ffn_exp", 2.66)),
            vjepa_dim=self.vjepa_dim,
            vjepa_grid=vjepa_grid,
            vjepa_layer_per_scale=tuple(pbcfg.get("vjepa_layer_per_scale", (0, 0, 1, 2, 3))),
            cross_attn_at_scales=tuple(pbcfg.get("cross_attn_at_scales", (1, 2, 3, 4))),
            prompt_bank_at_scales=tuple(pbcfg.get("prompt_bank_at_scales", (1, 2, 3, 4))),
            prompt_num_conditions=int(rcfg.get("num_conditions", 8)),
            prompt_dim=int(pbcfg.get("prompt_dim", 96)),
            prompt_grid=tuple(pbcfg.get("prompt_grid", (8, 8, 8))),
            pixel_parallel_dims=tuple(pbcfg.get("pixel_parallel_dims", (32, 64, 128, 192))),
            use_cfim_bottleneck=bool(pbcfg.get("use_cfim_bottleneck", True)),
            use_gradient_checkpoint=bool(pbcfg.get("use_gradient_checkpoint", True)),
        )

    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for p in self.predictor_refiner.trainable_parameters():
            yield p
        for p in self.illumination_router.parameters():
            if p.requires_grad:
                yield p
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
        target_dtype = video_lq.dtype
        e_lq_layers = self.encode_video_features(video_lq)
        e_lq_layers = [f.to(target_dtype) for f in e_lq_layers]

        # Pooled feature (mean over tokens) of deepest layer for routing
        z_lq_deep = e_lq_layers[self.target_layer]  # (B, N, D)
        vjepa_pooled = z_lq_deep.mean(dim=1)         # (B, D)
        mixture_weights = self.illumination_router(vjepa_pooled.float()).to(target_dtype)  # (B, K)

        # K-step Predictor rollout for counterfactual latent
        rollout = self.predictor_refiner(z_lq_deep)
        z_hq_pred = rollout["z_final"]
        intermediates = rollout["intermediates"]

        out: Dict[str, torch.Tensor] = {
            "z_hq_pred": z_hq_pred,
            "intermediates": intermediates,
            "mixture_weights": mixture_weights,
            "e_lq_layers": e_lq_layers,
        }

        # For decoder: use refined latent at deepest layer
        cond_layers = list(e_lq_layers)
        if video_hq is not None:
            # Training: teacher-force decoder with E(HQ)[deepest] for stability
            with torch.no_grad():
                e_hq_layers = self.encode_video_features(video_hq)
            e_hq_layers = [f.to(target_dtype) for f in e_hq_layers]
            cond_layers[self.target_layer] = e_hq_layers[self.target_layer]
            out["z_hq_target"] = e_hq_layers[self.target_layer]
        else:
            # Inference: use Predictor's z_hq_pred as deepest layer
            cond_layers[self.target_layer] = z_hq_pred

        # Big pixel decoder
        decoder_out = self.pixel_branch(video_lq, cond_layers, mixture_weights)
        out.update(decoder_out)
        return out

    @torch.no_grad()
    def sample(self, video_lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(video_lq, video_hq=None)
