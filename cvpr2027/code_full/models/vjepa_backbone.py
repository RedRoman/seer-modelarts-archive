"""V-JEPA 2.1 backbone wrapper — frozen feature extractor.

We only support the Local21 path (V-JEPA 2.1 from local clone of
facebookresearch/vjepa2). HuggingFace doesn't yet host V-JEPA 2.1 weights
and our Predictor counterfactual logic depends on the 2.1-specific predictor.

Input: (B, 3, T, H, W) in [0, 1]
Output:
    {"feats": List[Tensor (B, N, D)],   # multi-layer dense tokens
     "token_grid": (T', H', W'),
     "hidden_dim": int}

Pre-processing:
  1. spatial resize to image_size (default 256)
  2. temporal resample to frames_per_clip (default 16): nearest subsample if
     T_in > target_T, repeat-pad if T_in < target_T
  3. ImageNet normalization
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_bcthw(video_01: torch.Tensor) -> torch.Tensor:
    mean = video_01.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
    std  = video_01.new_tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
    return (video_01 - mean) / std


def _resize_bcthw(video: torch.Tensor, target_hw: int,
                  mode: str = "bilinear") -> torch.Tensor:
    B, C, T, H, W = video.shape
    if H == target_hw and W == target_hw:
        return video
    v = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    v = F.interpolate(
        v, size=(target_hw, target_hw), mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return v.reshape(B, T, C, target_hw, target_hw).permute(0, 2, 1, 3, 4).contiguous()


def _resample_frames(video: torch.Tensor, target_T: int) -> torch.Tensor:
    B, C, T, H, W = video.shape
    if T == target_T:
        return video
    if T < target_T:
        rep = (target_T + T - 1) // T
        return video.repeat(1, 1, rep, 1, 1)[:, :, :target_T]
    idx = torch.linspace(0, T - 1, target_T, device=video.device).round().long()
    return video[:, :, idx]


# ---------------------------------------------------------------------------
# Local 21 loader (V-JEPA 2.1 via app.vjepa_2_1.models)
# ---------------------------------------------------------------------------
class VJepaBackboneLocal21(nn.Module):
    """V-JEPA 2.1 ViT-L from local clone of facebookresearch/vjepa2.

    The 2.1 encoder lives in `app.vjepa_2_1.models.vision_transformer` with
    extra machinery: modality_embedding, norms_block (DSS), native out_layers
    support, RoPE attention.

    Loads weights from a checkpoint that has structure:
      {"encoder": {...}, "predictor": {...}, "ema_encoder": {...}, ...}
    State keys are wrapped as `module.backbone.<name>` and we strip those.
    """

    def __init__(
        self,
        local_repo:      str,
        local_ckpt:      str,
        feature_layers:  Tuple[int, ...] = (5, 11, 17, 23),
        image_size:      int  = 256,
        frames_per_clip: int  = 16,
        patch_size:      int  = 16,
        tubelet_size:    int  = 2,
        freeze:          bool = True,
        # Pass-through for V-JEPA 2.1 ViT constructor
        use_rope:           bool = True,
        modality_embedding: bool = True,
        interpolate_rope:   bool = True,
        is_causal:          bool = False,
        uniform_power:      bool = True,
        has_cls_first:      bool = False,
        n_registers:        int  = 0,
    ) -> None:
        super().__init__()
        if not os.path.isdir(local_repo):
            raise FileNotFoundError(
                f"local_repo not found: {local_repo}. "
                f"Clone facebookresearch/vjepa2 there."
            )
        if local_repo not in sys.path:
            sys.path.insert(0, local_repo)
        try:
            from app.vjepa_2_1.models import vision_transformer as vjepa21_vit  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Failed to import app.vjepa_2_1.models.vision_transformer "
                f"from {local_repo}: {e}"
            )

        self.feature_layers = list(feature_layers)
        self.image_size = int(image_size)
        self.frames_per_clip = int(frames_per_clip)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)

        self.model = vjepa21_vit.vit_large(
            patch_size         = patch_size,
            img_size           = (image_size, image_size),
            num_frames         = frames_per_clip,
            tubelet_size       = tubelet_size,
            out_layers         = self.feature_layers,
            use_rope           = use_rope,
            modality_embedding = modality_embedding,
            interpolate_rope   = interpolate_rope,
            is_causal          = is_causal,
            uniform_power      = uniform_power,
            has_cls_first      = has_cls_first,
            n_registers        = n_registers,
            img_temporal_dim_size = None,    # video-only for LLVE
            n_output_distillation = 4,
            use_activation_checkpointing = False,
        )

        if local_ckpt:
            state = torch.load(local_ckpt, map_location="cpu", weights_only=False)
            for key in ("encoder", "target_encoder", "ema_encoder", "model", "state_dict"):
                if isinstance(state, dict) and key in state:
                    state = state[key]
                    break
            cleaned: Dict[str, torch.Tensor] = {}
            for k, v in state.items():
                nk = k
                for prefix in ("module.backbone.", "module.", "backbone."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                        break
                cleaned[nk] = v
            miss, unexp = self.model.load_state_dict(cleaned, strict=False)
            print(
                f"[VJepaBackboneLocal21] loaded {local_ckpt} | "
                f"missing={len(miss)} unexpected={len(unexp)}"
            )
            if miss:
                print(f"  missing (first 5): {list(miss)[:5]}")
            if unexp:
                print(f"  unexpected (first 5): {list(unexp)[:5]}")

        self.hidden_dim = int(
            getattr(self.model, "embed_dim",
                    getattr(self.model, "hidden_dim", 1024))
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def _forward(self, video_01: torch.Tensor) -> Dict[str, Any]:
        # 1. Spatial resize
        video_01 = _resize_bcthw(video_01, self.image_size, mode="bilinear")
        # 2. Temporal resample
        video_01 = _resample_frames(video_01, self.frames_per_clip)
        # 3. ImageNet normalize, keep (B, C, T, H, W) layout (2.1 ViT accepts this)
        x = _normalize_bcthw(video_01)

        out = self.model(x)
        if not isinstance(out, list):
            out = [out]
        feats: List[torch.Tensor] = out

        token_grid = (
            self.frames_per_clip // self.tubelet_size,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
        )
        return {"feats": feats, "token_grid": token_grid,
                "hidden_dim": self.hidden_dim}

    def forward(self, video_01: torch.Tensor) -> Dict[str, Any]:
        """Note: caller decides whether to wrap in torch.no_grad. We keep grad
        flow open here so that callers can use the backbone as a perceptual
        feature extractor with gradient through the input."""
        return self._forward(video_01)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_vjepa_backbone(cfg: Dict) -> nn.Module:
    """Build a V-JEPA backbone. v5 supports only `source: local_21`.

    Expected fields (with defaults):
        source:          "local_21"  (default and only)
        local_repo:      str (required)
        local_ckpt:      str (required)
        feature_layers:  list[int]   default [5, 11, 17, 23]
        image_size:      int         default 256
        frames_per_clip: int         default 16
        patch_size:      int         default 16
        tubelet_size:    int         default 2
        freeze:          bool        default True
    """
    src = cfg.get("source", "local_21").lower()
    if src not in ("local_21", "local_2.1", "vjepa21"):
        raise ValueError(
            f"V-JEPA-LLVE v5 only supports source=local_21 (V-JEPA 2.1). Got: {src!r}"
        )
    return VJepaBackboneLocal21(
        local_repo      = cfg["local_repo"],
        local_ckpt      = cfg.get("local_ckpt", ""),
        feature_layers  = tuple(cfg.get("feature_layers", (5, 11, 17, 23))),
        image_size      = cfg.get("image_size", 256),
        frames_per_clip = cfg.get("frames_per_clip", 16),
        patch_size      = int(cfg.get("patch_size", 16)),
        tubelet_size    = int(cfg.get("tubelet_size", 2)),
        freeze          = cfg.get("freeze", True),
    )
