"""V-JEPA 2.1 Predictor body + LoRA, used as a counterfactual feature refiner.

Key change vs the v4.1 implementation that caused PSNR oscillation:
    v4.1 used the predictor's masked-prediction API with two complementary
    random masks per forward. This created stochastic input distribution to
    the downstream decoder cross-attn, destabilizing training.

    v5 bypasses the masking machinery entirely. We use the predictor as a
    plain transformer body that ingests ALL positions and refines them via
    LoRA-adapted attention. The pretrained predictor weights still carry
    1B+ video pretraining knowledge; LoRA learns the LQ→HQ counterfactual
    delta. Identity-at-init via a learnable gate (zero-init).

Forward:
    tokens (B, N, 1024)
        │
        ▼ predictor.predictor_embed  (1024 → 384)
    h (B, N, 384)
        │
        ▼ + video_mod_embed (broadcast)
        ▼ for each predictor_block: h = block(h, mask=arange(N))   # LoRA-adapted attn
        ▼ predictor.predictor_norm
    h_norm (B, N, 384)
        │
        ▼ new_proj  (384 → 1024, trainable, trunc_normal init)
    delta (B, N, 1024)
        │
        ▼ refined = tokens + gate * delta   # gate=0 at init → identity

Trainable params (~5M total):
    - LoRA A/B on attn.qkv across 12 blocks (~1.5M)
    - new_proj  (384 × 1024 ≈ 400K)
    - gate (1)
    - mask_tokens, predictor_proj_context, mod embeds: not used in our forward;
      remain frozen if loaded.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with low-rank residual update.
        out = base(x) + (alpha/rank) * (x @ A^T) @ B^T
    A is kaiming_uniform-init, B is zero-init → delta = 0 at init.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling


def _inject_lora_on_attn_qkv(predictor: nn.Module, rank: int = 16) -> int:
    n = 0
    for blk in predictor.predictor_blocks:
        if not hasattr(blk, "attn") or not hasattr(blk.attn, "qkv"):
            continue
        if not isinstance(blk.attn.qkv, nn.Linear):
            continue
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank=rank, alpha=float(rank))
        n += 1
    return n


# ---------------------------------------------------------------------------
# Counterfactual refiner
# ---------------------------------------------------------------------------
class CounterfactualRefiner(nn.Module):
    """V-JEPA 2.1 predictor body + LoRA + identity-gate, used as a feature
    refiner without the masked-prediction interface.

    Args
    ----
    local_repo, ckpt_path : str
        Same paths as the V-JEPA backbone uses (the ckpt file has both
        encoder and predictor weight blobs).
    encoder_embed_dim : int
        Dimension of the V-JEPA encoder output (1024 for ViT-L). Predictor
        embedding maps 1024 → predictor_embed_dim, and our new_proj maps
        back to encoder_embed_dim for the counterfactual loss target.
    image_size, num_frames, patch_size, tubelet_size :
        Same geometry as the encoder; predictor's RoPE blocks need this.
    predictor_embed_dim, depth, num_heads :
        Predictor body architecture; must match loaded ckpt (V-JEPA 2.1
        ViT-L predictor: 384, 12, 12).
    lora_rank : int
        LoRA rank on attn.qkv. Default 16 ⇒ ~1.5M trainable.
    """

    def __init__(
        self,
        local_repo: str,
        ckpt_path: str,
        encoder_embed_dim: int = 1024,
        image_size: int = 256,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        lora_rank: int = 16,
        freeze_body: bool = True,
    ) -> None:
        super().__init__()
        if local_repo not in sys.path:
            sys.path.insert(0, local_repo)
        try:
            from app.vjepa_2_1.models.predictor import vit_predictor  # type: ignore
        except Exception as e:
            raise ImportError(
                f"Could not import V-JEPA 2.1 predictor from {local_repo}: {e}"
            )

        self.encoder_embed_dim = int(encoder_embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.grid = (
            num_frames // tubelet_size,
            image_size // patch_size,
            image_size // patch_size,
        )

        # Build predictor matching ckpt structure. NB: vit_predictor() factory
        # hardcodes mlp_ratio=4, qkv_bias=True, norm_layer=LayerNorm.
        self.predictor = vit_predictor(
            img_size=(image_size, image_size),
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            use_mask_tokens=True,
            num_mask_tokens=8,
            zero_init_mask_tokens=False,
            use_rope=True,
            modality_embedding=True,
            img_temporal_dim_size=None,
            interpolate_rope=True,
            n_output_distillation=1,
            teacher_embed_dim=1664,
            use_activation_checkpointing=True,
        )

        self._load_predictor_weights(ckpt_path)

        # Replace predictor_proj with our own 384 → 1024 head (trunc_normal init)
        self.new_proj = nn.Linear(
            predictor_embed_dim, encoder_embed_dim, bias=True,
        )
        nn.init.trunc_normal_(self.new_proj.weight, std=0.02)
        nn.init.zeros_(self.new_proj.bias)

        # Freeze body
        if freeze_body:
            for p in self.predictor.parameters():
                p.requires_grad_(False)

        # Inject LoRA on attn.qkv
        n_lora = _inject_lora_on_attn_qkv(self.predictor, rank=lora_rank)
        self.lora_rank = lora_rank
        self.n_lora_blocks = n_lora

        # Identity-at-init gate
        self.gate = nn.Parameter(torch.zeros(1))

    def _load_predictor_weights(self, ckpt_path: str) -> None:
        if not ckpt_path:
            return
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not (isinstance(state, dict) and "predictor" in state):
            raise ValueError(
                f"ckpt at {ckpt_path} has no 'predictor' field; got keys "
                f"{list(state.keys()) if isinstance(state, dict) else type(state).__name__}"
            )
        cleaned = {}
        for k, v in state["predictor"].items():
            nk = k
            for prefix in ("module.backbone.", "module.", "backbone."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            cleaned[nk] = v
        miss, unexp = self.predictor.load_state_dict(cleaned, strict=False)
        print(
            f"[CounterfactualRefiner] loaded predictor from {ckpt_path} | "
            f"missing={len(miss)} unexpected={len(unexp)}"
        )

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Custom forward: bypass mask machinery, full-token deterministic pass
    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor, mod: str = "video") -> torch.Tensor:
        """tokens: (B, N, encoder_embed_dim). Returns refined (B, N, encoder_embed_dim).

        At init: gate=0 → refined = tokens (identity). After training: gate
        and LoRA learn the counterfactual residual toward E(HQ).
        """
        B, N, D = tokens.shape
        if D != self.encoder_embed_dim:
            raise ValueError(f"token dim {D} != {self.encoder_embed_dim}")
        N_grid = self.grid[0] * self.grid[1] * self.grid[2]
        if N != N_grid:
            raise ValueError(
                f"N={N} != T*H*W={N_grid} for grid {self.grid}"
            )

        # 1. predictor_embed: 1024 → 384 (uses pretrained weights)
        h = self.predictor.predictor_embed(tokens)

        # 2. Modality embed (video). The 2.1 predictor only creates these if
        # img_temporal_dim_size is not None at construction. We constructed
        # with None, so video_mod_embed isn't a parameter — skip safely.
        if hasattr(self.predictor, "video_mod_embed") and getattr(
            self.predictor, "modality_embedding", False,
        ):
            h = h + self.predictor.video_mod_embed.to(h.dtype)

        # 3. Pass through transformer blocks. The mask param to each block
        # gives RoPE position indices. Pass natural [0..N-1] indices to
        # treat all tokens as visible at their grid positions.
        masks = torch.arange(N, device=h.device).unsqueeze(0).expand(B, -1)
        for blk in self.predictor.predictor_blocks:
            h, _ = blk(h, mask=masks)

        # 4. Final norm + our new projection back to encoder dim
        h = self.predictor.predictor_norm(h)
        delta = self.new_proj(h)

        # 5. Identity-at-init residual
        delta_dtype = delta.dtype
        tokens_cast = tokens.to(delta_dtype) if tokens.dtype != delta_dtype else tokens
        return tokens_cast + self.gate * delta
