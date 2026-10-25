"""Frozen V-JEPA 2.1 predictor used as a prior on the output's temporal dynamics.

A video world model has two halves, an encoder and a predictor. The earlier
version of this work used only the encoder, as a source of conditioning tokens,
and its two predictor-based modules (M1, M2) were measured inert: the gate
converged to ~1e-2 and the jepa loss never decreased. The reason is visible in
``models/predictor_lora.py`` -- the pretrained ``predictor_proj`` head was
replaced by a randomly initialised 384->1024 projection, so the predictor's
learned predictions were discarded, and the loss compared two latents without
ever reaching the pixels.

This module uses the predictor the way the checkpoint trained it. Given the
latents of the first part of a clip the predictor estimates the rest in the
teacher's representation space (``predictor_proj``, 384 -> 1664); run on the
later part instead, its auxiliary context head (``predictor_proj_context``,
trained alongside) reports what is actually there. The L1 gap between estimate
and report is how unpredictable the clip's dynamics are under the world model.

That gap is not something to drive to zero. A static or over-smoothed clip is
*more* predictable than a real one and a flickering clip is less so, so natural
video sits at a particular gap rather than at no gap at all. The loss therefore
matches the output's gap to the ground truth's rather than minimising it; the
one-sided variant is kept only as an ablation.

Everything here is frozen. The only gradient path is through ``tokens``, i.e.
back into the encoder and from there into the predicted pixels.
"""
from __future__ import annotations

import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class WorldModelPrior(nn.Module):
    """Measures how unpredictable a clip's latent dynamics are, and compares
    that quantity between the prediction and the ground truth.

    Parameters
    ----------
    local_repo, ckpt_path :
        V-JEPA 2.1 repository and the checkpoint holding the ``predictor`` blob.
    grid :
        Token grid (T_tubelets, H_patches, W_patches) of the encoder feeding us.
    context_tubelets :
        How many leading temporal tubelets act as context. The remainder are the
        prediction target, so this sets how far ahead the prior looks.
    two_sided :
        True matches the ground truth's level; False minimises the gap (ablation).
    """

    def __init__(
        self,
        local_repo: str,
        ckpt_path: str,
        grid: Tuple[int, int, int] = (8, 16, 16),
        encoder_embed_dim: int = 1024,
        predictor_embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        image_size: int = 256,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        context_tubelets: int = 4,
        two_sided: bool = True,
        metric: str = "cos",
    ) -> None:
        super().__init__()
        if local_repo not in sys.path:
            sys.path.insert(0, local_repo)
        try:
            from app.vjepa_2_1.models.predictor import vit_predictor  # type: ignore
        except Exception as e:  # pragma: no cover - import-time environment issue
            raise ImportError(
                f"Could not import V-JEPA 2.1 predictor from {local_repo}: {e}"
            )

        self.grid = tuple(int(g) for g in grid)
        self.encoder_embed_dim = int(encoder_embed_dim)
        self.context_tubelets = int(context_tubelets)
        self.two_sided = bool(two_sided)
        self.metric = str(metric).lower()
        if self.metric not in ("l1", "rel", "cos"):
            raise ValueError(f"metric must be l1|rel|cos, got {metric!r}")
        if not 0 < self.context_tubelets < self.grid[0]:
            raise ValueError(
                f"context_tubelets must be in (0, {self.grid[0]}), got {context_tubelets}"
            )

        # return_all_tokens=True exposes predictor_proj_context, which is what
        # lets us read the observed side in the same 1664-d teacher space.
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
            return_all_tokens=True,
            use_activation_checkpointing=True,
        )
        self._load_predictor(ckpt_path)

        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.predictor.eval()

        tt, hh, ww = self.grid
        per_tubelet = hh * ww
        split = self.context_tubelets * per_tubelet
        idx = torch.arange(tt * per_tubelet, dtype=torch.long)
        self.register_buffer("early_idx", idx[:split], persistent=False)
        self.register_buffer("late_idx", idx[split:], persistent=False)

    # ------------------------------------------------------------------
    def _load_predictor(self, ckpt_path: str) -> None:
        if not ckpt_path:
            raise ValueError("WorldModelPrior needs the pretrained predictor")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not (isinstance(state, dict) and "predictor" in state):
            raise ValueError(f"ckpt at {ckpt_path} has no 'predictor' field")
        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in state["predictor"].items():
            nk = k
            for prefix in ("module.backbone.", "module.", "backbone."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    break
            cleaned[nk] = v
        miss, unexp = self.predictor.load_state_dict(cleaned, strict=False)
        # Both heads must come from the checkpoint; a randomly initialised head
        # here would silently reproduce the M1/M2 failure this module exists to
        # avoid, so we refuse rather than train against noise.
        for required in ("predictor_proj.weight", "predictor_proj_context.weight"):
            if required in miss:
                raise RuntimeError(
                    f"{required} missing from {ckpt_path}; the predictor's output "
                    f"head is not pretrained and the prior would be meaningless"
                )
        print(
            f"[WorldModelPrior] loaded predictor from {ckpt_path} | "
            f"missing={len(miss)} unexpected={len(unexp)} | "
            f"context={self.context_tubelets}/{self.grid[0]} tubelets | "
            f"metric={self.metric} two_sided={self.two_sided}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def unpredictability(self, tokens: torch.Tensor,
                         metric: Optional[str] = None) -> torch.Tensor:
        """Gap between the predictor's estimate of the late tubelets and its own
        reading of them, both in teacher space.

        tokens : (B, N, encoder_embed_dim) final-layer encoder tokens, N = prod(grid).
        Returns a scalar; gradient flows back through ``tokens``.

        ``metric`` selects how the gap is measured, which matters more than it
        looks. The plain L1 of the difference is **not** scale invariant: a
        spatial blur shrinks the magnitude of the target features and so shrinks
        the L1 along with them, which made the first version of this measure
        respond three times more strongly to blur than to flicker and fail the
        specificity test in ``tools/lpm_probe.py``. ``rel`` and ``cos`` divide
        that scale out.
        """
        b, n, d = tokens.shape
        if d != self.encoder_embed_dim:
            raise ValueError(f"token dim {d} != {self.encoder_embed_dim}")
        if n != self.early_idx.numel() + self.late_idx.numel():
            raise ValueError(f"N={n} does not match grid {self.grid}")
        metric = (metric or self.metric).lower()

        early = self.early_idx.unsqueeze(0).expand(b, -1)
        late = self.late_idx.unsqueeze(0).expand(b, -1)

        # Estimate: see the early tubelets, predict the late ones.
        est, _ = self.predictor(tokens[:, self.early_idx, :], masks_x=early, masks_y=late)
        # Report: see the late tubelets, read them out through the context head.
        _, obs = self.predictor(tokens[:, self.late_idx, :], masks_x=late, masks_y=early)
        obs = obs.detach()

        if metric == "l1":
            return (est - obs).abs().mean()
        if metric == "rel":
            return (est - obs).abs().mean() / obs.abs().mean().clamp_min(1e-6)
        if metric == "cos":
            return 1.0 - torch.nn.functional.cosine_similarity(est, obs, dim=-1).mean()
        raise ValueError(f"unknown metric {metric!r}")

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens_pred: torch.Tensor,
        tokens_gt: Optional[torch.Tensor] = None,
        target_level: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (loss, e_pred, e_ref).

        ``tokens_gt`` gives the per-clip reference level and is read under
        no_grad. ``target_level`` substitutes a fixed scalar for it, which is the
        variant usable without paired ground truth; pass exactly one of them when
        ``two_sided`` is set.
        """
        e_pred = self.unpredictability(tokens_pred)

        if not self.two_sided:
            return e_pred, e_pred.detach(), e_pred.new_zeros(())

        if target_level is not None:
            e_ref = e_pred.new_tensor(float(target_level))
        elif tokens_gt is not None:
            with torch.no_grad():
                e_ref = self.unpredictability(tokens_gt)
        else:
            raise ValueError("two_sided needs tokens_gt or target_level")
        return (e_pred - e_ref).abs(), e_pred.detach(), e_ref.detach()
