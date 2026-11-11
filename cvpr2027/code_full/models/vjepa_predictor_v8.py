"""V-JEPA Predictor refiner with K-step rollout + illumination action token.

Extends models/predictor_lora.py CounterfactualRefiner:
  - Adds learnable illumination_action_token (1024-dim)
  - Adds K-step iterative rollout (K=3 default)
  - Returns intermediate latents for monotonicity loss

Conceptual model (counterfactual world rollout):
    z_0 = V-JEPA(LQ)
    for k = 1..K:
        z_k = Predictor(z_{k-1} + α * action_token)
    return z_K (= z_HQ_pred)

Each Predictor call applies a small "step" toward natural-light latent,
borrowing the action-conditioned JEPA paradigm from robotics world models.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .predictor_lora import CounterfactualRefiner


class VJepaPredictorRefinerV8(nn.Module):
    """V-JEPA Predictor + LoRA + action token + K-step rollout.

    Wraps an existing CounterfactualRefiner and:
      1. Maintains a single learnable illumination action token (1024,)
      2. Maintains a learnable scale α (per-step) to inject the token
      3. Runs K iterations: z_k = refiner(z_{k-1} + α_k * action_token)
      4. Returns final z_K AND list of intermediate (z_0, z_1, ..., z_K)
         for monotonicity supervision
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
        num_iters: int = 3,
    ) -> None:
        super().__init__()
        self.refiner = CounterfactualRefiner(
            local_repo=local_repo,
            ckpt_path=ckpt_path,
            encoder_embed_dim=encoder_embed_dim,
            image_size=image_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            lora_rank=lora_rank,
            freeze_body=True,
        )
        self.encoder_embed_dim = encoder_embed_dim
        self.K = int(num_iters)

        # Learnable illumination action direction in V-JEPA latent
        self.action_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        nn.init.trunc_normal_(self.action_token, std=0.02)

        # Per-step injection scale (zero-init → action contributes 0 at init,
        # so refiner.gate=0 keeps identity). Learnable per step for richer dynamics.
        self.alpha = nn.Parameter(torch.zeros(num_iters))

    def trainable_parameters(self):
        for p in self.refiner.trainable_parameters():
            yield p
        yield self.action_token
        yield self.alpha

    def forward(self, z: torch.Tensor) -> dict:
        """z: (B, N, 1024) V-JEPA latent of LQ.

        Returns dict:
          "z_final": (B, N, 1024) refined latent (= z_HQ_pred)
          "intermediates": List[(B, N, 1024)] of length K+1: [z_0, z_1, ..., z_K]
        """
        intermediates = [z]
        for k in range(self.K):
            z_in = z + self.alpha[k] * self.action_token  # (B, N, D), broadcast token
            z = self.refiner(z_in)
            intermediates.append(z)
        return {"z_final": z, "intermediates": intermediates}
