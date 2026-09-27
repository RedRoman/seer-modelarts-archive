"""V-JEPA token positional embedding for v10.

Adds learnable 3D positional embeddings (P_t + P_y + P_x) to V-JEPA tokens
BEFORE they are passed to pixel_branch (and hence to cross-attn K/V).

Identity-at-init guarantees (CH3 + CL1 fix):
    gamma_pos = 1.0   (was 0, broke gradient flow per CL1)
    P_t/P_y/P_x weights = 0
    → output is byte-identical to input at iteration 0 (gamma * 0 == 0).
    → dL/dP ∝ gamma_pos = 1 so M1 can learn.

Spec: V10_DESIGN_SPEC.md §3 / §9.R2.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class VJepaPosEmb(nn.Module):
    """3D learned positional embedding applied to V-JEPA tokens.

    The embedding is shared across all V-JEPA feature layers (same P_t/P_y/P_x
    is added to every layer in the e_lq_layers list). This is a v10 MVP
    simplification — per-layer embeddings can be added in v10.5 if needed.

    Args:
        token_grid: (T_v, H_v, W_v) — V-JEPA token grid. Fixed (8, 16, 16) for
            v10 MVP per CH1 (V-JEPA wrapper resizes inputs to 256x256).
        dim: V-JEPA token dim (= vjepa_dim, typically 1024 for ViT-L).
    """

    def __init__(self, token_grid: Tuple[int, int, int] = (8, 16, 16), dim: int = 1024) -> None:
        super().__init__()
        self.T_v, self.H_v, self.W_v = token_grid
        self.dim = dim

        # Per-axis learned embeddings (factorized: P_t + P_y + P_x).
        # ALL initialized to zero so iter-0 contribution is exactly zero
        # regardless of what gamma_pos does.
        self.P_t = nn.Embedding(self.T_v, dim)
        self.P_y = nn.Embedding(self.H_v, dim)
        self.P_x = nn.Embedding(self.W_v, dim)
        nn.init.zeros_(self.P_t.weight)
        nn.init.zeros_(self.P_y.weight)
        nn.init.zeros_(self.P_x.weight)

        # Learnable scale, init to ONE (CL1 fix). With P_*=0, output is still
        # identity at iter 0 (tokens + 1.0 * 0 == tokens). But dL/dP ∝ gamma_pos = 1
        # so M1 can actually learn (CL1 dead-gradient blocker).
        self.gamma_pos = nn.Parameter(torch.ones(1))

        # Pre-compute index tensors (registered as buffer for device transfer).
        t_idx = torch.arange(self.T_v).view(self.T_v, 1, 1).expand(
            self.T_v, self.H_v, self.W_v).reshape(-1)
        y_idx = torch.arange(self.H_v).view(1, self.H_v, 1).expand(
            self.T_v, self.H_v, self.W_v).reshape(-1)
        x_idx = torch.arange(self.W_v).view(1, 1, self.W_v).expand(
            self.T_v, self.H_v, self.W_v).reshape(-1)
        self.register_buffer("t_idx", t_idx, persistent=False)
        self.register_buffer("y_idx", y_idx, persistent=False)
        self.register_buffer("x_idx", x_idx, persistent=False)

    def _compute_pos(self) -> torch.Tensor:
        """Returns (N, D) positional embedding where N = T_v*H_v*W_v."""
        return self.P_t(self.t_idx) + self.P_y(self.y_idx) + self.P_x(self.x_idx)

    def forward(self, vjepa_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply pos-emb to each V-JEPA layer's tokens.

        Args:
            vjepa_layers: list of tensors, each shape (B, N, D) where
                N = T_v * H_v * W_v matches self.token_grid.

        Returns:
            list of same length and shapes, with pos-emb added.
        """
        if not vjepa_layers:
            return vjepa_layers
        # All layers share same pos-emb (added in same dtype/device as tokens).
        pos = self._compute_pos()  # (N, D), float32 from Embedding default
        out = []
        for f in vjepa_layers:
            B, N, D = f.shape
            assert N == self.t_idx.numel(), \
                f"vjepa layer N={N} doesn't match token_grid N={self.t_idx.numel()}"
            # Cast pos to f's dtype for autocast compatibility, then add scaled.
            f_out = f + self.gamma_pos * pos.to(f.dtype).unsqueeze(0)
            out.append(f_out)
        return out
