"""BiasFree LayerNorm — adapted from VLU-Net (CVPR 2025) / Restormer.

Key vs standard LayerNorm:
  * Subtracts MEAN before normalization (not just variance).
  * No bias in affine.
  * 2D variant operates on channels-first (B, C, H, W) tensors.

Why it helps low-light: dark regions have near-zero mean intensity; standard
LN's bias term can introduce drift in these regions. BiasFree LN's zero-bias
+ explicit demeaning is more stable for high-DR signal recovery.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiasFreeLayerNorm2d(nn.Module):
    """For (B, C, H, W) tensors, channel-wise normalization without bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1)


class BiasFreeLayerNorm1d(nn.Module):
    """For (B, N, C) tensors (token sequences), per-token normalization without bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight
