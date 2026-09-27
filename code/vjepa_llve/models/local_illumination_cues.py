"""Local illumination cues for v9.0 — replaces v8's global Illumination Prompt Bank.

Per reviewer §3.4 + F2.4 + G2:
    Global pooled V-JEPA feat → softmax routing was too coarse for low-light.
    Replace with LOCAL pixel-domain cues per decoder scale:
      - luminance (Y from RGB)
      - channel-max (max(R,G,B))
      - local mean (3x3 neighborhood average)
      - low-frequency luminance (Gaussian-blurred Y)
      - gradient magnitude (|∇LQ|)

These 5 maps are extracted from LQ pixels (no V-JEPA), downsampled to each
decoder scale, concatenated, projected to decoder dim, and ADDED as bias to
the decoder stage features (no learnable mixing weight in v9.0; just additive
per reviewer F4 spec).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, T, H, W) → (B, 1, T, H, W) Y = 0.299R + 0.587G + 0.114B."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def channel_max(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, T, H, W) → (B, 1, T, H, W) per-pixel max over channels."""
    return x.max(dim=1, keepdim=True).values


def local_mean(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """(B, C, T, H, W) → (B, C, T, H, W) k×k spatial mean (per frame)."""
    B, C, T, H, W = x.shape
    x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    pad = k // 2
    out = F.avg_pool2d(F.pad(x_bt, (pad, pad, pad, pad), mode="reflect"),
                       kernel_size=k, stride=1)
    return out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


def gaussian_blur_lf(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """(B, C, T, H, W) → (B, C, T, H, W) low-freq via separable Gaussian (per frame)."""
    B, C, T, H, W = x.shape
    k = max(3, int(2 * round(3 * sigma) + 1))  # kernel size
    if k % 2 == 0:
        k += 1
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g1d = g.view(1, 1, 1, k)
    pad = k // 2

    x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    # horizontal
    x_bt = F.conv2d(F.pad(x_bt, (pad, pad, 0, 0), mode="reflect"),
                    g1d.expand(C, 1, 1, k), groups=C)
    # vertical
    x_bt = F.conv2d(F.pad(x_bt, (0, 0, pad, pad), mode="reflect"),
                    g1d.transpose(2, 3).expand(C, 1, k, 1), groups=C)
    return x_bt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) → (B, 1, T, H, W) Sobel-like gradient magnitude (per frame), averaged over channels."""
    B, C, T, H, W = x.shape
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    x_bt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    # Apply per channel, then average
    gx = F.conv2d(F.pad(x_bt, (1, 1, 1, 1), mode="reflect"),
                  sobel_x.expand(C, 1, 3, 3), groups=C)
    gy = F.conv2d(F.pad(x_bt, (1, 1, 1, 1), mode="reflect"),
                  sobel_y.expand(C, 1, 3, 3), groups=C)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    mag = mag.mean(dim=1, keepdim=True)  # average over channels → (BT, 1, H, W)
    return mag.reshape(B, T, 1, H, W).permute(0, 2, 1, 3, 4).contiguous()


def extract_illumination_cues(lq: torch.Tensor) -> torch.Tensor:
    """(B, 3, T, H, W) [0,1] → (B, 5, T, H, W) cue stack.

    5 channels: [luminance, channel_max, local_mean_of_luminance, low_freq_luminance, grad_mag]
    """
    y = rgb_to_luminance(lq)                       # (B, 1, T, H, W)
    cmax = channel_max(lq)                          # (B, 1, T, H, W)
    lmean = local_mean(y, k=3)                      # (B, 1, T, H, W) — local mean of luminance
    ylf = gaussian_blur_lf(y, sigma=4.0)            # (B, 1, T, H, W) — low-freq luminance
    grad = gradient_magnitude(lq)                   # (B, 1, T, H, W) — average channel gradient mag
    return torch.cat([y, cmax, lmean, ylf, grad], dim=1)  # (B, 5, T, H, W)


class IlluminationCueInjection(nn.Module):
    """Inject local illumination cues into decoder stage features additively.

    For each decoder scale s with feature shape (BT, dim_s, H_s, W_s):
      1. Compute 5-channel cue stack from LQ at original resolution
      2. Downsample (avg pool) cues to match (H_s, W_s)
      3. Project 5 → dim_s via 1x1 conv + SiLU
      4. Add to decoder feature (additive injection, no learnable gate per F4 spec)

    Per scale: ~5*dim_s + dim_s parameters (small).
    """

    def __init__(self, decoder_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(5, decoder_dim, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=True),
        )
        # Init last conv to zero so injection starts as identity (decoder unchanged at init)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, decoder_feat: torch.Tensor, cues_5ch_full: torch.Tensor,
                target_T: int) -> torch.Tensor:
        """decoder_feat: (BT, dim, H, W); cues_5ch_full: (B, 5, T, H_full, W_full)."""
        BT, _, H, W = decoder_feat.shape
        B = BT // target_T
        # Downsample cues to (H, W)
        cues = cues_5ch_full
        if cues.shape[-1] != W or cues.shape[-2] != H:
            # 3D interpolate spatial only (T preserved)
            cues = F.interpolate(
                cues.permute(0, 2, 1, 3, 4).reshape(B * target_T, 5, cues.shape[-2], cues.shape[-1]),
                size=(H, W), mode="bilinear", align_corners=False,
            )
            # cues now (BT, 5, H, W)
        else:
            cues = cues.permute(0, 2, 1, 3, 4).reshape(B * target_T, 5, H, W)
        bias = self.proj(cues.to(decoder_feat.dtype))  # (BT, dim, H, W)
        return decoder_feat + bias
