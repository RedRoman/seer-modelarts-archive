"""
VAE wrapper for v5.0 latent diffusion.

Wraps Stable Diffusion's VAE (sd-vae-ft-mse, 84M params, frozen) for
encoding/decoding video frames into a 4-channel 32x32 latent space.

Reconstruction quality (verified on SDSD-indoor):
    HQ frame: 42.84 dB PSNR
    LQ frame: 43.00 dB PSNR
Both well above the 35 dB threshold; latent diffusion is viable.

The VAE is per-frame: video (B, 3, T, H, W) is encoded by folding T into
batch, then unfolded back. No temporal compression in the VAE.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


SD_VAE_HF_REPO = "stabilityai/sd-vae-ft-mse"


class SDVAEWrapper(nn.Module):
    """Frozen Stable Diffusion VAE for video LLVE.

    Input space: pixels in [0, 1]
    Latent space: (B, 4, T, H/8, W/8), Gaussian-like (post-sampling).

    Encoder uses .latent_dist.sample() during training (stochastic) and
    .latent_dist.mean during inference (deterministic). Set
    `deterministic_encode=True` to always use mean.
    """

    LATENT_SCALE = 0.18215  # SD VAE scaling factor (latents in roughly N(0,1))

    def __init__(
        self,
        repo: str = SD_VAE_HF_REPO,
        dtype: torch.dtype = torch.float16,
        deterministic_encode: bool = False,
    ) -> None:
        super().__init__()
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from diffusers import AutoencoderKL  # type: ignore
        self.vae = AutoencoderKL.from_pretrained(repo)
        # Freeze + cast
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        # Keep the VAE in fp16 for memory but allow autocast contexts to
        # work alongside it. Storage dtype matters; computations during
        # autocast will respect autocast dtype.
        self.vae.to(dtype=dtype)
        self.dtype = dtype
        self.deterministic_encode = bool(deterministic_encode)
        self.latent_channels = int(self.vae.config.latent_channels)
        self.downsample = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def train(self, mode: bool = True):
        """Override: keep VAE in eval mode regardless of parent training flag."""
        super().train(mode)
        self.vae.eval()
        return self

    @staticmethod
    def _bcthw_to_btchw(x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def _btchw_to_bcthw(x_bt: torch.Tensor, T: int) -> torch.Tensor:
        BT, C, H, W = x_bt.shape
        B = BT // T
        return x_bt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def encode(self, video_01: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) in [0,1] -> (B, 4, T, H/8, W/8) latent."""
        B, C, T, H, W = video_01.shape
        x_bt = self._bcthw_to_btchw(video_01)
        # SD VAE expects [-1, 1] input
        x_bt = (x_bt.to(self.dtype).clamp(0, 1) - 0.5) * 2.0
        dist = self.vae.encode(x_bt).latent_dist
        z_bt = dist.mean if self.deterministic_encode else dist.sample()
        z_bt = z_bt * self.LATENT_SCALE
        return self._btchw_to_bcthw(z_bt, T)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 4, T, H/8, W/8) latent -> (B, 3, T, H, W) in [0,1]."""
        B, C, T, h, w = z.shape
        z_bt = self._bcthw_to_btchw(z)
        z_bt = z_bt.to(self.dtype) / self.LATENT_SCALE
        x_bt = self.vae.decode(z_bt).sample
        x_bt = (x_bt / 2.0 + 0.5).clamp(0, 1)
        return self._btchw_to_bcthw(x_bt, T)
