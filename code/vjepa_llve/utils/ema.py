"""Exponential moving average wrapper for trainable parameters.

Used in v6.D to stabilize val PSNR (the v6.0 val noise was ±2 dB which made
"best" criterion unreliable). EMA of decoder + DiT typically reduces val
variance by 2-3x while improving final PSNR by ~0.2 dB.

Usage:
    ema = ParameterEMA(model.trainable_parameters(), decay=0.999)
    ...
    optim.step()
    ema.update(model.trainable_parameters())
    ...
    # at eval time:
    ema.swap_in(model.trainable_parameters())
    metrics = run_eval(model)
    ema.swap_out(model.trainable_parameters())
"""
from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


class ParameterEMA:
    def __init__(self, parameters: Iterable[nn.Parameter], decay: float = 0.999) -> None:
        self.decay = decay
        params = [p for p in parameters if p.requires_grad]
        self._shadow: List[torch.Tensor] = [p.detach().clone() for p in params]
        self._backup: List[torch.Tensor] = []

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        params = [p for p in parameters if p.requires_grad]
        assert len(params) == len(self._shadow), (
            f"EMA param count mismatch: shadow={len(self._shadow)} vs given={len(params)}"
        )
        d = self.decay
        for s, p in zip(self._shadow, params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def swap_in(self, parameters: Iterable[nn.Parameter]) -> None:
        """Save live params and swap in EMA shadow."""
        params = [p for p in parameters if p.requires_grad]
        assert len(params) == len(self._shadow)
        self._backup = [p.detach().clone() for p in params]
        for p, s in zip(params, self._shadow):
            p.data.copy_(s.data)

    @torch.no_grad()
    def swap_out(self, parameters: Iterable[nn.Parameter]) -> None:
        params = [p for p in parameters if p.requires_grad]
        assert len(self._backup) == len(params), "swap_out without prior swap_in"
        for p, b in zip(params, self._backup):
            p.data.copy_(b.data)
        self._backup = []

    def state_dict(self):
        return {"decay": self.decay, "shadow": [s.cpu() for s in self._shadow]}

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        # Preserve the device of existing shadow (shadow has been .to(device)'d already
        # at construction time; loading should not silently move tensors to CPU).
        target_device = self._shadow[0].device if self._shadow else None
        self._shadow = [
            (s.to(target_device) if target_device is not None else s.clone())
            for s in state["shadow"]
        ]

    def to(self, device):
        self._shadow = [s.to(device) for s in self._shadow]
        return self
