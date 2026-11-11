"""Scale-aware wrapper around JEPADeltaAdapter (DP5).

Adds per-V-JEPA-layer learnable gain + confidence:
  delta_used_l = sigmoid(conf_logit_l) * gain_l * delta_pred_l

Initialization from Probe A evidence:
  layer 5 (idx 0): low confidence (sigmoid(-1)=0.27)
  layer 11 (idx 1): low confidence (0.27)
  layer 17 (idx 2): medium (0.5)
  layer 23 (idx 3): high (sigmoid(1)=0.73)
  gain: [0.5, 0.5, 1.0, 1.5] (boost layer 23, dampen shallow)

Spec: VJEPA_LLVE_CODE_REVIEW.md §DP5.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from .jepa_delta_adapter import JEPADeltaAdapter


class ScaleAwareDeltaAdapter(nn.Module):
    def __init__(self, base_adapter: JEPADeltaAdapter,
                 gain_init=(0.5, 0.5, 1.0, 1.5),
                 conf_logit_init=(-1.0, -1.0, 0.0, 1.0)):
        super().__init__()
        self.base = base_adapter
        n = base_adapter.n_layers
        self.gain = nn.Parameter(torch.tensor(list(gain_init[:n]), dtype=torch.float32))
        self.conf_logit = nn.Parameter(torch.tensor(list(conf_logit_init[:n]), dtype=torch.float32))

    @property
    def n_layers(self): return self.base.n_layers

    def predict_delta(self, z_L_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        raw = self.base.predict_delta(z_L_layers)
        out = []
        for i, d in enumerate(raw):
            scale = torch.sigmoid(self.conf_logit[i]) * self.gain[i]
            out.append(scale * d)
        return out

    def forward(self, z_L_layers: List[torch.Tensor]) -> List[torch.Tensor]:
        deltas = self.predict_delta(z_L_layers)
        return [z + d for z, d in zip(z_L_layers, deltas)]

    def trainable_parameters(self):
        # All trainable (base adapter + scale params)
        return self.parameters()

    def gain_confidence_summary(self) -> str:
        return " | ".join([
            f"L{i}: gain={self.gain[i].item():.3f} conf={torch.sigmoid(self.conf_logit[i]).item():.3f}"
            for i in range(self.n_layers)
        ])
