"""Ablation switches shared by v35_stage4_ctrl.py and seer_eval_unified.py.

Every switch is applied after the Stage-3 weights are loaded, so a variant
starts from the same initialisation as the reference arm and only differs in
the disabled path. Bypassed modules are removed from the module tree, so the
saved state_dict simply lacks their keys.
"""
from __future__ import annotations

import torch
import torch.nn as nn

RANDOM_VJEPA_SEED = 1234
VARIANT_KEYS = ("no_vjepa", "random_vjepa", "disable_igab", "disable_lic", "disable_bottleneck")


class _SkipIGAB(nn.Module):
    def forward(self, x: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        return x


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def apply_pixel_branch_variant(pb: nn.Module, disable_igab: bool = False,
                               disable_lic: bool = False,
                               disable_bottleneck: bool = False) -> nn.Module:
    if disable_igab:
        for i in range(len(pb.dec_igab)):
            pb.dec_igab[i] = _SkipIGAB()
    if disable_lic:
        pb.use_illumination_cues = False
    if disable_bottleneck:
        pb.temporal_mixer = _Identity()
        pb.cfim_bottleneck = None
    return pb


@torch.no_grad()
def param_fingerprint(module: nn.Module) -> float:
    """Cheap checksum used to verify that train and eval rebuild the same random backbone."""
    return float(sum(p.double().sum() for p in list(module.parameters())[:8]))
