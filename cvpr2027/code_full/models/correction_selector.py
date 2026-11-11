"""Small, capacity-matched selectors for a train-only learnability experiment.

Only ``observable_patch_features`` constructs inputs. Its signature deliberately
excludes GT, error correlations a and oracle weights. GT correlations enter the
supervised loss elsewhere, never this network. The risk parameterization and
quadratic objective are classical, not an innovation claim.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = tuple(
    [f"low_mean_{c}" for c in "rgb"]
    + [f"base_mean_{c}" for c in "rgb"]
    + [f"past_correction_mean_{c}" for c in "rgb"]
    + [f"future_correction_mean_{c}" for c in "rgb"]
    + ["low_rgb_variance", "base_rgb_variance", "gram_00", "gram_01", "gram_11",
       "past_valid_fraction", "future_valid_fraction"])
MODES = ("risk_full", "direct_mse", "direct_mse_normalized")
_RISK_SOLVER = None


def _patch_pool(video, patch):
    batch, channels, time, height, width = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
    pooled = F.avg_pool2d(frames, patch, patch)
    return pooled.reshape(batch, time, channels, height // patch, width // patch)


def observable_patch_features(low, base, corrections, valid_mask, patch=8):
    """Return raw features [B,T,19,h,w] and observable Gram [B,T,h,w,2,2].

    low/base [B,3,T,H,W], corrections [B,2,3,T,H,W], valid_mask [B,2,1,T,H,W].
    Corrections must already be masked, as in cache_correction_oracle.py. RGB
    variances are mean within-patch spatial variance of the three color planes.
    Gram is recomputed from observable corrections, not read from a GT bundle.
    Context is subsequently provided by exactly one 3x3 patch-grid convolution.
    """
    if not isinstance(patch, int) or isinstance(patch, bool) or patch < 1:
        raise ValueError("patch must be a positive integer")
    if not isinstance(low, torch.Tensor) or low.ndim != 5 or low.shape[1] != 3:
        raise ValueError("low must have shape [B,3,T,H,W]")
    b, c, t, height, width = low.shape
    if min(b, t, height, width) < 1 or height % patch or width % patch:
        raise ValueError("Positive dimensions and complete nonoverlapping patches are required")
    if (base.shape != low.shape or corrections.shape != (b, 2, c, t, height, width)
            or valid_mask.shape != (b, 2, 1, t, height, width)):
        raise ValueError("Observable tensor shapes disagree")
    values = (low, base, corrections, valid_mask)
    if any(value.device != low.device for value in values):
        raise ValueError("Observable tensors must share a device")
    if low.dtype != torch.float32 or base.dtype != low.dtype or corrections.dtype != low.dtype:
        raise TypeError("Observable RGB tensors must be float32")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Observable tensors must be finite")
    if bool(((valid_mask < 0) | (valid_mask > 1)).any()):
        raise ValueError("valid_mask must be in [0,1]")
    if bool((corrections * (1 - valid_mask.float())).abs().amax() > 1e-7):
        raise ValueError("Corrections must be zero wherever the donor is invalid")
    with torch.no_grad():
        low_mean, base_mean = _patch_pool(low, patch), _patch_pool(base, patch)
        d0, d1 = corrections[:, 0], corrections[:, 1]
        d0_mean, d1_mean = _patch_pool(d0, patch), _patch_pool(d1, patch)
        low_var = (_patch_pool(low.square(), patch) - low_mean.square()).clamp_min(0).mean(2, keepdim=True)
        base_var = (_patch_pool(base.square(), patch) - base_mean.square()).clamp_min(0).mean(2, keepdim=True)
        g00 = _patch_pool(d0.square(), patch).mean(2)
        g01 = _patch_pool(d0 * d1, patch).mean(2)
        g11 = _patch_pool(d1.square(), patch).mean(2)
        gram = torch.stack((g00, g01, g01, g11), -1).reshape(b, t, height // patch, width // patch, 2, 2)
        valid = [_patch_pool(valid_mask[:, j].float(), patch) for j in range(2)]
        features = torch.cat((low_mean, base_mean, d0_mean, d1_mean, low_var, base_var,
                              g00[:, :, None], g01[:, :, None], g11[:, :, None], *valid), dim=2)
    return features, gram


class TinyCorrectionSelector(nn.Module):
    """19 -> 32 -> 32 -> 2, shared architecture and zero head in all three modes.

    A 3x3 first convolution followed by 1x1 layers uses only a 3x3 patch
    neighborhood. No normalization layers, dropout, GT path or temporal state.
    Inputs must already be standardized with fit-scene-only statistics.
    """
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(len(FEATURE_NAMES), 32, 3, padding=1),
                                   nn.ReLU(), nn.Conv2d(32, 32, 1), nn.ReLU())
        self.head = nn.Conv2d(32, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, standardized_features):
        if standardized_features.ndim != 4 or standardized_features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Expected standardized frame features [N,19,h,w]")
        return self.head(self.trunk(standardized_features)).permute(0, 2, 3, 1)


def moment_denominator(gram, energy_floor):
    """Observable residual RMS, floored in RGB units; never invert energy."""
    if not isinstance(energy_floor, (float, int)) or not 0 < energy_floor < float("inf"):
        raise ValueError("energy_floor must be finite and positive")
    diagonal = torch.stack((gram[..., 0, 0], gram[..., 1, 1]), dim=-1)
    return diagonal.clamp_min(0).sqrt().clamp_min(energy_floor)


def normalized_moment_target(a, gram, target_scale, energy_floor):
    """TRAIN-ONLY target (a - G @ [1/3,1/3]) / residual_RMS / fit_RMS_scale."""
    baseline_a = gram.sum(dim=-1) / 3
    return (a - baseline_a) / moment_denominator(gram, energy_floor) / target_scale


def weights_from_head(head, gram, mode, target_scale, energy_floor):
    """Decode all modes to the SAME simplex with a common initial function.

    Direct mode adds a fixed zero none-logit. Risk mode predicts a and uses the
    exact FULL observable Gram. On singular Gram the minimizer need not be
    unique; an exactly zero head explicitly chooses the optimal [1/3,1/3]
    anchor, so all methods also start with identical weights in that case.
    """
    if mode in ("direct_mse", "direct_mse_normalized"):
        none = torch.zeros_like(head[..., :1])
        return torch.softmax(torch.cat((none, head), dim=-1), dim=-1)[..., 1:]
    if mode != "risk_full":
        raise ValueError(f"Unknown selector mode: {mode}")
    global _RISK_SOLVER
    if _RISK_SOLVER is None:
        path = Path(__file__).with_name("correction_risk.py")
        spec = importlib.util.spec_from_file_location("_selector_correction_risk", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RISK_SOLVER = module.solve_two_candidate_risk
    a_hat = gram.sum(dim=-1) / 3 + moment_denominator(gram, energy_floor) * (head * target_scale)
    weights, _ = _RISK_SOLVER(a_hat, gram)
    anchor = torch.full_like(weights, 1 / 3)
    return torch.where((head == 0).all(dim=-1, keepdim=True), anchor, weights)


def quadratic_mse(e0, a, gram, weights):
    """Actual unrounded RGB MSE per patch; GT statistics are loss-only inputs."""
    quadratic = (weights.unsqueeze(-1) * gram * weights.unsqueeze(-2)).sum(dim=(-1, -2))
    return e0 + quadratic - 2 * (a * weights).sum(dim=-1)
