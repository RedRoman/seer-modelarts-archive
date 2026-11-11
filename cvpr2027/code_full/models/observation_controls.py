"""Fixed-observation controls, separate from the frozen v1 candidate recipe.

No GT, risk label, learned parameter or data-dependent fit is accepted. The
same-frame control removes explicit neighbour values at this construction
stage; its fixed base frame can still contain all-clip restoration information.
These are controls, not new filtering principles or noise-independence claims.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F


# Load just the frozen helper, without executing models/__init__.py or loading
# restoration backbones. This also supports independent file-based CPU tests.
_spec = importlib.util.spec_from_file_location(
    "_observation_controls_frozen_reference", Path(__file__).with_name("observation_candidates.py"))
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)

CONTROL_MODES = ("raw_detail", "same_frame_detail", "mismatched_detail")


def observation_control_config(mode, clip_len, offset=2):
    """JSON-only recipe and explicit per-direction, per-frame value indices.

An out-of-clip original direction remains invalid. For the mismatch control,
only the opposite *value* index is clamped to a real frame, with every such
fallback recorded. It is never represented as a valid opposite-time match.
"""
    if mode not in CONTROL_MODES:
        raise ValueError(f"Unknown observation control: {mode}")
    recipe = _reference.candidate_config("raw_detail", offset)
    if (not isinstance(clip_len, int) or isinstance(clip_len, bool)
            or clip_len <= offset):
        raise ValueError("clip_len must be an integer greater than offset")
    directions = []
    for name, delta in (("past", -offset), ("future", offset)):
        original = [t + delta for t in range(clip_len)]
        requested = [t if mode == "same_frame_detail" else
                     t - delta if mode == "mismatched_detail" else t + delta
                     for t in range(clip_len)]
        actual = [min(max(t, 0), clip_len - 1) for t in requested]
        directions.append({
            "direction": name, "original_offset": delta,
            "original_source_requested": original,
            "original_source_in_clip": [0 <= t < clip_len for t in original],
            "value_source_requested": requested, "value_source_indices": actual,
            "value_index_clamped": [a != b for a, b in zip(requested, actual)],
            "value_source_is_target": [source == target for target, source in enumerate(actual)],
        })
    return {
        "schema_version": 1, "mode": mode, "clip_len": clip_len, "offset": offset,
        "base_recipe": copy.deepcopy(recipe),
        "direction_order": ["past", "future"], "directions": directions,
        "value_alignment": "none; target frame values" if mode == "same_frame_detail" else
                           "original direction's saved target-to-source flow, unchanged",
        "validity": "original direction's validity and reflect-LP9 support erosion; never relaxed",
        "mismatch_edge_policy": "opposite requested source index clamped to [0,T-1]; no wrapping",
        "gt_input": False, "target_domain_fit": False,
        "interpretation": "fixed-input candidate control; base_t may contain all-clip information; "
                          "no independence or novelty claim",
    }


@torch.no_grad()
def build_observation_controls(low, base, flows, masks, *, mode, offset=2):
    """Return corrections [B,2,3,T,H,W], masks [B,2,1,T,H,W], and recipe.

    Inputs are exactly those of the frozen ``build_observation_candidates``:
    low/base float32 RGB [B,3,T,H,W] in [0,1], target-to-source pixel flows
    [B,2,2,T,H,W], and binary masks [B,2,1,T,H,W]. Direction order is [-k,+k].
    All three modes have exactly the frozen raw-detail support mask. With L
    the spatial reflect box9 filter, q=clip(L(base)+gain*(low-L(low))), where
    gain=clip(L(base)/(L(low)+1e-4),.25,16). Each candidate preserves L(base_t).

    raw_detail: delegate to the frozen implementation, without recomputation.
    same_frame_detail: clip(L(base_t)+q_t-L(q_t)); no value warping or explicit
        neighbour-frame value access. Original flows still validate the masks.
    mismatched_detail: original direction j's flow samples q at the opposite
        temporal index, clamped to a real frame when missing. Its original
        temporal/geometric mask is retained. Subtract the warped q's L before
        adding target L(base_t), exactly as in the frozen detail construction.
    Corrections are candidate minus base, and exactly zero outside the mask.
    """
    if any(not isinstance(value, torch.Tensor) for value in (low, base, flows, masks)):
        raise ValueError("All observable inputs must be tensors")
    if low.ndim != 5 or low.shape[1] != 3:
        raise ValueError("low must have shape [B,3,T,H,W]")
    config = observation_control_config(mode, low.shape[2], offset)
    if mode == "raw_detail":
        correction, validity, recipe = _reference.build_observation_candidates(
            low, base, flows, masks, mode="raw_detail", offset=offset)
        config["base_recipe"] = recipe
        return correction, validity, config

    _reference._validate(low, base, flows, masks, offset)
    b, _, t, h, w = low.shape
    low_l, base_l = _reference._lowpass(low), _reference._lowpass(base)
    gain = (base_l / (low_l + _reference.GAIN_EPSILON)).clamp(
        _reference.GAIN_MIN, _reference.GAIN_MAX)
    source_values = (base_l + gain * (low - low_l)).clamp(0, 1)
    target = _reference._frames(base)
    times = torch.arange(t, device=low.device)
    yy, xx = torch.meshgrid(torch.arange(h, device=low.device),
                            torch.arange(w, device=low.device), indexing="ij")
    if mode == "same_frame_detail":
        same_candidate = (base_l + source_values - _reference._lowpass(source_values)).clamp(0, 1)

    result, validity = [], []
    for j, delta in enumerate((-offset, offset)):
        original_times = times + delta
        time_valid = (original_times >= 0) & (original_times < t)
        flow = _reference._frames(flows[:, j])
        x, y = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
        geometric = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))
        geometric = geometric.reshape(b, t, h, w) & time_valid[None, :, None, None]
        if bool((masks[:, j, 0].bool() & ~geometric).any()):
            raise ValueError("Mask marks an out-of-time or out-of-image flow sample valid")
        valid = _reference._support_mask(masks[:, j].bool())
        if mode == "same_frame_detail":
            candidate = same_candidate
        else:
            opposite_times = (times - delta).clamp(0, t - 1)
            source = _reference._frames(source_values.index_select(2, opposite_times))
            grid = torch.stack(((x + .5) * (2 / w) - 1,
                                (y + .5) * (2 / h) - 1), dim=-1)
            warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border",
                                   align_corners=False)
            warped_video = _reference._video(warped, b, t)
            candidate = (base_l + warped_video - _reference._lowpass(warped_video)).clamp(0, 1)
        correction = _reference._video(_reference._frames(candidate) - target, b, t) * valid
        result.append(correction)
        validity.append(valid)
    return torch.stack(result, 1), torch.stack(validity, 1), config
