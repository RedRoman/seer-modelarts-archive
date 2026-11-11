"""Observable donor-value controls for source-trained video corrections.

These are candidate construction controls, not new filtering principles.
All alignment uses saved target-to-source flow computed on restored RGB.
No GT, error statistic, oracle, learned head or target-domain fit is accepted.
Different donor indices do not imply statistically independent noise/evidence.
"""
from __future__ import annotations

import copy

import torch
from torch.nn import functional as F


CANDIDATE_MODES = ("restored", "restored_detail", "raw_detail")
LOWPASS_KERNEL = 9
GAIN_MIN, GAIN_MAX, GAIN_EPSILON = .25, 16., 1e-4


def candidate_config(mode, offset=2):
    if mode not in CANDIDATE_MODES:
        raise ValueError(f"Unknown candidate mode: {mode}")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 1:
        raise ValueError("offset must be a positive integer")
    return {"schema_version": 1, "mode": mode, "offset": offset,
            "lowpass": {"type": "box", "kernel": LOWPASS_KERNEL, "padding": "reflect",
                        "time_filtering": False},
            "photometric_gain": {"minimum": GAIN_MIN, "maximum": GAIN_MAX,
                                 "denominator_epsilon": GAIN_EPSILON,
                                 "fit": "none; observable per-source lowpass RGB ratio"},
            "source_values": "low-light RGB detail" if mode == "raw_detail" else "restored RGB",
            "candidate_clip": "source calibrated RGB and final detail candidate clamp to [0,1]",
            "detail_mask": "reflect-minpool over lowpass support; restored uses original valid mask",
            "alignment": "saved restored-RGB target-to-source flow; bilinear gather; align_corners=False",
            "interpretation": "Observable candidate controls; no independence or novelty claim"}


def _frames(video):
    b, c, t, h, w = video.shape
    return video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)


def _video(frames, b, t):
    _, c, h, w = frames.shape
    return frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _lowpass(video):
    b, _, t, _, _ = video.shape
    r = LOWPASS_KERNEL // 2
    values = F.avg_pool2d(F.pad(_frames(video), (r, r, r, r), mode="reflect"),
                          LOWPASS_KERNEL, stride=1)
    return _video(values, b, t)


def _support_mask(mask):
    b, _, t, _, _ = mask.shape
    r = LOWPASS_KERNEL // 2
    values = -F.max_pool2d(-F.pad(_frames(mask.float()), (r, r, r, r), mode="reflect"),
                            LOWPASS_KERNEL, stride=1)
    return _video(values, b, t) == 1


def _validate(low, base, flows, masks, offset):
    if not isinstance(low, torch.Tensor) or low.ndim != 5 or low.shape[1] != 3:
        raise ValueError("low must have shape [B,3,T,H,W]")
    b, _, t, h, w = low.shape
    if min(b, t, h, w) < 1 or t <= offset or min(h, w) <= LOWPASS_KERNEL // 2:
        raise ValueError("Need positive batches, T>offset and reflect-compatible spatial dimensions")
    if (base.shape != low.shape or flows.shape != (b, 2, 2, t, h, w)
            or masks.shape != (b, 2, 1, t, h, w)):
        raise ValueError("Observable video/flow/mask shapes disagree")
    if any(value.device != low.device for value in (base, flows, masks)):
        raise ValueError("All observable inputs must share a device")
    if any(value.dtype != torch.float32 for value in (low, base, flows)):
        raise TypeError("Observable videos and flows must be float32")
    if any(not bool(torch.isfinite(value).all()) for value in (low, base, flows, masks)):
        raise ValueError("Observable inputs must be finite")
    if any(bool(((value < 0) | (value > 1)).any()) for value in (low, base)):
        raise ValueError("Low and restored RGB must be in [0,1]")
    if bool(((masks != 0) & (masks != 1)).any()):
        raise ValueError("Candidate validity masks must be binary")


@torch.no_grad()
def build_observation_candidates(low, base, flows, masks, *, mode, offset=2):
    """Return (corrections, masks, JSON config) without accessing GT.

    low/base [B,3,T,H,W], flows [B,2,2,T,H,W] in target-to-source
    pixel units, masks [B,2,1,T,H,W]. Direction order is [-offset,+offset].
    Corrections have shape [B,2,3,T,H,W] and are exactly zero at invalid pixels.
    ``restored_detail`` controls for the highpass/lowpass construction while
    retaining restored source values. ``raw_detail`` substitutes calibrated
    source observations as detail values, with the same filter/masking rules.
    """
    config = candidate_config(mode, offset)
    _validate(low, base, flows, masks, offset)
    b, _, t, h, w = low.shape
    target = _frames(base)
    times = torch.arange(t, device=base.device)
    yy, xx = torch.meshgrid(torch.arange(h, device=base.device),
                            torch.arange(w, device=base.device), indexing="ij")
    if mode == "raw_detail":
        low_l, base_l = _lowpass(low), _lowpass(base)
        gain = (base_l / (low_l + GAIN_EPSILON)).clamp(GAIN_MIN, GAIN_MAX)
        source_values = (base_l + gain * (low - low_l)).clamp(0, 1)
    else:
        source_values = base
    target_low = _lowpass(base) if mode != "restored" else None
    result, validity = [], []
    for j, delta in enumerate((-offset, offset)):
        source_times = times + delta
        time_valid = (source_times >= 0) & (source_times < t)
        flow = _frames(flows[:, j])
        x, y = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
        grid = torch.stack(((x + .5) * (2 / w) - 1,
                            (y + .5) * (2 / h) - 1), dim=-1)
        geometric = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))
        geometric = geometric.reshape(b, t, h, w) & time_valid[None, :, None, None]
        given = masks[:, j, 0].bool()
        if bool((given & ~geometric).any()):
            raise ValueError("Mask marks an out-of-time or out-of-image flow sample valid")
        source = _frames(source_values.index_select(2, source_times.clamp(0, t - 1)))
        warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border",
                               align_corners=False)
        valid = masks[:, j].bool()
        if mode != "restored":
            warped_video = _video(warped, b, t)
            candidate = (target_low + warped_video - _lowpass(warped_video)).clamp(0, 1)
            warped = _frames(candidate)
            valid = _support_mask(valid)
        correction = _video(warped - target, b, t) * valid
        result.append(correction)
        validity.append(valid)
    return torch.stack(result, dim=1), torch.stack(validity, dim=1), copy.deepcopy(config)
