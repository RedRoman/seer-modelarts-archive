"""Bipolar Innovation-Enveloped Photometric-Invariant Trajectory Transport (EID-TT v2).

Cross-domain and cross-backbone video restoration operator acting strictly on
observable raw low-light frames X, frozen canonical-RGB enhancer predictions B,
and GT-free reciprocal optical flows. Zero ground-truth tensors or target-domain
statistics ever enter feature extraction, envelope estimation, conductance
prediction, or trajectory transport.

Mathematical guarantees:
1. Exact anchor identity when trajectory conductances and sensor gate vanish.
2. Exact Null-Flicker Boundary Condition: when trajectory innovations arise from
   monotonic scene motion or lighting velocity without uncorrelated enhancer
   flicker (overline{C} <= 0), the unbiased lag-1 de-attenuated bipolar envelope
   Omega^{bip} vanishes identically (0.0), preventing cross-backbone over-smoothing.
3. Exact Geometric Occlusion Suppression: all directional conductances scale
   multiplicatively with reciprocal forward-backward cycle confidence r^{fb} * m.
4. Exact independent two-parameter Lie-group scaling invariance/equivariance under
   (X, B) -> (beta * X, alpha * B) for all alpha, beta > 0 prior to [0, 1] clamp.
5. Zero static exposure bias by Markov conservation on two-band log-radiometry,
   paired with phase-aware forward-backward back-projection detail transport.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

OFFSETS: Tuple[int, ...] = (-2, -1, 1, 2)
REVERSE_INDEX: Tuple[int, ...] = (3, 2, 1, 0)
MODES: Tuple[str, ...] = (
    "eid_full",
    "ablate_no_bipolar_env",
    "ablate_no_rad",
    "ablate_no_deatt_raw",
    "ablate_no_sensor_inv",
)
LOSS_MODES: Tuple[str, ...] = ("decoupled", "raw_gt")

FEATURE_NAMES: Tuple[str, ...] = (
    "inv_mid_r",
    "inv_mid_g",
    "inv_mid_b",
    "inv_mid_l1",
    "inv_coarse_r",
    "inv_coarse_g",
    "inv_coarse_b",
    "enh_mid_l1",
    "sens_mid_l1",
    "detail_rel_diff",
    "detail_cos_corr",
    "sens_enh_snr_corr",
    "fb_cycle_conf",
    "subpixel_phase",
    "valid_fraction",
    "temporal_dist",
    "bipolar_omega_coarse",
    "bipolar_omega_mid",
    "bipolar_omega_detail",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    _require(kernel_size >= 3 and kernel_size % 2 == 1 and sigma > 0.0, "Invalid Gaussian parameters")
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size // 2)
    weights = torch.exp(-0.5 * (coords / sigma).square())
    return weights / weights.sum()


def separable_gaussian_2d(frames: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply exact constant-preserving separable reflective Gaussian blur to [N, C, H, W]."""
    _require(frames.ndim == 4, f"Expected 4D tensor, got shape {tuple(frames.shape)}")
    _, c, h, w = frames.shape
    pad = kernel_size // 2
    _require(h > pad and w > pad, f"Spatial size {(h, w)} smaller than reflect pad {pad}")
    k1d = gaussian_kernel_1d(kernel_size, sigma, frames.device, frames.dtype)
    kh = k1d.view(1, 1, 1, kernel_size).expand(c, 1, 1, kernel_size).contiguous()
    kv = k1d.view(1, 1, kernel_size, 1).expand(c, 1, kernel_size, 1).contiguous()
    out = F.pad(frames, (pad, pad, 0, 0), mode="reflect")
    out = F.conv2d(out, kh, groups=c)
    out = F.pad(out, (0, 0, pad, pad), mode="reflect")
    out = F.conv2d(out, kv, groups=c)
    return out


def separable_gaussian_video(video: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Apply spatial separable Gaussian blur framewise to [B, C, T, H, W]."""
    _require(video.ndim == 5, f"Expected 5D video [B,C,T,H,W], got {tuple(video.shape)}")
    b, c, t, h, w = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    blurred = separable_gaussian_2d(flat, kernel_size, sigma)
    return blurred.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def box_smooth_2d(frames: Tensor, kernel_size: int = 33) -> Tensor:
    """Two-pass reflective box smoothing on [N, C, H, W]."""
    _require(kernel_size % 2 == 1 and kernel_size >= 3, "Box kernel must be odd >= 3")
    pad = kernel_size // 2
    out = F.avg_pool2d(F.pad(frames, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    out = F.avg_pool2d(F.pad(out, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    return out


def warp_frames_2d(source: Tensor, flow_target_to_source: Tensor) -> Tuple[Tensor, Tensor]:
    """Sample source [N, C, H, W] at (x, y) + flow(x, y) in pixel coordinates."""
    _require(source.ndim == 4 and flow_target_to_source.ndim == 4, "Expected 4D tensors for warp")
    n, _, h, w = source.shape
    _require(flow_target_to_source.shape == (n, 2, h, w), f"Unexpected flow shape {tuple(flow_target_to_source.shape)}")
    ys = torch.arange(h, device=source.device, dtype=source.dtype)
    xs = torch.arange(w, device=source.device, dtype=source.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sample_x = grid_x.unsqueeze(0) + flow_target_to_source[:, 0]
    sample_y = grid_y.unsqueeze(0) + flow_target_to_source[:, 1]
    in_bounds = (
        (sample_x >= 0.0)
        & (sample_x <= float(w - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(h - 1))
    ).unsqueeze(1).to(source.dtype)
    norm_x = (2.0 * sample_x + 1.0) / float(w) - 1.0
    norm_y = (2.0 * sample_y + 1.0) / float(h) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped, in_bounds


def compute_cycle_geometry(
    flows: Tensor,
    offsets: Sequence[int] = OFFSETS,
    reverse_index: Sequence[int] = REVERSE_INDEX,
    fb_sigma_px: float = 1.5,
) -> Dict[str, Tensor]:
    """Derive exact in-bounds mask, forward-backward cycle error/confidence, and subpixel phase."""
    _require(flows.ndim == 6 and flows.shape[1] == len(offsets) and flows.shape[2] == 2, "Invalid flows shape")
    b, k_count, _, t, h, w = flows.shape
    times = torch.arange(t, device=flows.device)

    valid_list = []
    err_list = []
    conf_list = []
    phase_list = []

    for k in range(k_count):
        delta = int(offsets[k])
        rev_k = int(reverse_index[k])
        src_times = times + delta
        time_ok = ((src_times >= 0) & (src_times < t)).to(flows.dtype)
        clamped_src = src_times.clamp(0, t - 1)

        f_fwd = flows[:, k].permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
        f_rev_src = flows[:, rev_k, :, clamped_src].permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)

        f_rev_warped, spatial_ok = warp_frames_2d(f_rev_src, f_fwd)
        valid = spatial_ok * time_ok.repeat(b).view(b * t, 1, 1, 1)

        cycle_vec = f_fwd + f_rev_warped
        fb_err = torch.linalg.norm(cycle_vec, dim=1, keepdim=True) * valid
        fb_conf = torch.exp(-(fb_err / fb_sigma_px).square()) * valid

        fx = f_fwd[:, 0:1] - torch.floor(f_fwd[:, 0:1])
        fy = f_fwd[:, 1:2] - torch.floor(f_fwd[:, 1:2])
        phase = (2.0 * fx * (1.0 - fx) + 2.0 * fy * (1.0 - fy)).clamp(0.0, 1.0) * valid

        valid_list.append(valid.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4))
        err_list.append(fb_err.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4))
        conf_list.append(fb_conf.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4))
        phase_list.append(phase.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4))

    return {
        "valid_mask": torch.stack(valid_list, dim=1).contiguous(),
        "fb_error": torch.stack(err_list, dim=1).contiguous(),
        "fb_conf": torch.stack(conf_list, dim=1).contiguous(),
        "subpixel_phase": torch.stack(phase_list, dim=1).contiguous(),
    }


@torch.no_grad()
def compute_reciprocal_raft_flows(
    video_rgb: Tensor,
    raft: nn.Module,
    offsets: Sequence[int] = OFFSETS,
    flow_iters: int = 12,
    pair_batch: int = 16,
) -> Tensor:
    """Compute GT-free target-to-source RAFT flows [B, K, 2, T, H, W] on canonical RGB video."""
    _require(video_rgb.ndim == 5 and video_rgb.shape[1] == 3, "Expected canonical RGB video [B, 3, T, H, W]")
    b, _, t, h, w = video_rgb.shape
    _require(t >= 3 and min(h, w) >= 16, "Clip too short or small for reciprocal flows")
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    device = next(raft.parameters()).device

    all_flows = video_rgb.new_zeros((b, len(offsets), 2, t, h, w), dtype=torch.float32, device="cpu")
    for bi in range(b):
        frames = video_rgb[bi].permute(1, 0, 2, 3).contiguous().to(device, dtype=torch.float32)
        norm_frames = frames * 2.0 - 1.0
        if pad_h or pad_w:
            norm_frames = F.pad(norm_frames, (0, pad_w, 0, pad_h), mode="reflect")
        for k, delta in enumerate(offsets):
            times = torch.arange(t, device=device)
            src_times = times + delta
            valid = (src_times >= 0) & (src_times < t)
            tgt_idx = times[valid]
            src_idx = src_times[valid]
            if len(tgt_idx) == 0:
                continue
            out_dir = torch.zeros((t, 2, h, w), device=device, dtype=torch.float32)
            for start in range(0, len(tgt_idx), pair_batch):
                ti = tgt_idx[start:start + pair_batch]
                si = src_idx[start:start + pair_batch]
                pred_flow = raft(
                    norm_frames.index_select(0, ti),
                    norm_frames.index_select(0, si),
                    num_flow_updates=flow_iters,
                )[-1]
                out_dir.index_copy_(0, ti, pred_flow[:, :, :h, :w].float())
            all_flows[bi, k] = out_dir.permute(1, 0, 2, 3).cpu()
    return all_flows


def warp_video_offset(video: Tensor, flow_dir: Tensor, delta: int) -> Tensor:
    """Warp donor frame t + delta onto frame t's grid for [B, C, T, H, W]."""
    b, c, t, h, w = video.shape
    times = torch.arange(t, device=video.device)
    src_times = (times + delta).clamp(0, t - 1)
    src = video.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    fl = flow_dir.permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
    warped, _ = warp_frames_2d(src, fl)
    return warped.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def deattenuated_warp_offset(
    detail: Tensor,
    flow_fwd_dir: Tensor,
    flow_rev_dir: Tensor,
    fb_conf_rev_dir: Tensor,
    delta: int,
    beta_bp: Tensor,
) -> Tensor:
    """Phase-aware forward-backward back-projection warp compensating subpixel blur."""
    b, c, t, h, w = detail.shape
    times = torch.arange(t, device=detail.device)
    src_times = (times + delta).clamp(0, t - 1)

    src_h = detail.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    fl_fwd = flow_fwd_dir.permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
    w_t_from_s, _ = warp_frames_2d(src_h, fl_fwd)

    fl_rev_at_s = flow_rev_dir.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
    conf_rev_at_s = fb_conf_rev_dir.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    roundtrip_s, _ = warp_frames_2d(w_t_from_s, fl_rev_at_s)

    boosted_s = src_h + beta_bp * conf_rev_at_s * (src_h - roundtrip_s)
    deatt, _ = warp_frames_2d(boosted_s, fl_fwd)
    return deatt.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _pool_4d(video: Tensor, patch: int) -> Tensor:
    """Average-pool [B, C, T, H, W] spatially by patch -> [B, T, C, H//patch, W//patch]."""
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return F.avg_pool2d(frames, patch, patch).reshape(b, t, c, h // patch, w // patch)


@torch.no_grad()
def compute_bipolar_envelope(
    band: Tensor,
    flows: Tensor,
    r_dir: Tensor,
    fb_conf: Tensor,
    tau: float = 0.030,
    beta_cov: float = 0.80,
    spatial_k: int = 17,
    spatial_sigma: float = 4.5,
) -> Tensor:
    """Compute unbiased lag-1 de-attenuated Bipolar Trajectory Innovation Envelope [B, 1, T, H, W].

    Separates uncorrelated enhancer flicker (positive <d_{t,-1}, d_{t,+1}>) from
    monotonic scene motion or lighting velocity (non-positive <d_{t,-1}, d_{t,+1}>)
    by pooling unclamped covariance spatio-temporally prior to ReLU.
    """
    _, _, t, _, _ = band.shape
    beta_t = band.new_tensor(beta_cov)
    # OFFSETS[1] = -1, OFFSETS[2] = +1 (reciprocal indices 2 and 1)
    dn = deattenuated_warp_offset(band, flows[:, 1], flows[:, 2], fb_conf[:, 2], -1, beta_t) - band
    dp = deattenuated_warp_offset(band, flows[:, 2], flows[:, 1], fb_conf[:, 1], +1, beta_t) - band

    pv = r_dir[:, 1] * r_dir[:, 2]  # [B, 1, T, H, W]
    cov_num = (dn * dp).mean(dim=1, keepdim=True) * pv
    en_num = 0.5 * (dn.square() + dp.square()).mean(dim=1, keepdim=True) * pv

    cov_sp = separable_gaussian_video(cov_num, spatial_k, spatial_sigma)
    en_sp = separable_gaussian_video(en_num, spatial_k, spatial_sigma)
    w_sp = separable_gaussian_video(pv, spatial_k, spatial_sigma)

    t_idx = torch.arange(t, device=band.device)
    cov_st = torch.zeros_like(cov_sp)
    en_st = torch.zeros_like(en_sp)
    w_st = torch.zeros_like(w_sp)
    for dt in (-2, -1, 0, 1, 2):
        idx = (t_idx + dt).clamp(0, t - 1)
        cov_st = cov_st + cov_sp.index_select(2, idx)
        en_st = en_st + en_sp.index_select(2, idx)
        w_st = w_st + w_sp.index_select(2, idx)

    cov_clip = cov_num.sum(dim=(2, 3, 4), keepdim=True) / pv.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)
    en_clip = en_num.sum(dim=(2, 3, 4), keepdim=True) / pv.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)

    c_mean = torch.where(w_st > 0.15, cov_st / w_st.clamp_min(1e-4), cov_clip)
    e_mean = torch.where(w_st > 0.15, en_st / w_st.clamp_min(1e-4), en_clip)
    c_pos = F.relu(c_mean)
    omega = (2.0 * c_pos / (e_mean + c_pos + (tau ** 2))).clamp(0.0, 0.90)
    return omega.contiguous()


def extract_invariant_features(
    low: Tensor,
    base: Tensor,
    flows: Tensor,
    geom: Dict[str, Tensor],
    patch: int = 4,
    k_mid: int = 7,
    sigma_mid: float = 1.6,
    k_coarse: int = 25,
    sigma_coarse: float = 6.0,
    eps_rel: float = 1e-2,
    eps_det: float = 1e-2,
    tau_rad: float = 0.030,
    tau_det: float = 0.035,
    use_sensor_invariant: bool = True,
    use_bipolar_envelope: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Build 19 strictly dimensionless, photometric-scaling-invariant donor features."""
    _require(low.shape == base.shape and low.ndim == 5 and low.shape[1] == 3, "Invalid low/base shapes")
    _, _, _, h, w = base.shape
    _require(h % patch == 0 and w % patch == 0, "Spatial dims must be divisible by patch")

    mu_b = base.detach().mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    mu_x = low.detach().mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)

    b_mid = separable_gaussian_video(base, k_mid, sigma_mid)
    b_coarse = separable_gaussian_video(base, k_coarse, sigma_coarse)
    x_mid = separable_gaussian_video(low, k_mid, sigma_mid)
    x_coarse = separable_gaussian_video(low, k_coarse, sigma_coarse)

    u1 = torch.log(b_mid + eps_rel * mu_b)
    u2 = torch.log(b_coarse + eps_rel * mu_b)
    u_coarse = u2
    u_mid = u1 - u2

    v1 = torch.log(x_mid + eps_rel * mu_x)
    v2 = torch.log(x_coarse + eps_rel * mu_x)
    v_coarse = v2
    v_mid = v1 - v2

    h_b0 = base - b_mid
    h_x0 = low - x_mid

    with torch.no_grad():
        valid_mask = geom["valid_mask"]
        fb_conf = geom["fb_conf"]
        subpixel_phase = geom["subpixel_phase"]
        r_dir = fb_conf * valid_mask

        hb_norm = h_b0 / mu_b
        hx_norm = h_x0 / mu_x

        if use_bipolar_envelope:
            omega_coarse = compute_bipolar_envelope(u_coarse, flows, r_dir, fb_conf, tau=tau_rad)
            omega_mid = compute_bipolar_envelope(u_mid, flows, r_dir, fb_conf, tau=tau_rad)
            omega_detail = compute_bipolar_envelope(hb_norm, flows, r_dir, fb_conf, tau=tau_det)
            om_c_pool = _pool_4d(omega_coarse, patch)
            om_m_pool = _pool_4d(omega_mid, patch)
            om_h_pool = _pool_4d(omega_detail, patch)
        else:
            omega_coarse = torch.ones_like(u_coarse[:, :1])
            omega_mid = torch.ones_like(u_mid[:, :1])
            omega_detail = torch.ones_like(hb_norm[:, :1])
            om_c_pool = torch.zeros_like(_pool_4d(omega_coarse, patch))
            om_m_pool = om_c_pool
            om_h_pool = om_c_pool

        se_dot = _pool_4d((hb_norm * hx_norm).mean(dim=1, keepdim=True), patch)
        se_eb = _pool_4d(hb_norm.square().mean(dim=1, keepdim=True), patch)
        se_ex = _pool_4d(hx_norm.square().mean(dim=1, keepdim=True), patch)
        sens_enh_corr = (se_dot / torch.sqrt(se_eb * se_ex + (eps_det ** 4))).clamp(-1.0, 1.0)

        donor_feats = []
        for k, delta in enumerate(OFFSETS):
            m = valid_mask[:, k]
            fl = flows[:, k]

            du_mid = (warp_video_offset(u_mid, fl, delta) - u_mid) * m
            du_coarse = (warp_video_offset(u_coarse, fl, delta) - u_coarse) * m
            dv_mid = (warp_video_offset(v_mid, fl, delta) - v_mid) * m
            dv_coarse = (warp_video_offset(v_coarse, fl, delta) - v_coarse) * m

            if use_sensor_invariant:
                inv_mid = (du_mid - dv_mid).abs()
                inv_coarse = (du_coarse - dv_coarse).abs()
                sens_l1 = dv_mid.abs().mean(dim=1, keepdim=True)
            else:
                inv_mid = du_mid.abs()
                inv_coarse = du_coarse.abs()
                sens_l1 = torch.zeros_like(du_mid[:, :1])

            enh_l1 = du_mid.abs().mean(dim=1, keepdim=True)
            inv_mid_l1 = inv_mid.mean(dim=1, keepdim=True)

            hb_warp = warp_video_offset(hb_norm, fl, delta)
            det_num = _pool_4d(((hb_warp - hb_norm).abs() * m).mean(dim=1, keepdim=True), patch)
            det_den = _pool_4d(((hb_warp.abs() + hb_norm.abs()) * m).mean(dim=1, keepdim=True), patch) + eps_det
            detail_rel_diff = (det_num / det_den).clamp(0.0, 1.0)

            dc_dot = _pool_4d(((hb_norm * hb_warp) * m).mean(dim=1, keepdim=True), patch)
            dc_e1 = _pool_4d((hb_norm.square() * m).mean(dim=1, keepdim=True), patch)
            dc_e2 = _pool_4d((hb_warp.square() * m).mean(dim=1, keepdim=True), patch)
            detail_cos_corr = (dc_dot / torch.sqrt(dc_e1 * dc_e2 + (eps_det ** 4))).clamp(-1.0, 1.0)

            se_corr_masked = sens_enh_corr if use_sensor_invariant else torch.zeros_like(sens_enh_corr)
            tdist = torch.full_like(detail_rel_diff, abs(delta) / 2.0)

            feat_k = torch.cat(
                (
                    _pool_4d(inv_mid, patch),
                    _pool_4d(inv_mid_l1, patch),
                    _pool_4d(inv_coarse, patch),
                    _pool_4d(enh_l1, patch),
                    _pool_4d(sens_l1, patch),
                    detail_rel_diff,
                    detail_cos_corr,
                    se_corr_masked,
                    _pool_4d(fb_conf[:, k], patch),
                    _pool_4d(subpixel_phase[:, k], patch),
                    _pool_4d(m, patch),
                    tdist,
                    om_c_pool,
                    om_m_pool,
                    om_h_pool,
                ),
                dim=2,
            )
            donor_feats.append(feat_k)

        features = torch.stack(donor_feats, dim=2).contiguous()

    cache = {
        "mu_b": mu_b,
        "mu_x": mu_x,
        "u_coarse": u_coarse,
        "u_mid": u_mid,
        "h_x0": h_x0,
        "omega_coarse": omega_coarse,
        "omega_mid": omega_mid,
        "omega_detail": omega_detail,
    }
    return features, cache


class EIDTrajectoryTransport(nn.Module):
    """Bipolar Innovation-Enveloped Photometric-Invariant Trajectory Transport (v2)."""

    def __init__(
        self,
        mode: str = "eid_full",
        patch: int = 4,
        hidden: int = 24,
        k_mid: int = 7,
        sigma_mid: float = 1.6,
        k_coarse: int = 25,
        sigma_coarse: float = 6.0,
        tau_rad: float = 0.030,
        tau_det: float = 0.035,
        init_conductance_bias: float = -1.8,
        init_detail_bias: float = -2.0,
        init_sensor_bias: float = -3.2,
        max_sensor_gain: float = 0.25,
    ) -> None:
        super().__init__()
        _require(mode in MODES, f"Unsupported mode {mode!r}; expected one of {MODES}")
        _require(patch in (2, 4, 8) and hidden >= 8, "Invalid module hyperparameters")
        self.mode = mode
        self.patch = patch
        self.hidden = hidden
        self.k_mid = k_mid
        self.sigma_mid = sigma_mid
        self.k_coarse = k_coarse
        self.sigma_coarse = sigma_coarse
        self.tau_rad = float(tau_rad)
        self.tau_det = float(tau_det)
        self.init_conductance_bias = float(init_conductance_bias)
        self.init_detail_bias = float(init_detail_bias)
        self.init_sensor_bias = float(init_sensor_bias)
        self.max_sensor_gain = float(max_sensor_gain)

        in_ch = len(FEATURE_NAMES)
        self.donor_stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.consensus_head = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        self.sensor_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )
        self.beta_bp_logit = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

        nn.init.zeros_(self.consensus_head[-1].weight)
        nn.init.zeros_(self.consensus_head[-1].bias)
        nn.init.zeros_(self.sensor_head[-1].weight)
        nn.init.zeros_(self.sensor_head[-1].bias)

    def get_config(self) -> Dict[str, object]:
        return {
            "schema_version": 2,
            "mode": self.mode,
            "patch": self.patch,
            "hidden": self.hidden,
            "k_mid": self.k_mid,
            "sigma_mid": self.sigma_mid,
            "k_coarse": self.k_coarse,
            "sigma_coarse": self.sigma_coarse,
            "tau_rad": self.tau_rad,
            "tau_det": self.tau_det,
            "offsets": list(OFFSETS),
            "feature_names": list(FEATURE_NAMES),
            "parameters": sum(p.numel() for p in self.parameters()),
        }

    @staticmethod
    def _anchored_simplex(logits: Tensor, prior_bias: float) -> Tensor:
        """Map [B, K, 1, T, H, W] logits to strictly contractive conductances sum_k c_k < 1."""
        shifted = logits + prior_bias
        zero_anchor = torch.zeros_like(shifted[:, :1])
        probs = torch.softmax(torch.cat((zero_anchor, shifted), dim=1), dim=1)
        return probs[:, 1:]

    @torch.no_grad()
    def prepare_static_inputs(
        self,
        low: Tensor,
        base: Tensor,
        flows: Tensor,
        geom: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Tensor, Dict[str, Tensor]]:
        """Precompute deterministic GT-free geometry & features for fixed clip caching."""
        if geom is None:
            geom = compute_cycle_geometry(flows)
        use_sensor_inv = self.mode != "ablate_no_sensor_inv"
        use_bipolar_env = self.mode != "ablate_no_bipolar_env"
        features, cache = extract_invariant_features(
            low=low,
            base=base,
            flows=flows,
            geom=geom,
            patch=self.patch,
            k_mid=self.k_mid,
            sigma_mid=self.sigma_mid,
            k_coarse=self.k_coarse,
            sigma_coarse=self.sigma_coarse,
            tau_rad=self.tau_rad,
            tau_det=self.tau_det,
            use_sensor_invariant=use_sensor_inv,
            use_bipolar_envelope=use_bipolar_env,
        )
        return geom, features, cache

    def predict_conductances(
        self,
        features: Tensor,
        out_hw: Tuple[int, int],
        zero_identity_override: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b, t, k_count, c_in, hp, wp = features.shape
        h, w = out_hw
        if zero_identity_override:
            z_dir = features.new_zeros((b, k_count, 1, t, h, w))
            z_sens = features.new_zeros((b, 1, t, h, w))
            return z_dir, z_dir, z_dir, z_sens

        flat_in = features.reshape(b * t * k_count, c_in, hp, wp)
        emb = self.donor_stem(flat_in).reshape(b * t, k_count, self.hidden, hp, wp)
        consensus = emb.mean(dim=1, keepdim=True).expand_as(emb)
        cat_emb = torch.cat((emb, consensus), dim=2).reshape(b * t * k_count, self.hidden * 2, hp, wp)

        raw_dir = self.consensus_head(cat_emb)
        up_dir = F.interpolate(raw_dir, size=(h, w), mode="bilinear", align_corners=False)
        up_dir = up_dir.reshape(b, t, k_count, 3, h, w).permute(0, 2, 3, 1, 4, 5).contiguous()

        c_coarse = self._anchored_simplex(up_dir[:, :, 0:1], self.init_conductance_bias)
        c_mid = self._anchored_simplex(up_dir[:, :, 1:2], self.init_conductance_bias)
        w_det = self._anchored_simplex(up_dir[:, :, 2:3], self.init_detail_bias)

        mean_emb = emb.mean(dim=1)
        raw_sens = self.sensor_head(mean_emb)
        up_sens = F.interpolate(raw_sens, size=(h, w), mode="bilinear", align_corners=False)
        g_sens = self.max_sensor_gain * torch.sigmoid(up_sens + self.init_sensor_bias)
        g_sens = g_sens.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()

        if self.mode == "ablate_no_rad":
            c_coarse = torch.zeros_like(c_coarse)
            c_mid = torch.zeros_like(c_mid)
        if self.mode == "ablate_no_deatt_raw":
            g_sens = torch.zeros_like(g_sens)

        return c_coarse, c_mid, w_det, g_sens

    def forward(
        self,
        low: Tensor,
        base: Tensor,
        flows: Tensor,
        geom: Optional[Dict[str, Tensor]] = None,
        static_inputs: Optional[Tuple[Tensor, Dict[str, Tensor]]] = None,
        clamp_output: bool = True,
        zero_identity_override: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, object]:
        """GT-free forward pass. Only observable (low, base, flows) enter."""
        _require(low.ndim == 5 and base.shape == low.shape and low.shape[1] == 3, "Expected [B,3,T,H,W] low/base")
        b, _, t, h, w = base.shape
        _require(flows.shape == (b, len(OFFSETS), 2, t, h, w), "Expected reciprocal flows [B,4,2,T,H,W]")
        _require(low.device == base.device == flows.device, "All inputs must share a device")
        _require(low.dtype == base.dtype == flows.dtype == torch.float32, "float32 tensors required")

        if static_inputs is None:
            geom, features, cache = self.prepare_static_inputs(low, base, flows, geom=geom)
        else:
            _require(geom is not None, "geom required when static_inputs is provided")
            features, cache = static_inputs

        valid_mask = geom["valid_mask"]
        fb_conf = geom["fb_conf"]
        r_dir = fb_conf * valid_mask

        c_coarse, c_mid, w_det, g_sens = self.predict_conductances(
            features, (h, w), zero_identity_override=zero_identity_override
        )

        omega_coarse = cache["omega_coarse"]
        omega_mid = cache["omega_mid"]
        omega_detail = cache["omega_detail"]

        # Doubly-enveloped conductances: Bipolar Null-Flicker Envelope * Reciprocal Cycle Confidence
        eff_coarse = c_coarse * r_dir * omega_coarse.unsqueeze(1)
        eff_mid = c_mid * r_dir * omega_mid.unsqueeze(1)
        eff_det = w_det * r_dir * omega_detail.unsqueeze(1)
        eff_sens = g_sens * omega_detail

        beta_bp = (
            torch.zeros((), device=base.device, dtype=base.dtype)
            if (self.mode == "ablate_no_deatt_raw" or zero_identity_override)
            else torch.sigmoid(self.beta_bp_logit)
        )

        # Pillar 1: Conserved log-radiometric trajectory transport (coarse + de-attenuated mid)
        u_coarse_0 = cache["u_coarse"]
        u_mid_0 = cache["u_mid"]
        upd_c = torch.zeros_like(u_coarse_0)
        upd_m = torch.zeros_like(u_mid_0)

        if self.mode != "ablate_no_rad" and not zero_identity_override:
            for k, delta in enumerate(OFFSETS):
                rev_k = REVERSE_INDEX[k]
                wc = warp_video_offset(u_coarse_0, flows[:, k], delta)
                if self.mode == "ablate_no_deatt_raw":
                    wm = warp_video_offset(u_mid_0, flows[:, k], delta)
                else:
                    wm = deattenuated_warp_offset(
                        detail=u_mid_0,
                        flow_fwd_dir=flows[:, k],
                        flow_rev_dir=flows[:, rev_k],
                        fb_conf_rev_dir=fb_conf[:, rev_k],
                        delta=delta,
                        beta_bp=beta_bp,
                    )
                upd_c = upd_c + eff_coarse[:, k] * (wc - u_coarse_0)
                upd_m = upd_m + eff_mid[:, k] * (wm - u_mid_0)

        delta_log_rad = upd_c + upd_m
        rad_gain = torch.exp(delta_log_rad.clamp(-1.5, 1.5))
        base_rad = base * rad_gain

        # Pillar 2: Phase-aware de-attenuated detail + Jacobian-calibrated raw sensor transport
        base_rad_mid = separable_gaussian_video(base_rad, self.k_mid, self.sigma_mid)
        h_rad = base_rad - base_rad_mid

        det_update = torch.zeros_like(h_rad)
        if not zero_identity_override:
            for k, delta in enumerate(OFFSETS):
                rev_k = REVERSE_INDEX[k]
                if self.mode == "ablate_no_deatt_raw":
                    donor_h = warp_video_offset(h_rad, flows[:, k], delta)
                else:
                    donor_h = deattenuated_warp_offset(
                        detail=h_rad,
                        flow_fwd_dir=flows[:, k],
                        flow_rev_dir=flows[:, rev_k],
                        fb_conf_rev_dir=fb_conf[:, rev_k],
                        delta=delta,
                        beta_bp=beta_bp,
                    )
                det_update = det_update + eff_det[:, k] * (donor_h - h_rad)

        sens_update = torch.zeros_like(h_rad)
        if self.mode != "ablate_no_deatt_raw" and not zero_identity_override:
            h_x0 = cache["h_x0"]
            mu_b = cache["mu_b"]
            mu_x = cache["mu_x"]
            hx_num = h_x0.clone()
            hx_den = torch.ones_like(fb_conf[:, 0])
            for k, delta in enumerate(OFFSETS):
                weight_k = r_dir[:, k]
                hx_num = hx_num + weight_k * warp_video_offset(h_x0, flows[:, k], delta)
                hx_den = hx_den + weight_k
            hx_agg = hx_num / hx_den.clamp_min(1.0)

            eps_h = 1e-2
            rms_b = torch.sqrt(
                separable_gaussian_video((h_rad / mu_b).square(), self.k_mid, self.sigma_mid) + (eps_h ** 2)
            )
            rms_x = torch.sqrt(
                separable_gaussian_video((hx_agg / mu_x).square(), self.k_mid, self.sigma_mid) + (eps_h ** 2)
            )
            dimless_kappa = (rms_b / rms_x).clamp(0.0, 5.0)
            calibrated_hx = (mu_b / mu_x) * dimless_kappa * hx_agg
            sens_update = eff_sens * (calibrated_hx - h_rad)

        unclamped = base_rad_mid + h_rad + det_update + sens_update
        pred = unclamped.clamp(0.0, 1.0) if clamp_output else unclamped

        aux: Dict[str, object] = {
            "mean_c_coarse": float(eff_coarse.detach().sum(dim=1).mean().item()),
            "mean_c_mid": float(eff_mid.detach().sum(dim=1).mean().item()),
            "mean_w_det": float(eff_det.detach().sum(dim=1).mean().item()),
            "mean_g_sens": float(eff_sens.detach().mean().item()),
            "mean_omega_coarse": float(omega_coarse.detach().mean().item()),
            "mean_omega_mid": float(omega_mid.detach().mean().item()),
            "mean_omega_detail": float(omega_detail.detach().mean().item()),
            "beta_bp": float(beta_bp.detach().item()),
            "mean_abs_rad_log_shift": float(delta_log_rad.detach().abs().mean().item()),
        }
        if return_aux:
            aux.update(
                base_rad=base_rad,
                c_coarse=eff_coarse,
                c_mid=eff_mid,
                w_det=eff_det,
                g_sens=eff_sens,
                unclamped=unclamped,
            )
        return {"pred_pixels": pred, "aux": aux}


def clip_static_exposure_align_gt(
    base: Tensor,
    gt: Tensor,
    box_kernel: int = 33,
    eps: float = 1e-2,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0,
) -> Tensor:
    """Train-only sequence-constant exposure decoupling for source supervision."""
    _require(base.shape == gt.shape and base.ndim == 5 and base.shape[1] == 3, "Invalid base/gt shapes")
    with torch.no_grad():
        mean_b = base.detach().mean(dim=2)
        mean_g = gt.detach().mean(dim=2)
        smooth_b = box_smooth_2d(mean_b, box_kernel)
        smooth_g = box_smooth_2d(mean_g, box_kernel)
        ratio = ((smooth_b + eps) / (smooth_g + eps)).clamp(min_ratio, max_ratio)
        aligned = (gt * ratio.unsqueeze(2)).clamp(0.0, 1.0)
    return aligned


def _spatial_gradients(video: Tensor) -> Tuple[Tensor, Tensor]:
    dx = video[:, :, :, :, 1:] - video[:, :, :, :, :-1]
    dy = video[:, :, :, 1:, :] - video[:, :, :, :-1, :]
    return dx, dy


def compute_eid_training_loss(
    pred: Tensor,
    base: Tensor,
    gt: Tensor,
    flows: Tensor,
    geom: Dict[str, Tensor],
    loss_mode: str = "decoupled",
    w_spatial: float = 1.0,
    w_grad: float = 0.30,
    w_temp_diff: float = 0.35,
    w_traj_diff: float = 0.35,
    charb_eps: float = 1e-3,
) -> Dict[str, Tensor]:
    """Source-domain training objective combining spatial sharpness and matched temporal dynamics."""
    _require(loss_mode in LOSS_MODES, f"Unknown loss_mode {loss_mode!r}")
    target = clip_static_exposure_align_gt(base, gt) if loss_mode == "decoupled" else gt

    loss_spatial = torch.sqrt((pred - target).square() + (charb_eps ** 2)).mean()

    loss_grad = pred.new_zeros(())
    b, c, t, _, _ = pred.shape
    p_cur, g_cur = pred, target
    for scale_weight in (0.6, 0.4):
        pdx, pdy = _spatial_gradients(p_cur)
        gdx, gdy = _spatial_gradients(g_cur)
        loss_grad = loss_grad + scale_weight * ((pdx - gdx).abs().mean() + (pdy - gdy).abs().mean())
        p_frames = p_cur.permute(0, 2, 1, 3, 4).reshape(b * t, c, p_cur.shape[-2], p_cur.shape[-1])
        g_frames = g_cur.permute(0, 2, 1, 3, 4).reshape(b * t, c, g_cur.shape[-2], g_cur.shape[-1])
        p_cur = F.avg_pool2d(p_frames, 2, 2).reshape(b, t, c, p_cur.shape[-2] // 2, p_cur.shape[-1] // 2).permute(0, 2, 1, 3, 4)
        g_cur = F.avg_pool2d(g_frames, 2, 2).reshape(b, t, c, g_cur.shape[-2] // 2, g_cur.shape[-1] // 2).permute(0, 2, 1, 3, 4)

    loss_temp_diff = pred.new_zeros(())
    dp = pred[:, :, 1:] - pred[:, :, :-1]
    dg = target[:, :, 1:] - target[:, :, :-1]
    for pool_k, sw in ((1, 0.5), (2, 0.3), (4, 0.2)):
        if pool_k == 1:
            dp_k, dg_k = dp, dg
        else:
            dp_k = _pool_4d(dp, pool_k)
            dg_k = _pool_4d(dg, pool_k)
        loss_temp_diff = loss_temp_diff + sw * torch.sqrt((dp_k - dg_k).square() + (charb_eps ** 2)).mean()

    loss_traj = pred.new_zeros(())
    valid_mask = geom["valid_mask"]
    fb_conf = geom["fb_conf"]
    for k in (1, 2):  # OFFSETS[1] = -1, OFFSETS[2] = +1
        delta = OFFSETS[k]
        weight = (fb_conf[:, k] * valid_mask[:, k]).detach()
        res_pred = pred - warp_video_offset(pred, flows[:, k], delta)
        res_tgt = target - warp_video_offset(target, flows[:, k], delta)
        num = (weight * torch.sqrt((res_pred - res_tgt).square() + (charb_eps ** 2))).sum()
        den = (weight.sum() * 3.0).clamp_min(1.0)
        loss_traj = loss_traj + 0.5 * (num / den)

    total = (
        w_spatial * loss_spatial
        + w_grad * loss_grad
        + w_temp_diff * loss_temp_diff
        + w_traj_diff * loss_traj
    )
    return {
        "total": total,
        "spatial": loss_spatial.detach(),
        "grad": loss_grad.detach(),
        "temp_diff": loss_temp_diff.detach(),
        "traj_diff": loss_traj.detach(),
    }


def fixed_baseline_controls(
    base: Tensor,
    flows: Tensor,
    geom: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Non-learned trajectory controls on the identical flow & validity geometry."""
    valid_mask = geom["valid_mask"]
    fb_conf = geom["fb_conf"]

    controls: Dict[str, Tensor] = {"anchor": base}

    num = torch.zeros_like(base)
    den = torch.zeros_like(fb_conf[:, 0])
    for k, delta in enumerate(OFFSETS):
        w = fb_conf[:, k] * valid_mask[:, k]
        num = num + w * warp_video_offset(base, flows[:, k], delta)
        den = den + w
    has_donor = (den > 1e-4).to(base.dtype)
    donor_mean = num / den.clamp_min(1e-4)
    controls["fixed_temporal_mean_0.5"] = (
        base + 0.5 * has_donor * (donor_mean - base)
    ).clamp(0.0, 1.0)

    b_lp = separable_gaussian_video(base, 17, 4.0)
    b_hp = base - b_lp
    hp_num = torch.zeros_like(b_hp)
    for k, delta in enumerate(OFFSETS):
        w = fb_conf[:, k] * valid_mask[:, k]
        hp_num = hp_num + w * warp_video_offset(b_hp, flows[:, k], delta)
    hp_mean = hp_num / den.clamp_min(1e-4)
    controls["fixed_lp_locked_0.5"] = (
        b_lp + b_hp + 0.5 * has_donor * (hp_mean - b_hp)
    ).clamp(0.0, 1.0)

    return controls
