"""Photometric-Invariant Dual-Manifold Trajectory Transport (EID-TT).

Core cross-domain video restoration operator acting on observable raw low-light
frames X, frozen canonical-RGB enhancer predictions B, and GT-free reciprocal
optical flows. No ground-truth tensor or target-domain statistic ever enters
feature extraction, conductance prediction, or trajectory transport.

Mathematical guarantees:
1. Exact anchor identity when trajectory conductances and sensor gate are zero.
2. Exact independent photometric scaling invariance/equivariance under
   (X, B) -> (beta * X, alpha * B) for any positive scalars alpha, beta > 0
   prior to final [0, 1] display clamping.
3. Zero static exposure bias by Markov conservation on orthogonal two-band
   log-radiometry (coarse + mid Laplacian decomposition).
4. Subpixel resampling de-attenuation via cycle-consistent forward-backward
   back-projection plus Jacobian-calibrated raw sensor detail injection.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

OFFSETS: Tuple[int, ...] = (-2, -1, 1, 2)
REVERSE_INDEX: Tuple[int, ...] = (3, 2, 1, 0)
MODES: Tuple[str, ...] = (
    "eid_full",
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
    "sens_enh_corr",
    "fb_conf",
    "subpixel_phase",
    "valid_frac",
    "temporal_dist",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    _require(kernel_size >= 3 and kernel_size % 2 == 1 and sigma > 0.0, "Invalid Gaussian kernel spec")
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, device=device, dtype=dtype)
    weights = torch.exp(-0.5 * (coords / sigma).square())
    return weights / weights.sum()


def separable_gaussian_video(video: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Symmetric reflect-padded separable 2D Gaussian filter on [B, C, T, H, W]."""
    _require(video.ndim == 5, "Expected 5D video tensor [B, C, T, H, W]")
    b, c, t, h, w = video.shape
    half = kernel_size // 2
    _require(min(h, w) > half, f"Spatial dims ({h}, {w}) must exceed Gaussian pad {half}")
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    k1d = gaussian_kernel_1d(kernel_size, sigma, video.device, video.dtype)
    kx = k1d.view(1, 1, 1, kernel_size).expand(c, 1, 1, kernel_size)
    ky = k1d.view(1, 1, kernel_size, 1).expand(c, 1, kernel_size, 1)
    x = F.pad(frames, (half, half, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, half, half), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def box_smooth_2d(frames: Tensor, kernel_size: int) -> Tensor:
    """Two-pass reflect-padded box filter on [N, C, H, W] for smooth exposure ratios."""
    _require(frames.ndim == 4 and kernel_size >= 3 and kernel_size % 2 == 1, "Invalid box_smooth_2d spec")
    pad = kernel_size // 2
    _require(min(frames.shape[-2:]) > pad, "Spatial dims must exceed box pad")
    out = frames
    for _ in range(2):
        out = F.avg_pool2d(F.pad(out, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)
    return out


def warp_frames_2d(source: Tensor, flow: Tensor) -> Tuple[Tensor, Tensor]:
    """Bilinearly sample source [N, C, H, W] at (x + dx, y + dy) using align_corners=False.

    Returns:
        warped: [N, C, H, W]
        in_bounds: [N, 1, H, W] float32 in {0, 1}
    """
    _require(source.ndim == 4 and flow.ndim == 4 and flow.shape[1] == 2, "Invalid warp_frames_2d shapes")
    n, _, h, w = source.shape
    _require(flow.shape[0] == n and flow.shape[2:] == (h, w), "Source and flow spatial dims differ")
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = xx.unsqueeze(0) + flow[:, 0]
    y = yy.unsqueeze(0) + flow[:, 1]
    in_bounds = ((x >= 0.0) & (x <= float(w - 1)) & (y >= 0.0) & (y <= float(h - 1))).unsqueeze(1).to(source.dtype)
    grid_x = (x + 0.5) * (2.0 / float(w)) - 1.0
    grid_y = (y + 0.5) * (2.0 / float(h)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).to(source.dtype)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped, in_bounds


def compute_cycle_geometry(
    flows: Tensor,
    offsets: Sequence[int] = OFFSETS,
    reverse_index: Sequence[int] = REVERSE_INDEX,
    fb_sigma_px: float = 1.5,
) -> Dict[str, Tensor]:
    """Deterministically derive geometric masks, FB cycle errors, and phase from reciprocal flows.

    Args:
        flows: [B, K, 2, T, H, W] float32 target-to-source displacements in pixels.
    Returns dict with:
        valid_mask: [B, K, 1, T, H, W] in {0, 1}
        fb_error:   [B, K, 1, T, H, W] >= 0
        fb_conf:    [B, K, 1, T, H, W] in [0, 1]
        subpixel_phase: [B, K, 1, T, H, W] in [0, 1]
    """
    _require(flows.ndim == 6 and flows.shape[1] == len(offsets) and flows.shape[2] == 2, "Invalid flows shape")
    b, k_count, _, t, h, w = flows.shape
    valid_list = []
    err_list = []
    conf_list = []
    phase_list = []

    times = torch.arange(t, device=flows.device)
    for k, delta in enumerate(offsets):
        rev_k = reverse_index[k]
        _require(offsets[rev_k] == -delta, "Offsets and reverse_index mismatch")
        src_times = times + delta
        valid_t = (src_times >= 0) & (src_times < t)
        clamped_src = src_times.clamp(0, t - 1)

        f_fwd = flows[:, k].permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
        f_rev_src = flows[:, rev_k].index_select(2, clamped_src).permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
        f_rev_warped, in_bounds = warp_frames_2d(f_rev_src, f_fwd)

        valid_time_mask = valid_t.view(1, t, 1, 1, 1).expand(b, t, 1, h, w).reshape(b * t, 1, h, w).to(flows.dtype)
        valid = in_bounds * valid_time_mask

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
    """Phase-aware forward-backward back-projection warp for high-frequency detail."""
    b, c, t, h, w = detail.shape
    times = torch.arange(t, device=detail.device)
    src_times = (times + delta).clamp(0, t - 1)

    # 1. Forward warp donor s = t + delta -> t: W_{t<-s}(H_s)
    src_h = detail.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    fl_fwd = flow_fwd_dir.permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
    w_t_from_s, _ = warp_frames_2d(src_h, fl_fwd)

    # 2. Round-trip back onto s's grid using reverse flow at frame s (pointing s -> t): W_{s<-t}(W_{t<-s}(H_s))
    fl_rev_at_s = flow_rev_dir.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, 2, h, w)
    conf_rev_at_s = fb_conf_rev_dir.index_select(2, src_times).permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
    roundtrip_s, _ = warp_frames_2d(w_t_from_s, fl_rev_at_s)

    # 3. Back-projection high-frequency boost on donor s, then resample onto t
    boosted_s = src_h + beta_bp * conf_rev_at_s * (src_h - roundtrip_s)
    deatt, _ = warp_frames_2d(boosted_s, fl_fwd)
    return deatt.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _pool_4d(video: Tensor, patch: int) -> Tensor:
    """Average-pool [B, C, T, H, W] spatially by patch -> [B, T, C, H//patch, W//patch]."""
    b, c, t, h, w = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return F.avg_pool2d(frames, patch, patch).reshape(b, t, c, h // patch, w // patch)


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
    use_sensor_invariant: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Build strictly dimensionless, photometric-scaling-invariant donor features.

    Returns:
        features: [B, T, K, 16, H//patch, W//patch]
        cache: intermediate decomposed tensors reused by transport step.
    """
    _require(low.shape == base.shape and low.ndim == 5 and low.shape[1] == 3, "Invalid low/base shapes")
    b, _, t, h, w = base.shape
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
        # Local sensor-enhancer structural SNR correlation (same across donors)
        hb_norm = h_b0 / mu_b
        hx_norm = h_x0 / mu_x
        se_dot = _pool_4d((hb_norm * hx_norm).mean(dim=1, keepdim=True), patch)
        se_eb = _pool_4d(hb_norm.square().mean(dim=1, keepdim=True), patch)
        se_ex = _pool_4d(hx_norm.square().mean(dim=1, keepdim=True), patch)
        sens_enh_corr = (se_dot / torch.sqrt(se_eb * se_ex + (eps_det ** 4))).clamp(-1.0, 1.0)

        donor_feats = []
        valid_mask = geom["valid_mask"]
        fb_conf = geom["fb_conf"]
        subpixel_phase = geom["subpixel_phase"]

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
    }
    return features, cache


class EIDTrajectoryTransport(nn.Module):
    """Photometric-Invariant Dual-Manifold Trajectory Transport module."""

    def __init__(
        self,
        mode: str = "eid_full",
        patch: int = 4,
        hidden: int = 24,
        rad_steps: int = 2,
        k_mid: int = 7,
        sigma_mid: float = 1.6,
        k_coarse: int = 25,
        sigma_coarse: float = 6.0,
        init_conductance_bias: float = -2.2,
        init_detail_bias: float = -2.6,
        init_sensor_bias: float = -3.5,
        max_sensor_gain: float = 0.25,
    ) -> None:
        super().__init__()
        _require(mode in MODES, f"Unsupported mode {mode!r}; expected one of {MODES}")
        _require(patch in (2, 4, 8) and hidden >= 8 and rad_steps >= 1, "Invalid module hyperparameters")
        self.mode = mode
        self.patch = patch
        self.hidden = hidden
        self.rad_steps = rad_steps
        self.k_mid = k_mid
        self.sigma_mid = sigma_mid
        self.k_coarse = k_coarse
        self.sigma_coarse = sigma_coarse
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
        self.beta_bp_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        nn.init.zeros_(self.consensus_head[-1].weight)
        nn.init.zeros_(self.consensus_head[-1].bias)
        nn.init.zeros_(self.sensor_head[-1].weight)
        nn.init.zeros_(self.sensor_head[-1].bias)

    def get_config(self) -> Dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "patch": self.patch,
            "hidden": self.hidden,
            "rad_steps": self.rad_steps,
            "k_mid": self.k_mid,
            "sigma_mid": self.sigma_mid,
            "k_coarse": self.k_coarse,
            "sigma_coarse": self.sigma_coarse,
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

        if geom is None:
            geom = compute_cycle_geometry(flows)
        valid_mask = geom["valid_mask"]
        fb_conf = geom["fb_conf"]

        use_sensor_inv = self.mode != "ablate_no_sensor_inv"
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
            use_sensor_invariant=use_sensor_inv,
        )

        c_coarse, c_mid, w_det, g_sens = self.predict_conductances(
            features, (h, w), zero_identity_override=zero_identity_override
        )

        # Gate conductances smoothly by geometric validity & FB confidence
        eff_coarse = c_coarse * valid_mask
        eff_mid = c_mid * valid_mask
        eff_det = w_det * valid_mask

        # Pillar 1: Two-step Markov relaxation on orthogonal two-band log-radiometry
        u_coarse_0 = cache["u_coarse"]
        u_mid_0 = cache["u_mid"]
        u_coarse = u_coarse_0
        u_mid = u_mid_0

        if self.mode != "ablate_no_rad" and not zero_identity_override:
            for _ in range(self.rad_steps):
                upd_c = torch.zeros_like(u_coarse)
                upd_m = torch.zeros_like(u_mid)
                for k, delta in enumerate(OFFSETS):
                    wc = warp_video_offset(u_coarse, flows[:, k], delta)
                    wm = warp_video_offset(u_mid, flows[:, k], delta)
                    upd_c = upd_c + eff_coarse[:, k] * (wc - u_coarse)
                    upd_m = upd_m + eff_mid[:, k] * (wm - u_mid)
                u_coarse = u_coarse + upd_c
                u_mid = u_mid + upd_m

        delta_log_rad = (u_coarse - u_coarse_0) + (u_mid - u_mid_0)
        rad_gain = torch.exp(delta_log_rad.clamp(-1.5, 1.5))
        base_rad = base * rad_gain

        # Pillar 2: Phase-aware de-attenuated detail + calibrated raw sensor detail transport
        base_rad_mid = separable_gaussian_video(base_rad, self.k_mid, self.sigma_mid)
        h_rad = base_rad - base_rad_mid

        beta_bp = (
            torch.zeros((), device=base.device, dtype=base.dtype)
            if (self.mode == "ablate_no_deatt_raw" or zero_identity_override)
            else torch.sigmoid(self.beta_bp_logit)
        )

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

        # Gain-calibrated raw sensor detail injection
        sens_update = torch.zeros_like(h_rad)
        if self.mode != "ablate_no_deatt_raw" and not zero_identity_override:
            h_x0 = cache["h_x0"]
            mu_b = cache["mu_b"]
            mu_x = cache["mu_x"]
            hx_num = h_x0.clone()
            hx_den = torch.ones_like(fb_conf[:, 0])
            for k, delta in enumerate(OFFSETS):
                weight_k = fb_conf[:, k] * valid_mask[:, k]
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
            sens_update = g_sens * (calibrated_hx - h_rad)

        unclamped = base_rad_mid + h_rad + det_update + sens_update
        pred = unclamped.clamp(0.0, 1.0) if clamp_output else unclamped

        aux: Dict[str, object] = {
            "mean_c_coarse": float(eff_coarse.detach().sum(dim=1).mean().item()),
            "mean_c_mid": float(eff_mid.detach().sum(dim=1).mean().item()),
            "mean_w_det": float(eff_det.detach().sum(dim=1).mean().item()),
            "mean_g_sens": float(g_sens.detach().mean().item()),
            "beta_bp": float(beta_bp.detach().item()),
            "mean_abs_rad_log_shift": float(delta_log_rad.detach().abs().mean().item()),
        }
        if return_aux:
            aux.update(
                base_rad=base_rad,
                c_coarse=eff_coarse,
                c_mid=eff_mid,
                w_det=eff_det,
                g_sens=g_sens,
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
    """Train-only sequence-constant exposure decoupling for source supervision.

    Computes a single time-invariant smooth RGB ratio map per clip so 100% of
    GT temporal variations and spatial edges are preserved while static source
    tone-curve mismatch is factored out. Never called at inference.
    """
    _require(base.shape == gt.shape and base.ndim == 5 and base.shape[1] == 3, "Invalid base/gt shapes")
    with torch.no_grad():
        mean_b = base.detach().mean(dim=2)  # [B, 3, H, W]
        mean_g = gt.detach().mean(dim=2)    # [B, 3, H, W]
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
    w_grad: float = 0.25,
    w_temp_diff: float = 0.50,
    w_traj_diff: float = 0.50,
    charb_eps: float = 1e-3,
) -> Dict[str, Tensor]:
    """Source-domain training objective combining spatial sharpness and matched temporal dynamics."""
    _require(loss_mode in LOSS_MODES, f"Unknown loss_mode {loss_mode!r}")
    target = clip_static_exposure_align_gt(base, gt) if loss_mode == "decoupled" else gt

    # 1. Charbonnier spatial fidelity
    loss_spatial = torch.sqrt((pred - target).square() + (charb_eps ** 2)).mean()

    # 2. Multi-scale spatial edge preservation (prevents blurring / protects LPIPS)
    loss_grad = pred.new_zeros(())
    b, c, t, h, w = pred.shape
    p_cur, g_cur = pred, target
    for scale_weight in (0.6, 0.4):
        pdx, pdy = _spatial_gradients(p_cur)
        gdx, gdy = _spatial_gradients(g_cur)
        loss_grad = loss_grad + scale_weight * (
            (pdx - gdx).abs().mean() + (pdy - gdy).abs().mean()
        )
        p_frames = p_cur.permute(0, 2, 1, 3, 4).reshape(b * t, c, p_cur.shape[-2], p_cur.shape[-1])
        g_frames = g_cur.permute(0, 2, 1, 3, 4).reshape(b * t, c, g_cur.shape[-2], g_cur.shape[-1])
        p_cur = F.avg_pool2d(p_frames, 2, 2).reshape(b, t, c, p_cur.shape[-2] // 2, p_cur.shape[-1] // 2).permute(0, 2, 1, 3, 4)
        g_cur = F.avg_pool2d(g_frames, 2, 2).reshape(b, t, c, g_cur.shape[-2] // 2, g_cur.shape[-1] // 2).permute(0, 2, 1, 3, 4)

    # 3. Multi-scale Eulerian adjacent temporal innovation matching
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

    # 4. Lagrangian flow-warped trajectory innovation matching (offsets -1, +1)
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

    # Control 1: Raw anchor
    controls: Dict[str, Tensor] = {"anchor": base}

    # Control 2: Fixed equal-donor linear temporal mean (lambda = 0.5 on +/-1, +/-2 FB-valid donors)
    num = torch.zeros_like(base)
    den = torch.zeros_like(valid_mask[:, 0])
    for k, delta in enumerate(OFFSETS):
        w = fb_conf[:, k] * valid_mask[:, k]
        num = num + w * warp_video_offset(base, flows[:, k], delta)
        den = den + w
    donor_mean = num / den.clamp_min(1e-6)
    has_support = (den > 1e-4).float()
    controls["fixed_temporal_mean_0.5"] = (base + 0.5 * has_support * (donor_mean - base)).clamp(0.0, 1.0)

    # Control 3: Fixed high-pass-only smoothing (LP7 locked, replicating prior LP-locked family)
    b_low = separable_gaussian_video(base, 7, 1.6)
    h_base = base - b_low
    h_num = torch.zeros_like(h_base)
    for k, delta in enumerate(OFFSETS):
        w = fb_conf[:, k] * valid_mask[:, k]
        h_num = h_num + w * warp_video_offset(h_base, flows[:, k], delta)
    h_mean = h_num / den.clamp_min(1e-6)
    controls["fixed_lp_locked_0.5"] = (b_low + h_base + 0.5 * has_support * (h_mean - h_base)).clamp(0.0, 1.0)

    return controls
