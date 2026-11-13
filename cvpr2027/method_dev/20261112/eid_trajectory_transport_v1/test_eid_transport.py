"""Unit tests for Photometric-Invariant Dual-Manifold Trajectory Transport (EID-TT)."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

if torch is not None:
    from eid_transport import (  # noqa: E402
        EIDTrajectoryTransport,
        MODES,
        OFFSETS,
        clip_static_exposure_align_gt,
        compute_cycle_geometry,
        compute_eid_training_loss,
        compute_reciprocal_raft_flows,
        deattenuated_warp_offset,
        extract_invariant_features,
        fixed_baseline_controls,
        separable_gaussian_video,
        warp_frames_2d,
        warp_video_offset,
    )


def _synthetic_consistent_flows(b: int, t: int, h: int, w: int, vx: float = 0.5, vy: float = -0.5):
    """Constant translational velocity field so forward and reverse flows cancel exactly."""
    flows = torch.zeros((b, len(OFFSETS), 2, t, h, w), dtype=torch.float32)
    for k, delta in enumerate(OFFSETS):
        flows[:, k, 0] = float(delta) * vx
        flows[:, k, 1] = float(delta) * vy
    return flows


@unittest.skipIf(torch is None, "PyTorch not installed in local interpreter")
class TestEIDTransport(unittest.TestCase):
    def test_gt_free_inference_signatures(self) -> None:
        forbidden = {"gt", "hq", "label", "labels", "target_gt", "oracle"}
        for fn in (
            compute_cycle_geometry,
            compute_reciprocal_raft_flows,
            extract_invariant_features,
            fixed_baseline_controls,
            EIDTrajectoryTransport.forward,
            EIDTrajectoryTransport.predict_conductances,
        ):
            params = set(inspect.signature(fn).parameters)
            self.assertEqual(params & forbidden, set(), f"Forbidden GT param in {fn}")

    def test_zero_flow_warp_identity_and_gaussian_constant(self) -> None:
        torch.manual_seed(7)
        x = torch.rand(4, 3, 64, 64, dtype=torch.float32)
        zero_flow = torch.zeros(4, 2, 64, 64, dtype=torch.float32)
        warped, mask = warp_frames_2d(x, zero_flow)
        self.assertTrue(torch.allclose(warped, x, atol=1e-6))
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))

        const_vid = torch.full((1, 3, 6, 64, 64), 0.37, dtype=torch.float32)
        smoothed = separable_gaussian_video(const_vid, 25, 6.0)
        self.assertTrue(torch.allclose(smoothed, const_vid, atol=1e-6))

    def test_zero_conductance_exact_anchor_identity(self) -> None:
        torch.manual_seed(11)
        low = torch.rand(1, 3, 6, 64, 64, dtype=torch.float32) * 0.1
        base = torch.rand(1, 3, 6, 64, 64, dtype=torch.float32)
        flows = _synthetic_consistent_flows(1, 6, 64, 64, vx=1.25, vy=-0.75)
        model = EIDTrajectoryTransport(mode="eid_full")
        out = model(low, base, flows, zero_identity_override=True)["pred_pixels"]
        self.assertTrue(torch.allclose(out, base, atol=1e-6))

    def test_independent_photometric_scale_invariance_and_equivariance(self) -> None:
        """Verify F(beta * X, alpha * B) == F(X, B) and T(beta * X, alpha * B) == alpha * T(X, B)."""
        torch.manual_seed(19)
        low = (torch.rand(1, 3, 6, 64, 64, dtype=torch.float32) * 0.08 + 0.005)
        base = (torch.rand(1, 3, 6, 64, 64, dtype=torch.float32) * 0.35 + 0.02)
        flows = _synthetic_consistent_flows(1, 6, 64, 64, vx=0.5, vy=0.5)
        geom = compute_cycle_geometry(flows)

        model = EIDTrajectoryTransport(mode="eid_full")
        # Randomize final layers as well so non-trivial learned weights are tested!
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(mean=0.0, std=0.15)

        feat_ref, _ = extract_invariant_features(low, base, flows, geom)
        out_ref = model(low, base, flows, geom=geom, clamp_output=False)["pred_pixels"]

        for alpha, beta in ((0.25, 3.7), (2.4, 0.15), (0.6, 0.6)):
            feat_scaled, _ = extract_invariant_features(beta * low, alpha * base, flows, geom)
            max_feat_diff = float((feat_scaled - feat_ref).abs().max().item())
            self.assertLess(max_feat_diff, 1e-4, f"Feature scale drift {max_feat_diff} for ({alpha}, {beta})")

            out_scaled = model(beta * low, alpha * base, flows, geom=geom, clamp_output=False)["pred_pixels"]
            rel_err = float((out_scaled - alpha * out_ref).abs().max().item() / alpha)
            self.assertLess(rel_err, 2e-4, f"Operator equivariance drift {rel_err} for ({alpha}, {beta})")

    def test_zero_static_exposure_bias_markov_property(self) -> None:
        """A temporally static clip under zero motion must experience zero radiometric drift."""
        torch.manual_seed(23)
        frame_x = torch.rand(1, 3, 1, 64, 64) * 0.05 + 0.01
        frame_b = torch.rand(1, 3, 1, 64, 64) * 0.7 + 0.1
        low = frame_x.expand(1, 3, 6, 64, 64).clone()
        base = frame_b.expand(1, 3, 6, 64, 64).clone()
        flows = torch.zeros((1, len(OFFSETS), 2, 6, 64, 64), dtype=torch.float32)

        model = EIDTrajectoryTransport(mode="eid_full")
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0.0, 0.2)
            # Disable raw sensor injection to isolate pure trajectory transport conservation
            model.sensor_head[-1].weight.zero_()
            model.sensor_head[-1].bias.fill_(-50.0)

        res = model(low, base, flows, return_aux=True)
        self.assertLess(res["aux"]["mean_abs_rad_log_shift"], 1e-6)
        max_diff = float((res["pred_pixels"] - base).abs().max().item())
        self.assertLess(max_diff, 1e-5)

    def test_deattenuated_warp_restores_subpixel_high_frequency_energy(self) -> None:
        """At half-pixel shift (0.5, 0.5), de-attenuated warp preserves higher HF energy than bilinear."""
        h, w, t = 64, 64, 6
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
        pattern = torch.sin(xx * 1.1) * torch.cos(yy * 1.1) * 0.2
        # Build sequence translating by (-0.5, -0.5) per frame
        frames = []
        for ti in range(t):
            p_t = torch.sin((xx - 0.5 * ti) * 1.1) * torch.cos((yy - 0.5 * ti) * 1.1) * 0.2
            frames.append(p_t.view(1, 1, 1, h, w).expand(1, 3, 1, h, w))
        detail = torch.cat(frames, dim=2)
        flows = _synthetic_consistent_flows(1, t, h, w, vx=0.5, vy=0.5)
        geom = compute_cycle_geometry(flows)

        k = 2  # delta = +1 (subpixel shift = +0.5, +0.5)
        rev_k = 1
        lin_warp = warp_video_offset(detail, flows[:, k], OFFSETS[k])
        deatt_warp = deattenuated_warp_offset(
            detail=detail,
            flow_fwd_dir=flows[:, k],
            flow_rev_dir=flows[:, rev_k],
            fb_conf_rev_dir=geom["fb_conf"][:, rev_k],
            delta=OFFSETS[k],
            beta_bp=torch.tensor(0.75),
        )
        interior = slice(8, -8)
        target = detail[:, :, :-1, interior, interior]
        err_lin = float((lin_warp[:, :, :-1, interior, interior] - target).square().mean().item())
        err_deatt = float((deatt_warp[:, :, :-1, interior, interior] - target).square().mean().item())
        self.assertLess(err_deatt, err_lin)

    def test_all_modes_and_backward_gradients(self) -> None:
        torch.manual_seed(31)
        low = torch.rand(1, 3, 6, 64, 64, dtype=torch.float32) * 0.1
        base = torch.rand(1, 3, 6, 64, 64, dtype=torch.float32) * 0.8
        gt = torch.rand(1, 3, 6, 64, 64, dtype=torch.float32) * 0.8
        flows = _synthetic_consistent_flows(1, 6, 64, 64, vx=0.5, vy=-0.25)
        geom = compute_cycle_geometry(flows)

        aligned_gt = clip_static_exposure_align_gt(base, gt)
        self.assertEqual(aligned_gt.shape, gt.shape)

        for mode in MODES:
            model = EIDTrajectoryTransport(mode=mode)
            out = model(low, base, flows, geom=geom)
            pred = out["pred_pixels"]
            self.assertEqual(pred.shape, base.shape)
            self.assertTrue(bool(torch.isfinite(pred).all()))
            losses = compute_eid_training_loss(pred, base, gt, flows, geom, loss_mode="decoupled")
            losses["total"].backward()
            grad_norms = [float(p.grad.abs().max().item()) for p in model.parameters() if p.grad is not None]
            self.assertTrue(any(g > 0.0 for g in grad_norms), f"Zero gradients in mode {mode}")


if __name__ == "__main__":
    unittest.main()
