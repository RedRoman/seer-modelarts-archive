"""Unit tests for Bipolar Innovation-Enveloped EID-TT v2 mathematical invariants."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@unittest.skipIf(torch is None, "PyTorch required for EID-TT tensor tests")
class TestEIDTrajectoryTransportV2(unittest.TestCase):
    def setUp(self) -> None:
        from eid_transport import EIDTrajectoryTransport

        torch.manual_seed(42)
        self.model = EIDTrajectoryTransport(mode="eid_full").eval()

    def test_01_gt_free_public_signatures(self) -> None:
        from eid_transport import (
            EIDTrajectoryTransport,
            compute_bipolar_envelope,
            compute_cycle_geometry,
            compute_reciprocal_raft_flows,
            extract_invariant_features,
        )

        forbidden = {"gt", "target", "oracle", "label"}
        for fn in (
            compute_cycle_geometry,
            compute_reciprocal_raft_flows,
            compute_bipolar_envelope,
            extract_invariant_features,
            EIDTrajectoryTransport.prepare_static_inputs,
            EIDTrajectoryTransport.predict_conductances,
            EIDTrajectoryTransport.forward,
        ):
            params = set(inspect.signature(fn).parameters.keys())
            self.assertTrue(params.isdisjoint(forbidden), f"Forbidden parameter in {fn}: {params & forbidden}")

    def test_02_zero_flow_and_gaussian_constant_preservation(self) -> None:
        from eid_transport import separable_gaussian_video, warp_frames_2d

        const_vid = torch.full((1, 3, 6, 64, 64), 0.37, dtype=torch.float32)
        blurred = separable_gaussian_video(const_vid, 25, 6.0)
        self.assertLess(float((blurred - const_vid).abs().max()), 1e-6)

        img = torch.rand(2, 3, 64, 64)
        zero_fl = torch.zeros(2, 2, 64, 64)
        warped, mask = warp_frames_2d(img, zero_fl)
        self.assertTrue(bool((mask == 1.0).all()))
        self.assertLess(float((warped - img).abs().max()), 1e-6)

    def test_03_zero_identity_override_exact(self) -> None:
        low = torch.rand(1, 3, 6, 64, 64) * 0.2
        base = torch.rand(1, 3, 6, 64, 64) * 0.8 + 0.1
        flows = torch.randn(1, 4, 2, 6, 64, 64) * 1.5
        out = self.model(low, base, flows, zero_identity_override=True)
        self.assertLess(float((out["pred_pixels"] - base).abs().max()), 1e-6)

    def test_04_independent_photometric_lie_group_equivariance(self) -> None:
        from eid_transport import EIDTrajectoryTransport

        m = EIDTrajectoryTransport(mode="eid_full")
        for p in m.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.05)
        m.eval()

        low = torch.rand(1, 3, 6, 64, 64) * 0.25 + 0.02
        base = torch.rand(1, 3, 6, 64, 64) * 0.35 + 0.05
        flows = torch.randn(1, 4, 2, 6, 64, 64) * 0.8

        alpha = 1.73
        beta = 0.41
        y1 = m(low, base, flows, clamp_output=False)["pred_pixels"]
        y2 = m(low * beta, base * alpha, flows, clamp_output=False)["pred_pixels"]
        rel_err = float((y2 - alpha * y1).abs().max() / (alpha * y1.abs().mean()))
        self.assertLess(rel_err, 1e-5)

    def test_05_zero_static_exposure_bias(self) -> None:
        spatial = torch.rand(1, 3, 1, 64, 64) * 0.5 + 0.2
        base = spatial.expand(1, 3, 6, 64, 64).clone()
        low = (spatial * 0.15).expand(1, 3, 6, 64, 64).clone()
        flows = torch.zeros(1, 4, 2, 6, 64, 64)
        out = self.model(low, base, flows, clamp_output=False)
        self.assertLess(float((out["pred_pixels"] - base).abs().max()), 1e-6)

    def test_06_null_flicker_boundary_condition_on_linear_lighting_ramp(self) -> None:
        """A monotonic illumination ramp without frame-independent flicker must trigger Omega^bip == 0."""
        t_vals = torch.linspace(0.25, 0.75, 8).view(1, 1, 8, 1, 1)
        spatial = torch.rand(1, 3, 1, 64, 64) * 0.4 + 0.3
        base = (spatial * t_vals).contiguous()
        low = (base * 0.2).contiguous()
        flows = torch.zeros(1, 4, 2, 8, 64, 64)
        out = self.model(low, base, flows, clamp_output=False)
        self.assertEqual(out["aux"]["mean_omega_coarse"], 0.0)
        self.assertEqual(out["aux"]["mean_omega_mid"], 0.0)
        self.assertEqual(out["aux"]["mean_omega_detail"], 0.0)
        self.assertLess(float((out["pred_pixels"] - base).abs().max()), 1e-6)

    def test_07_positive_bipolar_activation_under_alternating_flicker(self) -> None:
        """Alternating inter-frame exposure flicker must activate Omega^bip > 0.5 and reduce temporal variance."""
        spatial = torch.full((1, 3, 8, 64, 64), 0.45)
        alt = torch.tensor([1.25, 0.80, 1.25, 0.80, 1.25, 0.80, 1.25, 0.80]).view(1, 1, 8, 1, 1)
        base = (spatial * alt).clamp(0.0, 1.0)
        low = (spatial * 0.1).contiguous()
        flows = torch.zeros(1, 4, 2, 8, 64, 64)
        out = self.model(low, base, flows, clamp_output=True)
        self.assertGreater(out["aux"]["mean_omega_coarse"], 0.50)
        before_diff = float((base[:, :, 1:] - base[:, :, :-1]).abs().mean())
        after_diff = float((out["pred_pixels"][:, :, 1:] - out["pred_pixels"][:, :, :-1]).abs().mean())
        self.assertLess(after_diff, before_diff * 0.72)

    def test_08_half_pixel_back_projection_preserves_high_frequency(self) -> None:
        from eid_transport import compute_cycle_geometry, deattenuated_warp_offset, warp_video_offset

        xs = torch.arange(64, dtype=torch.float32).view(1, 1, 1, 1, 64)
        ys = torch.arange(64, dtype=torch.float32).view(1, 1, 1, 64, 1)
        pattern = 0.25 * torch.cos(1.4 * xs) * torch.cos(1.4 * ys)
        detail = pattern.expand(1, 3, 6, 64, 64).clone()

        flows = torch.zeros(1, 4, 2, 6, 64, 64)
        flows[:, 1, :] = 0.5
        flows[:, 2, :] = -0.5
        geom = compute_cycle_geometry(flows)

        naive = warp_video_offset(detail, flows[:, 1], -1)
        deatt = deattenuated_warp_offset(
            detail=detail,
            flow_fwd_dir=flows[:, 1],
            flow_rev_dir=flows[:, 2],
            fb_conf_rev_dir=geom["fb_conf"][:, 2],
            delta=-1,
            beta_bp=torch.tensor(0.85),
        )
        interior = slice(8, 56)
        e_naive = float(naive[:, :, 2:4, interior, interior].square().mean())
        e_deatt = float(deatt[:, :, 2:4, interior, interior].square().mean())
        self.assertGreater(e_deatt, e_naive * 1.15)

    def test_09_static_inputs_cache_exact_equivalence(self) -> None:
        low = torch.rand(2, 3, 6, 64, 64) * 0.3
        base = torch.rand(2, 3, 6, 64, 64) * 0.7 + 0.1
        flows = torch.randn(2, 4, 2, 6, 64, 64) * 0.9
        y_direct = self.model(low, base, flows)["pred_pixels"]
        geom, feats, cache = self.model.prepare_static_inputs(low, base, flows)
        y_cached = self.model(low, base, flows, geom=geom, static_inputs=(feats, cache))["pred_pixels"]
        self.assertEqual(float((y_direct - y_cached).abs().max()), 0.0)

    def test_10_all_modes_gradient_flow(self) -> None:
        from eid_transport import EIDTrajectoryTransport, MODES, compute_cycle_geometry, compute_eid_training_loss

        low = torch.rand(1, 3, 6, 64, 64) * 0.2
        base = torch.rand(1, 3, 6, 64, 64) * 0.6 + 0.1
        gt = torch.rand(1, 3, 6, 64, 64) * 0.6 + 0.1
        flows = torch.randn(1, 4, 2, 6, 64, 64) * 0.6
        geom = compute_cycle_geometry(flows)

        for mode in MODES:
            m = EIDTrajectoryTransport(mode=mode).train()
            out = m(low, base, flows, geom=geom)
            losses = compute_eid_training_loss(out["pred_pixels"], base, gt, flows, geom, loss_mode="decoupled")
            losses["total"].backward()
            grads = [p.grad for p in m.parameters() if p.grad is not None]
            self.assertGreater(len(grads), 0, f"No gradients for mode {mode}")
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))


if __name__ == "__main__":
    unittest.main()
