"""Data-free checks for observable donor controls and sampling direction."""
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

CODE = Path(__file__).resolve().parents[1]
candidate = None
if torch is not None:
    spec = importlib.util.spec_from_file_location("_observation_test", CODE / "models/observation_candidates.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)


@unittest.skipIf(torch is None, "Torch runtime required")
class ObservationTests(unittest.TestCase):
    def inputs(self, t=5, h=16, w=16, batch=1):
        generator = torch.Generator().manual_seed(7)
        low = torch.rand(batch, 3, t, h, w, generator=generator) * .2
        base = torch.rand(batch, 3, t, h, w, generator=generator)
        flows = torch.zeros(batch, 2, 2, t, h, w)
        masks = torch.ones(batch, 2, 1, t, h, w, dtype=torch.bool)
        masks[:, 0, :, :2] = False
        masks[:, 1, :, -2:] = False
        return low, base, flows, masks

    def test_restored_zero_flow_samples_correct_temporal_source(self):
        low, base, flow, masks = self.inputs(batch=2)
        d, valid, _ = candidate.build_observation_candidates(low, base, flow, masks, mode="restored")
        torch.testing.assert_close(d[:, 0, :, 2:], base[:, :, :-2] - base[:, :, 2:], atol=0, rtol=0)
        torch.testing.assert_close(d[:, 1, :, :-2], base[:, :, 2:] - base[:, :, :-2], atol=0, rtol=0)
        self.assertTrue(torch.equal(valid, masks))
        self.assertEqual(float(d[:, 0, :, :2].abs().max()), 0)
        self.assertEqual(float(d[:, 1, :, -2:].abs().max()), 0)

    def test_pixel_translation_uses_target_to_source_gather(self):
        low, base, flow, masks = self.inputs()
        flow[:, :, 0] = 1
        masks[:, :, :, :, :, -1] = False
        d, _, _ = candidate.build_observation_candidates(low, base, flow, masks, mode="restored")
        expected = base[:, :, 2:, :, 1:] - base[:, :, :-2, :, :-1]
        torch.testing.assert_close(d[:, 1, :, :-2, :, :-1], expected, atol=0, rtol=0)

    def test_lowpass_removes_flat_donor_brightness_changes(self):
        low, base, flow, masks = self.inputs()
        for time in range(base.shape[2]):
            base[:, :, time] = .2 + .1 * time
            low[:, :, time] = .03 + .01 * time
        restored, _, _ = candidate.build_observation_candidates(low, base, flow, masks, mode="restored")
        self.assertGreater(float(restored.abs().max()), .1)
        for mode in ("restored_detail", "raw_detail"):
            d, _, _ = candidate.build_observation_candidates(low, base, flow, masks, mode=mode)
            self.assertLess(float(d.abs().max()), 2e-6)

    def test_raw_detail_uses_exact_source_low_frame_not_target_low_frame(self):
        low, base, flow, masks = self.inputs()
        low[:] = .08; base[:] = .4
        original, _, _ = candidate.build_observation_candidates(low, base, flow, masks, mode="raw_detail")
        # Perturb only low frame0: only the past source of target frame2 changes.
        low[:, :, 0, 5:10, 5:10] += .03
        changed, _, _ = candidate.build_observation_candidates(low, base, flow, masks, mode="raw_detail")
        delta = changed - original
        self.assertGreater(float(delta[:, 0, :, 2].abs().max()), .01)
        delta[:, 0, :, 2] = 0
        self.assertEqual(float(delta.abs().max()), 0)

    def test_both_detail_controls_share_eroded_mask_and_outputs_are_bounded(self):
        low, base, flow, masks = self.inputs()
        masks[:, :, :, :, 7, 7] = False
        masks_by_mode = {}
        for mode in candidate.CANDIDATE_MODES:
            d, valid, meta = candidate.build_observation_candidates(low, base, flow, masks, mode=mode)
            masks_by_mode[mode] = valid
            self.assertEqual(float((d * (~valid)).abs().max()), 0)
            result = base[:, None] + d
            self.assertGreaterEqual(float(result.min()), -1e-7)
            self.assertLessEqual(float(result.max()), 1 + 1e-7)
            self.assertEqual(json.loads(json.dumps(meta)), meta)
        self.assertTrue(torch.equal(masks_by_mode["raw_detail"], masks_by_mode["restored_detail"]))
        self.assertFalse(bool(masks_by_mode["raw_detail"][:, :, :, :, 3:12, 3:12].any()))
        self.assertTrue(bool(masks_by_mode["restored"][:, :, :, 2, 3, 3].any()))

    def test_no_gt_argument_or_autograd_path(self):
        self.assertEqual(set(inspect.signature(candidate.build_observation_candidates).parameters),
                         {"low", "base", "flows", "masks", "mode", "offset"})
        inputs = list(self.inputs())
        inputs[0].requires_grad_(True); inputs[1].requires_grad_(True)
        for mode in candidate.CANDIDATE_MODES:
            d, _, _ = candidate.build_observation_candidates(*inputs, mode=mode)
            self.assertFalse(d.requires_grad)

    def test_invalid_shapes_flow_mask_and_values_fail_closed(self):
        low, base, flow, masks = self.inputs()
        with self.assertRaisesRegex(ValueError, "out-of-time"):
            candidate.build_observation_candidates(low, base, flow, torch.ones_like(masks), mode="restored")
        for mode in ("typo",):
            with self.assertRaises(ValueError):
                candidate.build_observation_candidates(low, base, flow, masks, mode=mode)
        with self.assertRaises(TypeError):
            candidate.build_observation_candidates(low.double(), base, flow, masks, mode="restored")
        bad = flow.clone(); bad[:, :, 0, :, :, :] = float("nan")
        with self.assertRaises(ValueError):
            candidate.build_observation_candidates(low, base, bad, masks, mode="restored")
        bad_masks = masks.float(); bad_masks[:, 0, :, 2] = .5
        with self.assertRaises(ValueError):
            candidate.build_observation_candidates(low, base, flow, bad_masks, mode="raw_detail")


if __name__ == "__main__":
    unittest.main()
