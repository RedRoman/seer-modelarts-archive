"""Data-free CPU controls: exact v1 parity, value isolation, time/mask edges."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

CODE = Path(__file__).resolve().parents[1]
try:
    import torch
except ImportError:
    torch = None


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = None if torch is None else load_file(
    "_observation_controls_test", CODE / "models/observation_controls.py")
reference = None if torch is None else load_file(
    "_observation_reference_test", CODE / "models/observation_candidates.py")


class StaticContractTests(unittest.TestCase):
    def test_public_builder_accepts_only_observables_and_fixed_recipe(self):
        tree = ast.parse((CODE / "models/observation_controls.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "build_observation_controls")
        self.assertEqual([a.arg for a in function.args.args], ["low", "base", "flows", "masks"])
        self.assertEqual([a.arg for a in function.args.kwonlyargs], ["mode", "offset"])
        self.assertIsNone(function.args.kwarg)
        self.assertIsNone(function.args.vararg)


@unittest.skipIf(torch is None, "Torch CPU runtime required")
class ObservationControlTests(unittest.TestCase):
    def inputs(self, *, t=7, batch=1, offset=2, shifted=False):
        generator = torch.Generator().manual_seed(71)
        low = torch.rand(batch, 3, t, 16, 16, generator=generator) * .15 + .02
        base = torch.rand(batch, 3, t, 16, 16, generator=generator) * .7 + .1
        flow = torch.zeros(batch, 2, 2, t, 16, 16)
        mask = torch.ones(batch, 2, 1, t, 16, 16, dtype=torch.bool)
        mask[:, 0, :, :offset] = False
        mask[:, 1, :, -offset:] = False
        if shifted:
            flow[:, 0, 0] = .5
            flow[:, 1, 0] = -1.
            mask[:, 0, :, :, :, -1] = False
            mask[:, 1, :, :, :, 0] = False
        return low, base, flow, mask

    def build(self, inputs, mode, offset=2):
        return control.build_observation_controls(*inputs, mode=mode, offset=offset)

    def q(self, low, base):
        low_l, base_l = reference._lowpass(low), reference._lowpass(base)
        gain = (base_l / (low_l + reference.GAIN_EPSILON)).clamp(
            reference.GAIN_MIN, reference.GAIN_MAX)
        return (base_l + gain * (low - low_l)).clamp(0, 1)

    def test_raw_wrapper_is_elementwise_exact_with_frozen_recipe(self):
        for offset in (1, 2, 3):
            for shifted in (False, True):
                inputs = self.inputs(batch=2, offset=offset, shifted=shifted)
                d, valid, meta = self.build(inputs, "raw_detail", offset)
                old, old_valid, old_meta = reference.build_observation_candidates(
                    *inputs, mode="raw_detail", offset=offset)
                self.assertTrue(torch.equal(d, old))
                self.assertTrue(torch.equal(valid, old_valid))
                self.assertEqual(meta["base_recipe"], old_meta)

    def test_constant_frames_are_identity_even_with_distinct_exposures(self):
        low, base, flow, mask = self.inputs(shifted=True)
        for t in range(7):
            low[:, :, t] = .02 + .01 * t
            base[:, :, t] = .2 + .06 * t
        for mode in control.CONTROL_MODES:
            d, _, _ = self.build((low, base, flow, mask), mode)
            self.assertLess(float(d.abs().max()), 2e-6)

    def test_same_frame_matches_formula_on_both_direction_supports(self):
        inputs = self.inputs(batch=2, shifted=True)
        low, base, _, _ = inputs
        d, valid, _ = self.build(inputs, "same_frame_detail")
        q = self.q(low, base)
        candidate = (reference._lowpass(base) + q - reference._lowpass(q)).clamp(0, 1)
        for j in range(2):
            torch.testing.assert_close(d[:, j], (candidate - base) * valid[:, j], atol=0, rtol=0)

    def test_same_frame_value_is_isolated_from_every_neighbour_low_and_base(self):
        inputs = list(self.inputs())
        d, valid, _ = self.build(inputs, "same_frame_detail")
        changed = [value.clone() for value in inputs]
        target = 3
        for t in range(7):
            if t != target:
                changed[0][:, :, t] = .9 - .05 * t
                changed[1][:, :, t] = .05 * t
        after, after_valid, _ = self.build(changed, "same_frame_detail")
        self.assertTrue(torch.equal(valid, after_valid))
        self.assertTrue(torch.equal(d[:, :, :, target], after[:, :, :, target]))
        # The target observation must still be a real input to this control.
        changed[0][:, :, target, 6:10, 6:10] = .9
        perturbed, _, _ = self.build(changed, "same_frame_detail")
        self.assertGreater(float((perturbed[:, :, :, target] - d[:, :, :, target]).abs().max()), .01)

    def test_same_frame_values_do_not_use_flow_coordinates(self):
        inputs = list(self.inputs())
        inputs[3][:, :, :, :, :, (0, -1)] = False
        before, mask, _ = self.build(inputs, "same_frame_detail")
        shifted = [value.clone() for value in inputs]
        shifted[2][:, 0, 0] = .5
        shifted[2][:, 1, 0] = -.5
        after, mask_after, _ = self.build(shifted, "same_frame_detail")
        self.assertTrue(torch.equal(mask, mask_after))
        self.assertTrue(torch.equal(before, after))

    def test_mismatch_samples_opposite_values_but_original_direction_flow(self):
        inputs = self.inputs(shifted=True)
        low, base, _, _ = inputs
        d, valid, _ = self.build(inputs, "mismatched_detail")
        q, target_low = self.q(low, base), reference._lowpass(base)
        # Target3/past flow is +.5 in x but its value comes from future frame5.
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        grid = torch.stack(((xx + .5 + .5) / 8 - 1, (yy + .5) / 8 - 1), -1)[None]
        warped = torch.nn.functional.grid_sample(q[:, :, 5], grid, align_corners=False,
                                                 mode="bilinear", padding_mode="border")
        warped = warped[:, :, None]
        candidate = (target_low[:, :, 3:4] + warped - reference._lowpass(warped)).clamp(0, 1)
        expected = (candidate[:, :, 0] - base[:, :, 3]) * valid[:, 0, :, 3]
        torch.testing.assert_close(d[:, 0, :, 3], expected, atol=0, rtol=0)

    def test_mismatch_low_perturbation_changes_the_swapped_direction_only(self):
        low, base, flow, mask = self.inputs()
        low[:] = .08; base[:] = .4
        before, _, _ = self.build((low, base, flow, mask), "mismatched_detail")
        # Frame5 is the swapped value for target3/past, not target3/future.
        low[:, :, 5, 5:10, 5:10] += .03
        after, _, _ = self.build((low, base, flow, mask), "mismatched_detail")
        self.assertGreater(float((after[:, 0, :, 3] - before[:, 0, :, 3]).abs().max()), .01)
        self.assertTrue(torch.equal(after[:, 1, :, 3], before[:, 1, :, 3]))

    def test_mismatch_edges_use_recorded_legal_clamps_without_wrap_or_validity_relaxation(self):
        inputs = self.inputs(t=5)
        d, valid, meta = self.build(inputs, "mismatched_detail")
        past, future = meta["directions"]
        self.assertEqual(past["value_source_requested"], [2, 3, 4, 5, 6])
        self.assertEqual(past["value_source_indices"], [2, 3, 4, 4, 4])
        self.assertEqual(past["value_index_clamped"], [False, False, False, True, True])
        self.assertEqual(future["value_source_indices"], [0, 0, 0, 1, 2])
        self.assertEqual(future["value_index_clamped"], [True, True, False, False, False])
        self.assertEqual(future["original_source_in_clip"], [True, True, True, False, False])
        self.assertFalse(bool(valid[:, 0, :, :2].any()))
        self.assertFalse(bool(valid[:, 1, :, -2:].any()))
        self.assertEqual(float(d[:, 0, :, :2].abs().max()), 0)
        self.assertEqual(float(d[:, 1, :, -2:].abs().max()), 0)
        # The active past direction at target3 explicitly uses frame4, not a
        # modulo-wrapped frame0, and remains active under its original mask.
        self.assertTrue(bool(valid[:, 0, :, 3].all()))
        low, base, _, _ = inputs
        q = self.q(low, base)[:, :, 4:5]
        candidate = (reference._lowpass(base)[:, :, 3:4] + q - reference._lowpass(q)).clamp(0, 1)
        torch.testing.assert_close(d[:, 0, :, 3], candidate[:, :, 0] - base[:, :, 3], atol=0, rtol=0)

    def test_all_modes_have_exact_same_eroded_mask_and_zero_invalid_corrections(self):
        inputs = list(self.inputs(batch=2, shifted=True))
        inputs[3][:, :, :, :, 7, 7] = False
        _, expected, _ = reference.build_observation_candidates(*inputs, mode="raw_detail")
        for mode in control.CONTROL_MODES:
            d, valid, meta = self.build(inputs, mode)
            self.assertTrue(torch.equal(valid, expected))
            self.assertEqual(float((d * (~valid)).abs().max()), 0)
            output = inputs[1][:, None] + d
            self.assertGreaterEqual(float(output.min()), -1e-7)
            self.assertLessEqual(float(output.max()), 1 + 1e-7)
            self.assertEqual(json.loads(json.dumps(meta)), meta)

    def test_no_gt_keyword_no_gradient_and_inputs_not_mutated(self):
        inputs = list(self.inputs())
        inputs[0].requires_grad_(True); inputs[1].requires_grad_(True)
        snapshots = [value.clone() for value in inputs]
        self.assertEqual(set(inspect.signature(control.build_observation_controls).parameters),
                         {"low", "base", "flows", "masks", "mode", "offset"})
        with self.assertRaises(TypeError):
            control.build_observation_controls(*inputs, mode="same_frame_detail", gt=inputs[1])
        for mode in control.CONTROL_MODES:
            d, _, meta = self.build(inputs, mode)
            self.assertFalse(d.requires_grad)
            self.assertFalse(meta["gt_input"])
            self.assertIn("all-clip", meta["interpretation"])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, snapshots)))

    def test_all_modes_reject_invalid_original_temporal_or_geometric_masks(self):
        inputs = list(self.inputs())
        for mode in control.CONTROL_MODES:
            temporal = [value.clone() for value in inputs]
            temporal[3][:, 0, :, 0] = True
            with self.assertRaisesRegex(ValueError, "out-of-time"):
                self.build(temporal, mode)
            geometric = [value.clone() for value in inputs]
            geometric[2][:, 1, 0, 2] = 100
            with self.assertRaisesRegex(ValueError, "out-of-image"):
                self.build(geometric, mode)

    def test_fail_closed_for_shape_dtype_nonfinite_range_and_invalid_config(self):
        inputs = self.inputs()
        bad_cases = []
        for index in (0, 1, 2, 3):
            values = [value.clone() for value in inputs]
            values[index] = values[index][..., :-1]
            bad_cases.append(values)
        for index in (0, 1, 2):
            values = [value.clone() for value in inputs]
            values[index].reshape(-1)[0] = float("nan")
            bad_cases.append(values)
        values = [value.clone() for value in inputs]; values[0][0, 0, 0, 0, 0] = 1.1
        bad_cases.append(values)
        values = [value.clone() for value in inputs]; values[3] = values[3].float(); values[3][0, 0, 0, 2, 0, 0] = .5
        bad_cases.append(values)
        for mode in control.CONTROL_MODES:
            for values in bad_cases:
                with self.assertRaises(ValueError):
                    self.build(values, mode)
            with self.assertRaises(TypeError):
                self.build((inputs[0].double(), *inputs[1:]), mode)
        for mode, offset in (("typo", 2), ("raw_detail", True), ("same_frame_detail", 0),
                             ("mismatched_detail", 7), ("same_frame_detail", 1.5)):
            with self.assertRaises(ValueError):
                self.build(inputs, mode, offset)
        with self.assertRaises(ValueError):
            control.build_observation_controls(None, *inputs[1:], mode="raw_detail")


if __name__ == "__main__":
    unittest.main()
