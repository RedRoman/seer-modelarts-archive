"""Source-plan isolation and full-context/loss-only crop contracts.

All data here are synthetic. Tensor tests require the remote CPU environment;
local skips must not be reported as tensor-test passes.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "tools"))
spec = importlib.util.spec_from_file_location(
    "_test_train_observation_geometry", CODE / "tools/train_observation_geometry.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

try:
    import torch
except ImportError:
    torch = None


class GeometryPlanTests(unittest.TestCase):
    def names(self):
        scenes = ([f"SDSD_indoor/pair{i}" for i in range(23)]
                  + [f"SDSD_outdoor/MVI_{i:04d}" for i in range(45)])
        return [scene + suffix for scene in scenes for suffix in ("", "_2")]

    def test_fixed_plan_uses_only_fit_scenes_and_declared_budget(self):
        names = self.names()
        split = runner.make_scene_split(names)
        self.assertEqual((len(split["fit"]), len(split["internal_holdout"])), (51, 17))
        self.assertFalse(set(split["fit"]) & set(split["internal_holdout"]))
        rng_before = np.random.get_state()
        plan = runner.make_plan(names, split, 0)
        rng_after = np.random.get_state()
        self.assertEqual(rng_before[0], rng_after[0])
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        self.assertEqual(rng_before[2:], rng_after[2:])
        self.assertEqual(plan, runner.make_plan(names, split, 0))
        self.assertNotEqual(plan, runner.make_plan(names, split, 1))
        self.assertEqual((plan["steps"], len(plan["batches"])), (400, 400))
        self.assertEqual((plan["crop"], plan["patch"], plan["loss_frames"]), (128, 16, 8))
        for row in plan["batches"]:
            scene = runner.source_scene(names[row["clip_index"]])
            self.assertIn(scene, split["fit"])
            self.assertNotIn(scene, split["internal_holdout"])
            self.assertEqual(len(row["frames"]), 8)
            self.assertEqual(len(set(row["frames"])), 8)
            self.assertTrue(all(type(t) is int and 0 <= t < 16 for t in row["frames"]))
            self.assertTrue(all(type(row[k]) is int and 0 <= row[k] <= 128
                                and row[k] % 16 == 0 for k in ("top", "left")))

    def test_changed_or_leaking_scene_split_is_rejected(self):
        names = self.names()
        split = runner.make_scene_split(names)
        for mutation in ("leak", "seed", "order"):
            changed = copy.deepcopy(split)
            if mutation == "leak":
                changed["fit"].append(changed["internal_holdout"][0])
            elif mutation == "seed":
                changed["seed"] += 1
            else:
                changed["fit"].reverse()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.make_plan(names, changed, 0)


@unittest.skipIf(torch is None, "Torch required; run real tensor tests on remote CPU")
class GeometryTensorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        def value(shape, number):
            return torch.tensor(number, dtype=torch.float32).expand(shape)
        tensors = {
            "lq": value((1, 3, 16, 256, 256), .1),
            "base": value((1, 3, 16, 256, 256), .3),
            "flow": value((1, 2, 2, 16, 256, 256), .2),
            "valid_mask": torch.ones(1, dtype=torch.bool).expand(1, 2, 1, 16, 256, 256),
            "gt": value((1, 3, 16, 256, 256), .4),
        }
        identity = {"name": "SDSD_indoor/pair1", "scene": "SDSD_indoor/pair1"}
        raw = dict(identity, tensors=tensors)
        memory = dict(identity,
            prior_address=value((1, 16, 256, 64), .5),
            pixel_address=value((1, 16, 256, 64), .6))
        row = dict(identity, source_path="source.pt", address_path="address.pt")
        return row, raw, memory

    def test_load_sample_retains_all_context_and_has_no_gt_argument(self):
        row, raw, memory = self.fixture()
        with mock.patch.object(torch, "load", side_effect=[raw, memory]):
            inputs, gt = runner.load_sample(row)
        expected = tuple(raw["tensors"][k] for k in ("lq", "base", "flow", "valid_mask")) + (
            memory["prior_address"], memory["pixel_address"])
        self.assertEqual(len(inputs), 6)
        for actual, original in zip(inputs, expected):
            self.assertIs(actual, original, "Module inputs must retain full cached context")
            self.assertIsNot(actual, gt)
        self.assertEqual(inputs[0].shape, (1, 3, 16, 256, 256))
        self.assertEqual(inputs[-1].shape, (1, 16, 256, 64))
        self.assertIs(gt, raw["tensors"]["gt"])
        changed = dict(raw, tensors=dict(raw["tensors"], gt=torch.zeros_like(gt)))
        with mock.patch.object(torch, "load", side_effect=[changed, memory]):
            other_inputs, other_gt = runner.load_sample(row)
        self.assertTrue(all(a is b for a, b in zip(inputs, other_inputs)))
        self.assertFalse(torch.equal(gt, other_gt), "GT change must alter labels only")

    def test_load_sample_rejects_wrong_source_or_address_identity(self):
        row, raw, memory = self.fixture()
        for which in (0, 1):
            for field in ("name", "scene"):
                payloads = [dict(raw), dict(memory)]
                payloads[which][field] = "SDSD_indoor/wrong"
                with self.subTest(which=which, field=field):
                    with mock.patch.object(torch, "load", side_effect=payloads):
                        with self.assertRaises(ValueError):
                            runner.load_sample(row)

    def spec(self):
        return {"top": 128, "left": 64, "frames": [15, 0, 9, 3, 2, 8, 6, 14]}

    def test_loss_crop_uses_exact_spatial_and_frame_coordinates_and_gradients(self):
        tt = torch.arange(16, dtype=torch.float32)[None, None, :, None, None]
        yy = torch.arange(256, dtype=torch.float32)[None, None, None, :, None]
        xx = torch.arange(256, dtype=torch.float32)[None, None, None, None, :]
        video = (tt * 1_000_000 + yy * 1000 + xx).expand(1, 3, 16, 256, 256)
        spec = self.spec()
        crop = runner.loss_view(video, spec)
        self.assertEqual(crop.shape, (1, 3, 8, 128, 128))
        for j, frame in enumerate(spec["frames"]):
            self.assertEqual(float(crop[0, 2, j, 0, 0]), frame * 1_000_000 + 128_000 + 64)
            self.assertEqual(float(crop[0, 2, j, -1, -1]), frame * 1_000_000 + 255_000 + 191)
        variable = torch.zeros(1, 3, 16, 256, 256, requires_grad=True)
        runner.loss_view(variable, spec).sum().backward()
        expected = torch.zeros_like(variable)
        expected[:, :, spec["frames"], 128:256, 64:192] = 1
        self.assertTrue(torch.equal(variable.grad, expected))

    def test_loss_view_rejects_illegal_frames_and_crop_coordinates(self):
        video = torch.zeros(1).expand(1, 3, 16, 256, 256)
        bad_frames = (list(range(7)), [0] * 8, [0, 1, 2, 3, 4, 5, 6, 16],
                      [-1, 1, 2, 3, 4, 5, 6, 7], [0., 1, 2, 3, 4, 5, 6, 7],
                      [False, 1, 2, 3, 4, 5, 6, 7])
        for frames in bad_frames:
            spec = dict(self.spec(), frames=frames)
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                runner.loss_view(video, spec)
        for key in ("top", "left"):
            for coordinate in (-16, 8, 144, 16., False):
                spec = dict(self.spec(), **{key: coordinate})
                with self.subTest(key=key, coordinate=coordinate), self.assertRaises(ValueError):
                    runner.loss_view(video, spec)

    def test_loss_view_rejects_partial_or_empty_context(self):
        for shape in ((1, 3, 16, 128, 256), (1, 3, 16, 256, 192),
                      (1, 3, 15, 256, 256), (1, 1, 16, 256, 256),
                      (0, 3, 16, 256, 256), (3, 16, 256, 256)):
            video = torch.empty(shape) if 0 in shape else torch.zeros(1).expand(shape)
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                runner.loss_view(video, self.spec())
        with self.assertRaises(ValueError):
            runner.loss_view(np.zeros((1, 3, 16, 256, 256), dtype=np.float32), self.spec())

    def test_synthetic_train_step_passes_full_inputs_and_crops_only_loss(self):
        row, raw, memory = self.fixture()
        inputs = tuple(raw["tensors"][k] for k in ("lq", "base", "flow", "valid_mask")) + (
            memory["prior_address"], memory["pixel_address"])
        calls = []

        class FakeGeometry(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.delta = torch.nn.Parameter(torch.tensor(0.))
                self.node = torch.nn.Linear(1, 1).requires_grad_(False)

            def get_config(self):
                return {"synthetic_test_only": True}

            def forward(self, *actual):
                calls.append(actual)
                return {"pred_pixels": actual[1] + self.delta, "aux": {}}

        fake_module = type("FakeModule", (), {"ObservationGeometry": FakeGeometry})
        plan = {"steps": 1, "batches": [dict(self.spec(), clip_index=0)]}
        meta = {"per_sequence": [row], "split": {"fit": [row["scene"]]},
                "reference_head_path": "reference.pt", "reference_head_sha256": "synthetic",
                "normalization": {}, "fit_only_mse_normalizer": .01}
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary) / "prepared"
            prepared.mkdir()
            (prepared / "manifest.json").write_text("{}\n")
            out = Path(temporary) / "run"

            def create_output(path):
                Path(path).mkdir()
                return Path(path)

            with (mock.patch.object(runner, "STEPS", 1),
                  mock.patch.object(runner, "checked_metadata", return_value=meta),
                  mock.patch.object(runner, "new_directory", side_effect=create_output),
                  mock.patch.object(runner, "make_plan", return_value=plan),
                  mock.patch.object(runner, "reference_checkpoint", return_value={"model": {}}),
                  mock.patch.object(runner, "geometry_module", return_value=fake_module),
                  mock.patch.object(runner, "source_hashes", return_value={}),
                  mock.patch.object(runner, "load_sample", return_value=(inputs, raw["tensors"]["gt"]))):
                runner.train(prepared, out, "prior", 0)
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(actual is original for actual, original in zip(calls[0], inputs)))
            self.assertEqual(calls[0][0].shape[-3:], (16, 256, 256))
            self.assertEqual(calls[0][-1].shape[1:3], (16, 256))
            complete = json.loads((out / "complete.json").read_text())
            self.assertTrue(complete["frozen_node_unchanged"])
            checkpoint = torch.load(out / "final.pt", weights_only=True)
            self.assertGreater(float(checkpoint["model"]["delta"]), 0)
            curve = [json.loads(line) for line in (out / "curve.jsonl").read_text().splitlines()]
            self.assertEqual(len(curve), 1)
            self.assertAlmostEqual(curve[0]["rgb_mse"], .01, places=6)


if __name__ == "__main__":
    unittest.main()
