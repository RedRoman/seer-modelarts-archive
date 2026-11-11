"""Data-free protocol and CPU synthetic-cache tests for the selector probe.

The NumPy tests run without torch. Torch tests exercise feature isolation,
identical initial functions, exact objectives and a complete tiny 48-clip run.
No real data, foundation model, accelerator, downloads or training jobs.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1]


def local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = local_module("_selector_trainer_test", CODE / "tools/train_correction_selector.py")
try:
    import torch
except ImportError:
    torch = None
selector = None if torch is None else local_module("_selector_model_test", CODE / "models/correction_selector.py")
risk = local_module("_selector_risk_test", CODE / "models/correction_risk.py")


class ProtocolTests(unittest.TestCase):
    def test_scene_split_is_fixed_disjoint_and_input_order_independent(self):
        scenes = [f"scene{i:02d}" for i in range(24)]
        split = trainer.make_scene_split(scenes + scenes)
        self.assertEqual(split, trainer.make_scene_split(list(reversed(scenes))))
        expected = np.random.default_rng(20261114).permutation(24)
        self.assertEqual(split["fit"], [scenes[i] for i in expected[:18]])
        self.assertEqual(split["internal_holdout"], [scenes[i] for i in expected[18:]])
        self.assertFalse(set(split["fit"]) & set(split["internal_holdout"]))
        self.assertEqual(set(split["fit"]) | set(split["internal_holdout"]), set(scenes))
        with self.assertRaises(ValueError):
            trainer.make_scene_split(scenes[:-1])

    def test_sample_plan_never_uses_holdout_and_is_model_independent(self):
        fit = [2, 3, 5, 6, 8]
        plan = trainer.make_batch_plan(fit, 1000, 8, 0)
        self.assertEqual(plan.shape, (1000, 8))
        self.assertEqual(set(plan.reshape(-1)), set(fit))
        np.testing.assert_array_equal(plan, trainer.make_batch_plan(fit, 1000, 8, 0))
        self.assertFalse(np.array_equal(plan, trainer.make_batch_plan(fit, 1000, 8, 1)))
        with self.assertRaises(ValueError):
            trainer.make_batch_plan([1, 1], 2, 1, 0)

    def test_normalization_uses_fit_only_and_preserves_zero_target_origin(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(6, 19, 2, 3)).astype(np.float32)
        d = rng.normal(size=(6, 2, 3, 2, 5)) * .1
        gram = np.einsum("...id,...jd->...ij", d, d) / 5
        a = rng.normal(size=(6, 2, 3, 2)) * .01
        fit = [0, 2, 4]
        e0 = rng.uniform(.001, .01, size=(6, 2, 3))
        expected = trainer.fit_normalization(x, a, gram, fit, e0=e0)
        xx, aa, gg, ee = x.copy(), a.copy(), gram.copy(), e0.copy()
        xx[[1, 3, 5]], aa[[1, 3, 5]], gg[[1, 3, 5]], ee[[1, 3, 5]] = np.nan, np.nan, np.nan, np.nan
        self.assertEqual(expected, trainer.fit_normalization(xx, aa, gg, fit, e0=ee))
        self.assertEqual(expected["direct_mse_normalizer"], float(e0[fit].mean()))
        np.testing.assert_allclose(expected["feature_mean"], x[fit].astype(np.float64).mean((0, 2, 3)))
        np.testing.assert_allclose(expected["feature_std"], x[fit].astype(np.float64).std((0, 2, 3)))
        denominator = np.sqrt(np.stack((gram[fit, ..., 0, 0], gram[fit, ..., 1, 1]), -1)).clip(expected["energy_floor"])
        target = (a[fit] - gram[fit].sum(-1) / 3) / denominator
        np.testing.assert_allclose(expected["target_scale"], np.sqrt(np.square(target).mean((0, 1, 2))))

    def test_fit_normalization_handles_zero_energy_and_rejects_bad_indices(self):
        x, a, g = np.zeros((2, 19, 1, 1)), np.zeros((2, 1, 1, 2)), np.zeros((2, 1, 1, 2, 2))
        e0 = np.ones((2, 1, 1))
        stats = trainer.fit_normalization(x, a, g, [0], e0=e0)
        self.assertEqual(stats["energy_floor"], 1e-6)
        self.assertEqual(stats["target_scale"], [1e-6, 1e-6])
        self.assertTrue(all(value == 1e-6 for value in stats["feature_std"]))
        for bad in ([], [0, 0], [-1], [2]):
            with self.assertRaises(ValueError):
                trainer.fit_normalization(x, a, g, bad, e0=e0)
        with self.assertRaisesRegex(ValueError, "mean e0"):
            trainer.fit_normalization(x, a, g, [0], e0=np.zeros_like(e0))

    def test_manifest_rejects_incomplete_or_modified_cache_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            provenance = {"domain": "bvi", "purpose": "TRAIN-only synthetic fixture", "plan": [
                {"sequence_index": i, "augmentation_seed": i} for i in range(48)],
                "base_checkpoint_sha256": "a" * 64, "raft_weights_sha256": "b" * 64,
                "split_sha256": "c" * 64, "base_state_sha256": "d" * 64}
            rows = []
            for i in range(48):
                path = directory / f"clip_{i:03d}.pt"
                path.write_bytes(f"synthetic-unloaded-{i}".encode())
                rows.append({"cache_path": path.name, "cache_sha256": trainer.digest(path)})
            summary = {"n": 48, "base_state_unchanged": True, "per_sequence": rows}
            trainer.write_json(directory / "provenance.json", provenance)
            trainer.write_json(directory / "oracle_summary.json", summary)
            self.assertEqual(len(trainer.cache_manifest(directory)[0]), 48)
            (directory / "clip_012.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                trainer.cache_manifest(directory)
            (directory / "clip_012.pt").unlink()
            with self.assertRaisesRegex(ValueError, "48 completed"):
                trainer.cache_manifest(directory)


def synthetic_cache(directory):
    """Tiny but complete 24-scene/two-condition cache with learnable RGB truth."""
    directory.mkdir()
    generator = torch.Generator().manual_seed(27)
    plan = [{"sequence_index": i, "augmentation_seed": i + 100} for i in range(48)]
    provenance = {"domain": "bvi", "purpose": "TRAIN-only synthetic cache; no real data",
                  "plan": plan, "args": {"patch_size": 4},
                  "base_checkpoint_sha256": "a" * 64, "raft_weights_sha256": "b" * 64,
                  "split_sha256": "c" * 64, "base_state_sha256": "d" * 64}
    rows = []
    for i in range(48):
        scene, condition = f"scene{i // 2:02d}", ("10", "20")[i % 2]
        low = torch.rand(1, 3, 2, 8, 8, generator=generator)
        base = torch.full_like(low, .4)
        directions = torch.zeros(1, 2, 3, 2, 8, 8)
        directions[:, 0, 0], directions[:, 1, 1] = .18, .18
        u0, u1 = .55 + .1 * low[:, 0:1], .05 + .1 * low[:, 1:2]
        gt = base + u0 * directions[:, 0] + u1 * directions[:, 1]
        a, gram = risk.patch_moments(gt - base, directions, 4)
        oracle_weights, _ = risk.solve_two_candidate_risk(a, gram)
        name = f"BVI/{scene}__{condition}fps"
        sample = {"name": name, "scene": scene, "condition": condition, "plan": plan[i], "tensors": {
            "lq": low, "base": base, "gt": gt, "corrections": directions,
            "valid_mask": torch.ones(1, 2, 1, 2, 8, 8, dtype=torch.bool),
            "a": a, "gram": gram, "oracle_weights": oracle_weights}}
        path = directory / f"clip_{i:03d}.pt"
        torch.save(sample, path)
        rows.append({"name": name, "scene": scene, "condition": condition,
                     "cache_path": path.name, "cache_sha256": trainer.digest(path)})
    trainer.write_json(directory / "provenance.json", provenance)
    trainer.write_json(directory / "oracle_summary.json", {"n": 48, "base_state_unchanged": True, "per_sequence": rows})


@unittest.skipIf(torch is None, "torch required for synthetic CPU checks")
class TorchSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_observable_features_match_independent_pixels_and_have_no_gt_input(self):
        torch.manual_seed(6)
        low, base = torch.rand(1, 3, 2, 8, 12), torch.rand(1, 3, 2, 8, 12)
        directions = torch.randn(1, 2, 3, 2, 8, 12) * .1
        valid = torch.ones(1, 2, 1, 2, 8, 12, dtype=torch.bool)
        valid[:, 1, :, :, :, :4] = False
        directions *= valid
        x, gram = selector.observable_patch_features(low, base, directions, valid, 4)
        self.assertEqual(tuple(x.shape), (1, 2, 19, 2, 3))
        for t in range(2):
            for row in range(2):
                for column in range(3):
                    sl = (slice(row * 4, (row + 1) * 4), slice(column * 4, (column + 1) * 4))
                    l, b = low[0, :, t, sl[0], sl[1]], base[0, :, t, sl[0], sl[1]]
                    d = directions[0, :, :, t, sl[0], sl[1]].reshape(2, -1)
                    g = d @ d.T / d.shape[-1]
                    torch.testing.assert_close(gram[0, t, row, column], g)
                    torch.testing.assert_close(x[0, t, :3, row, column], l.mean((1, 2)))
                    torch.testing.assert_close(x[0, t, 12, row, column], l.var((1, 2), unbiased=False).mean())
                    torch.testing.assert_close(x[0, t, 13, row, column], b.var((1, 2), unbiased=False).mean())
        # These forbidden supervision keys cannot be accepted as input kwargs.
        for forbidden in ("gt", "a", "oracle_weights"):
            with self.assertRaises(TypeError):
                selector.observable_patch_features(low, base, directions, valid, 4, **{forbidden: torch.randn(1)})
        poisoned = directions.clone(); poisoned[:, 1, :, :, :, :4] = 1
        with self.assertRaisesRegex(ValueError, "zero wherever"):
            selector.observable_patch_features(low, base, poisoned, valid, 4)

    def test_all_zero_heads_give_exact_thirds_including_singular_gram(self):
        torch.manual_seed(0)
        model = selector.TinyCorrectionSelector()
        state = copy.deepcopy(model.state_dict())
        other = selector.TinyCorrectionSelector(); other.load_state_dict(state)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 6626)
        x = torch.randn(3, 19, 2, 2)
        head = model(x)
        self.assertTrue(torch.equal(head, torch.zeros_like(head)))
        grams = torch.stack((torch.eye(2), torch.ones(2, 2), torch.zeros(2, 2)))[:, None, None].expand(3, 2, 2, 2, 2)
        scale = torch.tensor([.01, .02])
        for mode in ("risk_full", "direct_mse", "direct_mse_normalized"):
            weight = selector.weights_from_head(head, grams, mode, scale, 1e-6)
            torch.testing.assert_close(weight, torch.full_like(weight, 1 / 3), rtol=0, atol=0)
        self.assertEqual(trainer.state_digest(model.state_dict()), trainer.state_digest(other.state_dict()))

    def test_perfect_moment_head_recovers_oracle_and_zero_energy_is_finite(self):
        torch.manual_seed(7)
        directions = torch.randn(4, 2, 30) * .1
        error = torch.randn(4, 30) * .1
        a, gram = risk.train_only_patch_statistics(error, directions)
        scale = torch.tensor([.03, .02])
        target = selector.normalized_moment_target(a, gram, scale, 1e-6)
        w = selector.weights_from_head(target, gram, "risk_full", scale, 1e-6)
        _, q = risk.solve_two_candidate_risk(a, gram)
        torch.testing.assert_close(selector.quadratic_mse(torch.zeros(4), a, gram, w), q, atol=1e-8, rtol=1e-5)
        zero_g, zero_a = torch.zeros(4, 2, 2), torch.zeros(4, 2)
        target = selector.normalized_moment_target(zero_a, zero_g, scale, 1e-6)
        self.assertTrue(torch.isfinite(target).all())
        torch.testing.assert_close(selector.weights_from_head(target, zero_g, "risk_full", scale, 1e-6), torch.full_like(target, 1 / 3))

    def test_direct_objective_and_gradient_equal_actual_pixel_reconstruction(self):
        torch.manual_seed(15)
        error, directions = torch.randn(5, 36) * .1, torch.randn(5, 2, 36) * .1
        a, gram = risk.train_only_patch_statistics(error, directions)
        head = torch.randn(5, 2, requires_grad=True)
        weight = selector.weights_from_head(head, gram, "direct_mse", torch.ones(2), 1e-6)
        normalized_weight = selector.weights_from_head(head, gram, "direct_mse_normalized", torch.ones(2), 1e-6)
        torch.testing.assert_close(normalized_weight, weight, atol=0, rtol=0)
        moment_loss = selector.quadratic_mse(error.square().mean(-1), a, gram, weight).mean()
        pixel_loss = (error - (weight[..., None] * directions).sum(-2)).square().mean()
        torch.testing.assert_close(moment_loss, pixel_loss, atol=1e-8, rtol=1e-6)
        grad_moment = torch.autograd.grad(moment_loss, head, retain_graph=True)[0]
        grad_pixel = torch.autograd.grad(pixel_loss, head)[0]
        torch.testing.assert_close(grad_moment, grad_pixel, atol=1e-9, rtol=1e-5)

    def test_3_by_3_context_has_no_more_distant_input_dependency(self):
        torch.manual_seed(11)
        model = selector.TinyCorrectionSelector()
        torch.nn.init.normal_(model.head.weight)
        x = torch.randn(1, 19, 9, 9, requires_grad=True)
        model(x)[0, 4, 4].sum().backward()
        outside = x.grad.clone(); outside[:, :, 3:6, 3:6] = 0
        self.assertEqual(float(outside.abs().max()), 0.)
        self.assertGreater(float(x.grad[:, :, 3:6, 3:6].abs().sum()), 0.)

    def test_complete_synthetic_cache_fixed_budget_and_end_only_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            synthetic_cache(directory / "cache")
            real_evaluate = trainer.final_holdout_evaluation
            def checked_evaluate(data, states, *args):
                self.assertEqual(set(states), {"risk_full", "direct_mse", "direct_mse_normalized"})
                for mode in states:
                    self.assertTrue((directory / "run" / f"selector_{mode}.pt").is_file())
                return real_evaluate(data, states, *args)
            with mock.patch.object(trainer, "final_holdout_evaluation", side_effect=checked_evaluate) as evaluate:
                report = trainer.run_experiment(directory / "cache", directory / "run", steps=24,
                                                lr=1e-2, seed=0, batch_frames=8, device_name="cpu")
                self.assertEqual(evaluate.call_count, 1)
            self.assertEqual((report["scenes"], report["clips"], report["frames"]), (6, 12, 24))
            provenance = json.loads((directory / "run/provenance.json").read_text())
            fit, held = set(provenance["split"]["fit"]), set(provenance["split"]["internal_holdout"])
            self.assertEqual((len(fit), len(held), len(fit & held)), (18, 6, 0))
            plan = json.loads((directory / "run/sample_plan.json").read_text())
            self.assertEqual(len(plan["batches"]), 24)
            self.assertTrue(all(plan["frame_index"][index]["scene"] in fit for batch in plan["batches"] for index in batch))
            first_mses, first_losses = [], {}
            for mode in ("risk_full", "direct_mse", "direct_mse_normalized"):
                curve = [json.loads(line) for line in (directory / f"run/curve_{mode}.jsonl").read_text().splitlines()]
                self.assertEqual(len(curve), 24)
                self.assertLess(curve[-1]["loss"], curve[0]["loss"])
                first_mses.append(curve[0]["batch_rgb_mse"])
                first_losses[mode] = curve[0]["loss"]
                checkpoint = torch.load(directory / f"run/selector_{mode}.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["steps"], 24)
                self.assertEqual(checkpoint["initial_state_sha256"], provenance["initial_state_sha256"])
                self.assertEqual(checkpoint["sample_plan_sha256"], provenance["sample_plan_sha256"])
                self.assertEqual(checkpoint["provenance"], provenance)
                self.assertGreater(checkpoint["normalization"]["direct_mse_normalizer"], 0)
                self.assertEqual(checkpoint["provenance"]["cache_provenance"]["base_checkpoint_sha256"], "a" * 64)
                self.assertEqual(checkpoint["provenance"]["cache_provenance"]["raft_weights_sha256"], "b" * 64)
                self.assertEqual(checkpoint["provenance"]["cache_provenance"]["split_sha256"], "c" * 64)
                sidecar = json.loads((directory / f"run/selector_{mode}.sha256.json").read_text())
                self.assertEqual(sidecar["checkpoint_sha256"], trainer.digest(directory / f"run/selector_{mode}.pt"))
                self.assertEqual(sidecar["model_state_sha256"], trainer.state_digest(checkpoint["model"]))
                self.assertEqual(checkpoint["model_state_sha256"], sidecar["model_state_sha256"])
                self.assertLess(report["methods"][mode]["rgb_mse"], report["methods"]["initial_thirds"]["rgb_mse"])
            self.assertAlmostEqual(first_mses[0], first_mses[1], places=12)
            self.assertAlmostEqual(first_mses[0], first_mses[2], places=12)
            self.assertAlmostEqual(first_losses["direct_mse_normalized"],
                                   first_losses["direct_mse"] / provenance["normalization"]["direct_mse_normalizer"], places=6)
            for method in report["methods"].values():
                self.assertEqual(len(method["per_scene"]), 6)
                self.assertEqual(method["rgb_elements"], 12 * 2 * 3 * 8 * 8)
                self.assertAlmostEqual(method["rgb_mse"], method["rgb_squared_error_sum"] / method["rgb_elements"], places=12)
                self.assertGreaterEqual(method["rgb_mse"] + 1e-8, report["methods"]["gt_oracle"]["rgb_mse"])


if __name__ == "__main__":
    unittest.main()
