"""Synthetic color/FIT-lineage tests; no real data or quality result is read."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rgb_fit as run


class RgbFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.math, cls.color, _ = run.load_dependencies()
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        shape = (1, 3, 16, 256, 256)
        low = torch.tensor((.02, .10, .15)).view(1, 3, 1, 1, 1).expand(shape).contiguous()
        gt = torch.tensor((.125, .375, .75)).view(1, 3, 1, 1, 1).expand(shape).contiguous()
        cls.expected_gt = gt[:, [2, 1, 0]].contiguous()
        cls.base = cls.expected_gt + .01
        source = {"name": "FIT/a", "scene": "FIT/a", "tensors": {"lq": low, "gt": gt,
            "flow": torch.zeros(1, 2, 2, 16, 256, 256),
            "valid_mask": torch.ones(1, 2, 1, 16, 256, 256, dtype=torch.bool)}}
        source_path = cls.root / "source.pt"
        torch.save(source, source_path)
        lineage = {"source_cache_sha256": run.digest(source_path), "source_provenance_sha256": run.SOURCE_SHA,
            "weight_sha256": run.WEIGHT_SHA, "model_state_sha256": run.STATE_SHA,
            "channel_boundary": copy.deepcopy(run.BOUNDARY), "input_reencoded_to_uint8": False,
            "native_lq_tensor_sha256": run.tensor_digest(low),
            "rgb_lq_tensor_sha256": run.tensor_digest(low[:, [2, 1, 0]]),
            "base_tensor_sha256": run.tensor_digest(cls.base)}
        cls.hvi = {"name": "FIT/a", "scene": "FIT/a", "base": cls.base, "lineage": lineage}
        hvi_path = cls.root / "hvi.pt"
        torch.save(cls.hvi, hvi_path)
        cls.row = {"name": "FIT/a", "scene": "FIT/a", "source_path": str(source_path),
            "source_sha256": run.digest(source_path), "hvi_path": str(hvi_path),
            "hvi_sha256": run.digest(hvi_path), **{k: lineage[k] for k in (
                "native_lq_tensor_sha256", "rgb_lq_tensor_sha256", "base_tensor_sha256")}}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_real_serialized_loader_swaps_native_gt_exactly_once_and_preserves_rgb_hvi(self):
        (base, flow, mask), gt = run.load_fit_sample(self.row, ["FIT/a"], self.color)
        self.assertTrue(torch.equal(gt, self.expected_gt))
        self.assertTrue(torch.equal(base, self.base))
        self.assertFalse(torch.equal(base, self.base[:, [2, 1, 0]]))
        self.assertGreater(float((base - gt[:, [2, 1, 0]]).square().mean()), .1)
        self.assertLess(float((base - gt).square().mean()), .001)
        self.assertEqual(flow.shape, (1, 2, 2, 16, 256, 256))
        self.assertTrue(bool(mask.all()))

    def test_rgb_mse_is_invariant_under_common_channel_permutation_for_all_fixed_predictions(self):
        generator = torch.Generator().manual_seed(2718)
        base = .1 + .7 * torch.rand(1, 3, 5, 8, 8, generator=generator)
        labels = torch.rand(1, 3, 5, 8, 8, generator=generator)
        flow = torch.zeros(1, 2, 2, 5, 8, 8)
        mask = torch.ones(1, 2, 1, 5, 8, 8, dtype=torch.bool)
        predictions, _ = self.math.fixed_predictions(base, flow, mask)
        score = self.math.score_predictions(predictions, labels)
        permuted = self.math.score_predictions({k: v[:, [2, 1, 0]] for k, v in predictions.items()},
                                                labels[:, [2, 1, 0]])
        self.assertEqual(set(predictions), set(run.METHODS))
        for name in run.METHODS:
            self.assertAlmostEqual(score[name]["rgb_mse_float"], permuted[name]["rgb_mse_float"], 14)
            self.assertAlmostEqual(score[name]["psnr_rgb_from_clip_mse"], permuted[name]["psnr_rgb_from_clip_mse"], 12)
        for value in (.25, .5, 1.):
            self.assertTrue(torch.equal(predictions[f"sameframe_identity_lambda_{value:g}"], base))

    def test_excluded_scene_is_rejected_before_digest_or_tensor_loading(self):
        with patch.object(run, "digest", side_effect=AssertionError("No archive byte read")) as sha, \
             patch("torch.load", side_effect=AssertionError("No tensor load")) as loader:
            with self.assertRaisesRegex(ValueError, "excluded"):
                run.load_fit_sample({"name": "held/a", "scene": "held"}, ["fit"], self.color)
            sha.assert_not_called()
            loader.assert_not_called()

    def test_evaluator_filters_excluded_rows_before_calling_loader(self):
        seen = []
        rows = [{"name": "FIT/a", "scene": "FIT/a"}, {"name": "held/a", "scene": "held"}]
        def loader(row, fit):
            seen.append(row["name"])
            return run.load_fit_sample(self.row, fit, self.color)
        result = run.evaluate_fit_rows(rows, {"fit": ["FIT/a"], "internal_holdout": ["held"]}, loader, self.math)
        self.assertEqual(seen, ["FIT/a"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["gt_color"], "RGB")
        self.assertEqual(set(result[0]["methods"]), set(run.METHODS))

    def test_manifest_rejects_legacy_unswapped_or_incomplete_hvi(self):
        good = {"complete": True, "input_color": "RGB", "native_source_color": "BGR",
                "output_color": "RGB", "channel_boundary": copy.deepcopy(run.BOUNDARY)}
        run.check_color_metadata(good)
        for key, value in (("complete", False), ("input_color", "BGR"), ("output_color", "BGR"),
                           ("native_source_color", "RGB"), ("channel_boundary", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.check_color_metadata({**good, key: value})

    def test_payload_wrong_native_input_hash_is_rejected(self):
        hvi = copy.deepcopy(self.hvi)
        hvi["lineage"]["rgb_lq_tensor_sha256"] = "0" * 64
        path = self.root / "bad_hvi.pt"
        torch.save(hvi, path)
        row = {**self.row, "hvi_path": str(path), "hvi_sha256": run.digest(path)}
        with self.assertRaisesRegex(ValueError, "tensor SHA"):
            run.load_fit_sample(row, ["FIT/a"], self.color)

    def test_payload_wrong_color_boundary_is_rejected_even_with_updated_archive_hash(self):
        hvi = copy.deepcopy(self.hvi)
        hvi["lineage"]["channel_boundary"]["input_permutation"] = [0, 1, 2]
        path = self.root / "bad_boundary.pt"
        torch.save(hvi, path)
        row = {**self.row, "hvi_path": str(path), "hvi_sha256": run.digest(path)}
        with self.assertRaisesRegex(ValueError, "provenance"):
            run.load_fit_sample(row, ["FIT/a"], self.color)

    def test_prediction_math_receives_no_gt_and_dependency_mutation_fails_before_execution(self):
        self.assertFalse({"gt", "labels", "fit_labels"} & set(inspect.signature(self.math.fixed_predictions).parameters))
        path = self.root / "bad_dependency.py"
        path.write_text("raise AssertionError('must not execute')\n")
        with self.assertRaisesRegex(ValueError, "SHA"):
            run.module_at("_bad", path, run.MATH_SHA)

    def test_unclosed_raw_contract_is_rejected_even_if_digest_field_is_claimed(self):
        path = self.root / "receipt.json"
        path.write_text(json.dumps({"verified": True, "native_source_color": "BGR",
            "source_provenance_sha256": run.SOURCE_SHA, "status": "draft", "raw_disk_byte_identity_verified": True}))
        with patch.object(run, "digest", return_value=run.RECEIPT_SHA):
            with self.assertRaisesRegex(ValueError, "closed"):
                run.check_bgr_evidence(path, run.RECEIPT_SHA)

    def test_global_metadata_validation_does_not_open_any_tensor_archives(self):
        folder = self.root / "metadata_only"
        source, hvi = folder / "source", folder / "hvi"
        source.mkdir(parents=True)
        hvi.mkdir()
        scenes = [f"SDSD_outdoor/MVI_{i:04d}" for i in range(68)]
        names = [n for scene in scenes for n in (scene, scene + "_2")]
        order = np.random.default_rng(20261114).permutation(68)
        split = {"seed": 20261114, "fit": [scenes[int(i)] for i in order[:51]],
                 "internal_holdout": [scenes[int(i)] for i in order[51:]]}
        origin = {"source_names": names, "raft_state_sha256": "a" * 64}
        sr, hr, gr = [], [], []
        for i, name in enumerate(names):
            common = {"name": name, "scene": scenes[i // 2]}
            sr.append({**common, "cache_path": f"clip_{i:03d}.pt", "cache_sha256": "b" * 64})
            hr.append({**common, "file": f"hvi_rgb_{i:03d}.pt", "source_cache_sha256": "b" * 64,
                "file_sha256": "c" * 64, "base_tensor_sha256": "d" * 64,
                "native_lq_tensor_sha256": "e" * 64, "rgb_lq_tensor_sha256": "f" * 64,
                "shape": [1, 3, 16, 256, 256], "dtype": "float32", "finite_range_verified": True})
            gr.append({**common, "source_sha256": "b" * 64,
                       "source_path": str((source / f"clip_{i:03d}.pt").resolve())})
        manifest = {"complete": True, "input_color": "RGB", "native_source_color": "BGR", "output_color": "RGB",
            "channel_boundary": run.BOUNDARY, "bgr_evidence_file": "unused_receipt.json", "bgr_evidence_sha256": run.RECEIPT_SHA,
            "n": 136, "scenes": 68, "model_state_unchanged": True, "weight_sha256": run.WEIGHT_SHA,
            "model_state_sha256": run.STATE_SHA, "source_provenance_sha256": run.SOURCE_SHA,
            "source_summary_sha256": run.SUMMARY_SHA, "gt_storage_entries_read": 0, "target_inputs_read": 0,
            "quality_scores_computed": False, "per_sequence": hr}
        for path, payload in ((source / "provenance.json", origin), (source / "cache_summary.json", {"per_sequence": sr}),
                              (hvi / "manifest.json", manifest), (folder / "geometry.json", {
                                  "source_cache_provenance": origin, "n": 136, "per_sequence": gr, "split": split})):
            path.write_text(json.dumps(payload))
        def json_digest(path):
            path = Path(path)
            self.assertEqual(path.suffix, ".json", "Tensor archive opened during global metadata validation")
            return {"provenance.json": run.SOURCE_SHA, "cache_summary.json": run.SUMMARY_SHA,
                    "manifest.json": "h" * 64, "geometry.json": run.GEOMETRY_SHA}[path.name]
        with patch.object(run, "digest", side_effect=json_digest), patch.object(run, "check_bgr_evidence", return_value={}):
            rows, checked_split, lineage = run.bound_inputs(source, hvi, folder / "geometry.json", "h" * 64)
        self.assertEqual(len(rows), 136)
        self.assertEqual(len(run.selected_fit_rows(rows, checked_split)), 102)
        self.assertEqual(lineage["excluded_tensor_archives_opened"], 0)
        self.assertFalse(list(source.glob("*.pt")))
        self.assertFalse(list(hvi.glob("*.pt")))


if __name__ == "__main__":
    unittest.main()
