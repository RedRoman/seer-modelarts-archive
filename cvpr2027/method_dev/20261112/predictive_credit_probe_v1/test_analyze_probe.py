#!/usr/bin/env python3
"""Analytic, schema-firewall, and aggregation tests; no Torch or research data."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_probe as ap


def subset_fixture():
    scenes = [f"SDSD_indoor/pair{i}" for i in range(12)]
    return {"schema": ap.SUBSET_SCHEMA, "target_inputs_read": False, "internal_inputs_read": False,
            "split": {"fit": scenes, "internal_holdout": ["SDSD_outdoor/internal"]},
            "entries": [{"scene": scene, "name": scene, "rgb_lq_tensor_sha256": "a" * 64,
                         "source_sha256": "b" * 64}
                        for scene in scenes], "spatial_grid": [1, 2], "target_frames": [8, 9],
            "donor_frames": [[6, 7], [10, 11]]}


def patch_fixture(*, valid=(1.0, 1.0), mse=None, losses=None):
    losses = losses or {"empty": 2.0, "donor1": 1.6, "donor2": 1.8, "both": 1.5}
    mse = mse or {"empty": 0.4, "donor1": 0.3, "donor2": 0.2, "both": 0.25}
    return {"valid_fraction": list(valid), "trained": ap.credit(losses),
            "random": ap.credit(losses), "wrong_time": ap.credit(losses),
            "utility": ap.utility(mse), "controls": {"ema_cosine": [0.6, 0.3],
                "negative_lq_mse": [-0.1, -0.2], "negative_hvi_mse": [-0.2, -0.3]}}


def raw_fixtures():
    subset = subset_fixture()
    critics, values = [], []
    for entry in subset["entries"]:
        for patch in range(2):
            identity = {"name": entry["name"], "scene": entry["scene"], "patch": patch,
                        "split_role": "source_fit"}
            loss = {"empty": 2.0, "donor1": 1.6, "donor2": 1.8, "both": 1.5}
            critics.append({**identity, "trained": dict(loss), "random": dict(loss), "wrong_time": dict(loss),
                            "controls": {"ema_cosine": [0.6, 0.3], "negative_lq_mse": [-0.1, -0.2],
                                         "negative_hvi_mse": [-0.2, -0.3]}})
            values.append({**identity, "valid_fraction": [0.8, 0.7],
                           "mse": {"empty": 0.4, "donor1": 0.3, "donor2": 0.2, "both": 0.25}})
    return subset, critics, values


class RankAndBootstrapTests(unittest.TestCase):
    def test_auc_analytic(self):
        self.assertEqual(ap.auroc([False, False, True, True], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(ap.auroc([False, True], [2.0, 1.0]), 0.0)
        self.assertEqual(ap.auroc([False, True], [1.0, 2.0]), 1.0)

    def test_auc_ties_midrank(self):
        self.assertEqual(ap.midranks([2, 1, 2, 2]), [3.0, 1.0, 3.0, 3.0])
        self.assertEqual(ap.auroc([False, True], [1.0, 1.0]), 0.5)
        self.assertEqual(ap.auroc([False, False, True, True], [0, 1, 1, 2]), 0.875)

    def test_auc_single_class_is_missing(self):
        self.assertIsNone(ap.auroc([], []))
        self.assertIsNone(ap.auroc([True, True], [1, 2]))
        self.assertIsNone(ap.auroc([False, False], [1, 2]))

    def test_auc_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            ap.auroc([0, 1], [1, 2])
        with self.assertRaises(ValueError):
            ap.auroc([False, True], [1, float("nan")])
        with self.assertRaises(ValueError):
            ap.auroc([False], [1, 2])

    def test_spearman_ties_and_constants(self):
        self.assertAlmostEqual(ap.spearman([1, 2, 3], [9, 8, 7]), -1.0)
        self.assertEqual(ap.spearman([1, 1, 2], [2, 2, 3]), 1.0)
        self.assertIsNone(ap.spearman([1, 1], [2, 3]))

    def test_bootstrap_constant_scene_difference(self):
        result = ap.bootstrap_mean([0.25] * 12, repeats=100)
        self.assertEqual(result["ci95"], [0.25, 0.25])
        self.assertEqual(result["resampling_unit"], "complete_scene")

    def test_bootstrap_seed_and_linear_percentiles(self):
        self.assertEqual(ap.bootstrap_mean([0.1, -0.2, 0.3], repeats=100),
                         ap.bootstrap_mean([0.1, -0.2, 0.3], repeats=100))
        self.assertEqual(ap.percentile([0, 10], 0.025), 0.25)
        self.assertEqual(ap.percentile([0, 10], 0.975), 9.75)


class CreditAndBudgetTests(unittest.TestCase):
    def test_conditional_and_shapley_identities(self):
        result = ap.credit({"empty": 2.0, "donor1": 1.5, "donor2": 1.8, "both": 1.4})
        self.assertAlmostEqual(result["conditional"][0], 0.4)
        self.assertAlmostEqual(result["conditional"][1], 0.1)
        self.assertAlmostEqual(sum(result["phi"]), result["v"][2])
        self.assertAlmostEqual(result["interaction"], -0.1)

    def test_nonnegative_uncapped_waterfill_degenerates_to_shapley(self):
        result = ap.credit({"empty": 2.0, "donor1": 1.5, "donor2": 1.8, "both": 1.4})
        self.assertTrue(result["nonnegative_uncapped_degeneracy"])
        for actual, expected in zip(result["alpha"], result["phi"]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["anchor_alpha"], 0.4)

    def test_zero_total_value_prevents_positive_donor_spending(self):
        result = ap.credit({"empty": 1.0, "donor1": 0.0, "donor2": 2.0, "both": 1.5})
        self.assertTrue(result["zero_budget"])
        self.assertTrue(result["single_negative_phi"])
        self.assertGreater(result["phi"][0], 0.0)
        self.assertEqual(result["alpha"], [0.0, 0.0])
        self.assertEqual(result["anchor_alpha"], 1.0)

    def test_total_cap_and_donor_caps(self):
        result = ap.credit({"empty": 2.0, "donor1": 1.7, "donor2": 1.2, "both": 0.9})
        self.assertTrue(result["v12_ge_one"])
        self.assertEqual(result["budget"], 1.0)
        self.assertAlmostEqual(sum(result["alpha"]), 1.0)
        for weight, cap in zip(result["alpha"], result["caps"]):
            self.assertGreaterEqual(weight, -1e-12)
            self.assertLessEqual(weight, cap + 1e-12)
        self.assertAlmostEqual(result["alpha"][0], 0.3)
        self.assertAlmostEqual(result["alpha"][1], 0.7)

    def test_utility_is_incremental_given_other(self):
        result = ap.utility({"empty": 0.4, "donor1": 0.3, "donor2": 0.2, "both": 0.25})
        self.assertAlmostEqual(result["conditional"][0], -0.05)
        self.assertAlmostEqual(result["conditional"][1], 0.05)


class SceneAggregationTests(unittest.TestCase):
    def test_ineligible_patches_and_zero_utility_are_excluded(self):
        tie = patch_fixture(mse={"empty": 0.4, "donor1": 0.3, "donor2": 0.2, "both": 0.2})
        row = ap.scene_statistics("scene", [patch_fixture(valid=(0.49, 1)), tie])
        self.assertEqual(row["eligible_patches"], 1)
        self.assertEqual(row["ineligible_patches"], 1)
        self.assertEqual(row["excluded_zero_utility_donors"], 1)
        self.assertEqual(row["donor_labels"], 1)
        self.assertIsNone(row["auc"][ap.CANDIDATE])

    def test_eligibility_threshold_is_inclusive(self):
        row = ap.scene_statistics("scene", [patch_fixture(valid=(0.5, 0.5))])
        self.assertEqual(row["eligible_patches"], 1)
        self.assertEqual(row["positive_donor_labels"], 1)
        self.assertEqual(row["negative_donor_labels"], 1)

    def test_no_eligible_data_is_missing_not_half_auc(self):
        row = ap.scene_statistics("scene", [patch_fixture(valid=(0.0, 0.0))])
        self.assertIsNone(row["auc"][ap.CANDIDATE])
        self.assertIsNone(row["budget_diagnostic"]["zero_budget_fraction"])

    def test_all_five_baselines_must_be_beaten(self):
        scenes = [{"scene": str(i), "auc": {ap.CANDIDATE: 0.7, **{b: 0.5 for b in ap.BASELINES}}}
                  for i in range(12)]
        self.assertEqual(ap.primary_gate(scenes, repeats=20)["status"], "pass")
        for row in scenes:
            row["auc"]["wrong_time_conditional"] = 0.8
        self.assertEqual(ap.primary_gate(scenes, repeats=20)["status"], "fail")

    def test_macro_scene_auc_not_pooled_by_patch_count(self):
        # Eleven small scenes cannot be outweighed by one large scene's patch count.
        scenes = [{"scene": str(i), "patches": 1 if i < 11 else 100000,
                   "auc": {ap.CANDIDATE: 0.8 if i < 11 else 0.2,
                           **{b: 0.5 for b in ap.BASELINES}}} for i in range(12)]
        gate = ap.primary_gate(scenes, repeats=20)
        self.assertAlmostEqual(gate["candidate_macro_auc"], 0.75)

    def test_shared_complete_scene_intersection_and_minimum_eight(self):
        scenes = [{"scene": str(i), "auc": {ap.CANDIDATE: 0.7, **{b: 0.5 for b in ap.BASELINES}}}
                  for i in range(12)]
        for i, baseline in enumerate(ap.BASELINES):
            scenes[i]["auc"][baseline] = None
        result = ap.primary_gate(scenes, repeats=20)
        self.assertEqual(result["status"], "not_evaluable")
        self.assertEqual(result["candidate_defined_scenes"], 7)
        for comparison in result["comparisons"].values():
            self.assertEqual(comparison["matched_scene_names"], [str(i) for i in range(5, 12)])


class GuardAndProvenanceTests(unittest.TestCase):
    def test_complete_source_fit_join(self):
        subset, critics, values = raw_fixtures()
        grouped = ap.join_records(subset, critics, values)
        self.assertEqual(len(grouped), 12)
        self.assertTrue(all(len(rows) == 2 for rows in grouped.values()))

    def test_no_target_or_internal_records(self):
        subset, critics, values = raw_fixtures()
        for bad_scene in ("BVI_lowlight/scene01", "SDSD_outdoor/internal"):
            bad = copy.deepcopy(critics)
            bad[0]["scene"] = bad_scene
            with self.assertRaises(ValueError):
                ap.join_records(subset, bad, values)

    def test_subset_fit_guard(self):
        subset = subset_fixture()
        subset["entries"][0]["scene"] = "SDSD_outdoor/internal"
        with self.assertRaises(ValueError):
            ap.validate_subset(subset)
        subset = subset_fixture()
        subset["target_inputs_read"] = True
        with self.assertRaises(ValueError):
            ap.validate_subset(subset)

    def test_duplicate_missing_and_wrong_role_rejected(self):
        subset, critics, values = raw_fixtures()
        with self.assertRaises(ValueError):
            ap.join_records(subset, critics + [critics[0]], values)
        with self.assertRaises(ValueError):
            ap.join_records(subset, critics[:-1], values)
        critics[0]["split_role"] = "internal_holdout"
        with self.assertRaises(ValueError):
            ap.join_records(subset, critics, values)

    def test_freeze_checks_bytes_before_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset, protocol = root / "subsets.json", root / "PLAN.md"
            subset.write_text(json.dumps(subset_fixture()))
            protocol.write_text("synthetic registered protocol\n")
            bound = root / "producer.py"
            bound.write_text("# fixture\n")
            frozen = root / "checks" / "analysis_frozen_manifest.json"
            ap.freeze_sources(subset, protocol, [bound], frozen)
            self.assertGreaterEqual(ap.verify_freeze(frozen, subset, protocol)["verified_bound_files"], 5)
            bound.write_text("# changed\n")
            with self.assertRaises(ValueError):
                ap.verify_freeze(frozen, subset, protocol)


class ExportFormatTests(unittest.TestCase):
    def write_critic(self, root, subset):
        clips = []
        for i, entry in enumerate(subset["entries"]):
            clip = {"scene": entry["scene"], "name": entry["name"],
                    "source_archive_sha256": entry["source_sha256"],
                    "rgb_lq_tensor_sha256": entry["rgb_lq_tensor_sha256"],
                    "observable_sidecar_sha256": "c" * 64,
                    "coalition_order": list(ap.COALITIONS), "patch_yx": [[0, 0], [0, 1]],
                    "target_frames": [8, 9], "donor_frames": [[6, 7], [10, 11]],
                    "losses": {mode: [[2.0, 2.0], [1.6, 1.6], [1.8, 1.8], [1.5, 1.5]]
                               for mode in ("trained", "random", "wrong_time")},
                    "static_full_ema_flow_aligned_cosine": [[0.6, 0.6], [0.3, 0.3]]}
            path = root / f"clip_{i:03d}.json"
            path.write_text(json.dumps(clip))
            clips.append({"scene": entry["scene"], "name": entry["name"], "path": path.name,
                          "sha256": ap.sha256(path), "rgb_lq_tensor_sha256": entry["rgb_lq_tensor_sha256"]})
        path = root / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "count": 12, "clips": clips,
                                    "gt_storage_entries_read": 0, "quality_scores_computed": False}))
        return path

    def write_values(self, root, subset, *, bad_utility=False, bad_sameframe=False):
        import numpy as np
        rows = len(subset["entries"]) * 2
        arrays = {"clip_index": np.repeat(np.arange(12, dtype=np.int64), 2),
                  "patch_y": np.zeros(rows, dtype=np.int64), "patch_x": np.tile(np.arange(2, dtype=np.int64), 12)}
        for key, value in (("mse_base", 0.4), ("mse_single1", 0.3), ("mse_single2", 0.2),
                           ("mse_pair", 0.25), ("mse_sameframe_zero", 0.4), ("mse_temporal_no_dc", 0.2),
                           ("feature_lq1", -0.1), ("feature_lq2", -0.2), ("feature_hvi1", -0.2),
                           ("feature_hvi2", -0.3), ("feature_valid1", 0.8), ("feature_valid2", 0.7)):
            arrays[key] = np.full(rows, value, dtype=np.float64)
        for key, left, right in (("u1", "mse_base", "mse_single1"), ("u2", "mse_base", "mse_single2"),
                                 ("u12", "mse_base", "mse_pair"), ("u1_given2", "mse_single2", "mse_pair"),
                                 ("u2_given1", "mse_single1", "mse_pair")):
            arrays[key] = arrays[left] - arrays[right]
        if bad_utility:
            arrays["u1"][0] += 1e-5
        if bad_sameframe:
            arrays["mse_sameframe_zero"][0] += 1e-5
        path = root / "patch_statistics.npz"
        np.savez_compressed(path, **arrays)
        return path

    def test_real_format_join_and_no_control_sign_flip(self):
        subset = subset_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            critics, critic_receipt = ap.load_critic_input(self.write_critic(root, subset), subset,
                                                         observable_sha256="c" * 64)
            values, receipt = ap.load_value_input(self.write_values(root, subset), subset)
            grouped = ap.join_records(subset, critics, values)
            self.assertEqual(len(critic_receipt["clips"]), 12)
            self.assertEqual(receipt["independently_checked_utility_subtractions"], 24 * 5)
            patch = grouped[subset["entries"][0]["scene"]][0]
            self.assertEqual(patch["controls"]["negative_lq_mse"], [-0.1, -0.2])

    def test_critic_hash_and_sidecar_mismatch_rejected(self):
        subset = subset_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.write_critic(root, subset)
            with self.assertRaises(ValueError):
                ap.load_critic_input(manifest, subset, observable_sha256="d" * 64)
            (root / "clip_000.json").write_text("{}")
            with self.assertRaises(ValueError):
                ap.load_critic_input(manifest, subset)

    def test_saved_utility_and_sameframe_identity_checked(self):
        subset = subset_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                ap.load_value_input(self.write_values(root, subset, bad_utility=True), subset)
            with self.assertRaises(ValueError):
                ap.load_value_input(self.write_values(root, subset, bad_sameframe=True), subset)

    def test_gt_free_observables_are_exactly_rechecked(self):
        subset = subset_fixture()
        observables = {(entry["scene"], entry["name"]): {
            "feature_lq": [[-0.1] * 2, [-0.2] * 2], "feature_hvi": [[-0.2] * 2, [-0.3] * 2],
            "valid_fraction": [[0.8] * 2, [0.7] * 2]} for entry in subset["entries"]}
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_values(Path(directory), subset)
            _, receipt = ap.load_value_input(path, subset, observables=observables)
            self.assertEqual(receipt["gt_free_observable_features_exact_checked"], 24 * 6)
            observables[(subset["entries"][0]["scene"], subset["entries"][0]["name"])]["feature_lq"][0][0] += 1e-9
            with self.assertRaises(ValueError):
                ap.load_value_input(path, subset, observables=observables)


if __name__ == "__main__":
    unittest.main()
