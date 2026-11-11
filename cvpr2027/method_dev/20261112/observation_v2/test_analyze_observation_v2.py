"""Synthetic metadata/stats only; these fixtures are not experimental results."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_observation_v2_stats_test", HERE / "analyze_observation_v2.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
audit = analysis.audit
CVPR = Path(os.environ.get("SEER_ANALYSIS_CVPR", analysis.CVPR))


def sha(label):
    return hashlib.sha256(str(label).encode()).hexdigest()


def function(path, name, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def fixture(arm, splits, sources):
    split = json.loads(splits.read_text())
    names = [f"{d}/{n}" for d in ("SDSD_indoor", "SDSD_outdoor") for n in split[d]["train"]]
    scene_split = audit.expected_scene_split(names)
    rng = np.random.default_rng(20261113)
    plan = [{"sequence_index": i, "augmentation_seed": int(rng.integers(0, 2**63 - 1))} for i in range(136)]
    pairs = [[f"{i:06d}.npy", f"{i:06d}.npy"] for i in range(16)]
    inventory = [{"name": n, "paired_frames": 16, "matching_complete_basename_lists": True,
                  "pairing": "sorted index; SDSD _2 passes retain different original frame numbers",
                  "first_last_pairs": [pairs[0], pairs[-1]], "complete_basename_pairs": pairs,
                  "basename_pairs_sha256": hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()}
                 for n in names]
    origin = {"domain": "sdsd", "source_names": names, "split_sha256": audit.digest(splits),
              "base_checkpoint_sha256": audit.BASE_SHAS[arm], "base_state_sha256": sha("base " + arm),
              "raft_weights_sha256": audit.RAFT_WEIGHTS_SHA, "raft_state_sha256": sha("raft"),
              "encoder_eval": True, "encoder_weights_sha256": sha("encoder") if arm == "pretrained" else None,
              "base_saved_args": {"tag": f"e17_small_{arm}_s0", "vjepa": arm, "seed": 0, "iters": 20000},
              "args": {"data_seed": 20261113, "crop": 256, "clip_len": 16, "offset": 2,
                       "flow_steps": 12, "flow_batch": 2, "patch_size": 8},
              "plan": plan, "paired_frame_inventory": inventory,
              "source_sha256": {k: sources[k] for k in audit.BASE_SOURCE_KEYS}}
    clips = [{"name": n, "scene": audit.source_scene(n), "sha256": sha(f"{arm}clip{i}")} for i, n in enumerate(names)]
    summary = audit.check_cache_clips(clips, names)
    fit_frames = sum(audit.source_scene(n) in scene_split["fit"] for n in names) * 16
    norm = {"feature_mean": [0.] * 19, "feature_std": [1.] * 19, "target_scale": [1.] * 2,
            "energy_floor": 1e-6, "direct_mse_normalizer": .001 if arm == "pretrained" else .002,
            "fit_frames": fit_frames, "fit_patches": fit_frames * 1024, "source": audit.NORMALIZATION_SCOPE}
    ap = {"kind": "source_adjacent_flow_cache", "source_cache": {
              "provenance": origin, "summary": summary, "provenance_sha256": audit.written_json_digest(origin),
              "summary_sha256": audit.written_json_digest(summary)},
          "args": {"offsets": [-1, 1], "flow_steps": 12, "flow_batch": 2},
          "tool_sha256": sources["tools/cache_adjacent_flows.py"], "raft_weights_sha256": audit.RAFT_WEIGHTS_SHA,
          "raft_state_sha256_before": origin["raft_state_sha256"], "gt_input": False,
          "target_input": False, "metric_computed": False}
    meta = {"schema_version": 1, "domain": "sdsd", "n": 136, "source_cache_provenance": origin,
            "adjacent_provenance": ap, "source_cache_summary_sha256": audit.written_json_digest(summary),
            "adjacent_summary_sha256": sha("adjacent summary " + arm), "split": scene_split,
            "normalization": {c: copy.deepcopy(norm) for c in analysis.CANDIDATES},
            "candidate_recipes": {c: analysis.expected_recipe(c) for c in analysis.CANDIDATES},
            "per_sequence": [{"name": n, "scene": audit.source_scene(n), "cache_path": f"prepared_{i:03d}.pt",
                              "cache_sha256": sha("prepared" + str(i) + arm), "source_cache_sha256": clips[i]["sha256"],
                              "adjacent_cache_sha256": sha("adjacent" + str(i) + arm)} for i, n in enumerate(names)],
            "source_sha256": {k: sources[k] for k in analysis.TRAIN_SOURCES},
            "variants": {k: list(v) for k, v in analysis.VARIANTS.items()},
            "oracle_computed": False, "target_frames_read": False}
    prepared_sha = audit.written_json_digest(meta)
    training = {}
    for variant, (candidate, mode) in analysis.VARIANTS.items():
        for seed in analysis.SEEDS:
            training[f"{variant}_s{seed}"] = {
                "variant": variant, "seed": seed, "candidate": candidate,
                "args": {"steps": 1000, "lr": 1e-3, "device": "cpu", "threads": 4, "clips_per_update": 1,
                         "clip_len": 16, "loss_frames": 8, "source_crop": 128, "target_crop": 256,
                         "optimizer": "Adam default betas/eps; no weight decay",
                         "loss": "source RGB MSE / fit-only mean base MSE"},
                "model_config": {"mode": mode, "steps": 2, "offsets": [-1, 1], "patch": 8,
                                 "coupling": 1.0, "backend": "CPU float32", "anchor_precision": 1.0},
                "prepared_manifest_sha256": prepared_sha, "source_names": names, "source_scene_split": scene_split,
                "source_split_sha256": origin["split_sha256"], "base_checkpoint_sha256": origin["base_checkpoint_sha256"],
                "source_sha256": meta["source_sha256"], "normalization": meta["normalization"][candidate],
                "candidate_recipe": meta["candidate_recipes"][candidate],
                "selection": "Final1000 only; no intermediate source holdout or target selection",
                "sample_plan_sha256": audit.written_json_digest(analysis.expected_plan(names, scene_split, seed)),
                "initial_state_sha256": sha("initial seed" + str(seed))}
    old = {str(seed): {"args": {"candidate_mode": "raw_detail", "seed": seed, "steps": 1000,
                               "modes": list(audit.LOSSES), "lr": 1e-3, "batch_frames": 8, "device": "cpu"},
                       "cache_provenance": origin, "split": scene_split, "cache_clips": clips,
                       "cache_provenance_sha256": audit.written_json_digest(origin),
                       "cache_summary_sha256": audit.written_json_digest(summary),
                       "candidate_recipe": audit.expected_candidate_recipe("raw_detail"),
                       "source_sha256": {k: sources[k] for k in audit.TRAIN_SOURCE_KEYS},
                       "sample_plan_sha256": audit.written_json_digest(audit.expected_sample_plan(names, scene_split["fit"], seed)),
                       "normalization": norm} for seed in analysis.SEEDS}
    target_names = sorted("BVI/" + n for n in split["BVI_RLV"]["val"])
    rows = []
    for i, name in enumerate(target_names):
        methods, gt = {}, .04 + i * 1e-5
        for j, method in enumerate(sorted(analysis.EXPECTED_METHODS)):
            seed = int(method[-1]) if method[-3:-1] == "_s" else 0
            gain = 0. if method == "base" else .01 * j + .03 * seed + .001 * i
            if method.startswith("raw_node_s"):
                gain = (-.2, .4, .1)[seed] + .02 * (i // 2)
            gain *= 1. if arm == "pretrained" else .6
            tlp = gt + .02 - gain * .003
            methods[method] = {"psnr_rgb": 20 + .1 * i + gain + (1 if arm == "pretrained" else 0),
                               "psnr_y": 22 + .1 * i + gain + (1 if arm == "pretrained" else 0),
                               "lpips": .25 + .001 * i - gain * .01,
                               "tlp": tlp, "gt_tlp": gt, "tlp_gap": abs(tlp - gt)}
        rows.append({"name": name, "methods": methods})
    return {"domain": "bvi", "split": "val", "n": 16, "split_sha256": audit.digest(splits),
            "state_unchanged": {"base": True, "raft": True, "new_heads": True, "v1_heads": True},
            "protocol": "center16/crop256; rounded uint8 RGB/Y PSNR, LPIPS, per-sequence absolute tLP gap",
            "runtime": {"torch": "SYNTHETIC", "pyiqa": "SYNTHETIC", "graph_backend": "CPU float32", "threads": 4,
                        "metric_device": "npu:0"}, "scoring_library": {"sha256": sha("synthetic scorer")},
            "per_sequence": rows, "means": {m: {k: float(np.mean([r["methods"][m][k] for r in rows]))
                                                  for k in analysis.ALL_METRICS} for m in analysis.EXPECTED_METHODS},
            "validation_pair_inventory": [{"name": n, "matching_complete_basename_lists": True,
                                            "paired_frames": 32, "centre_start": 8} for n in target_names],
            "prepared_manifest": meta, "prepared_manifest_sha256": prepared_sha, "training": training,
            "checkpoint_sha256": {k: sha(k + arm) for k in training.keys() | {f"v1_raw_direct_s{s}" for s in analysis.SEEDS}},
            "v1_reference_training": old, "source_sha256": {k: sources[k] for k in analysis.EVAL_SOURCES}}


class ObservationStatsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = CVPR / "results/splits_cvpr.json"
        cls.code = CVPR / "code_full"
        sources = {k: audit.digest(cls.code / k) for k in analysis.EVAL_SOURCES}
        cls.raw = {arm: fixture(arm, cls.splits, sources) for arm in ("pretrained", "none")}

    def write(self, root, values):
        paths = {}
        for arm, raw in values.items():
            paths[arm] = root / f"SYNTHETIC_{arm}.json"
            paths[arm].write_text(json.dumps(raw, allow_nan=False))
        return paths

    def test_full_positive_fixture_and_exact_bootstrap_reference(self):
        with tempfile.TemporaryDirectory() as td:
            paths = self.write(Path(td), self.raw)
            result = analysis.analyze(paths["pretrained"], paths["none"], self.splits, code=self.code)
        self.assertEqual(len(result["within_arms"]["pretrained"]["individual_methods"]), 31)
        self.assertEqual(len(result["within_arms"]["pretrained"]["comparisons"]), 12)
        self.assertEqual(len(result["between_arms"]), 27)
        names = sorted(r["name"] for r in self.raw["pretrained"]["per_sequence"])
        scenes = sorted({audit.target_scene(n) for n in names})
        effects = []
        for arm in ("pretrained", "none"):
            rows = {r["name"]: r["methods"] for r in self.raw[arm]["per_sequence"]}
            effects.append(np.array([[rows[n][f"raw_node_s{s}"]["psnr_y"] - rows[n]["base"]["psnr_y"]
                                      for n in names] for s in (0, 1, 2)]))
        draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
        sequence = effects[0].mean(0)
        per_scene = np.array([sequence[[audit.target_scene(n) == scene for n in names]].mean() for scene in scenes])
        expected = np.quantile(per_scene[draws].mean(1), [.025, .975])
        row = result["within_arms"]["pretrained"]["comparisons"]["raw_node_minus_base"]["psnr_y"]
        np.testing.assert_array_equal(row["scene_ci95"], expected)
        np.testing.assert_array_equal(row["per_selector_seed_difference"], effects[0].mean(1))
        self.assertEqual(row["selector_seeds_better"], 2)
        interaction = result["between_arms"]["contrast_effect_pretrained_minus_none/raw_node_minus_base"]["psnr_y"]
        self.assertAlmostEqual(interaction["difference"], (effects[0] - effects[1]).mean())
        self.assertIn("not isolated causal", " ".join(result["limitations"]))
        self.assertIn("SOTA", " ".join(result["limitations"]))
        self.assertEqual(result["none_gt_tlp_max_abs_vs_pretrained"], 0.)
        self.assertTrue(analysis.table(result).endswith("\n"))

    def test_plan_and_recipe_match_actual_source_functions_without_torch(self):
        names = self.raw["pretrained"]["prepared_manifest"]["source_cache_provenance"]["source_names"]
        split = audit.expected_scene_split(names)
        make_plan = function(self.code / "tools/train_observation_v2.py", "make_plan",
                             {"np": np, "source_scene": audit.source_scene, "STEPS": 1000, "CROP": 128, "LOSS_FRAMES": 8})
        for seed in (0, 1, 2):
            self.assertEqual(analysis.expected_plan(names, split, seed), make_plan(names, split, seed))
        config = function(self.code / "models/observation_controls.py", "observation_control_config",
                          {"_reference": SimpleNamespace(candidate_config=audit.expected_candidate_recipe), "copy": copy,
                           "CONTROL_MODES": analysis.CANDIDATES})
        # candidate_config has a second offset parameter in the actual helper.
        config.__globals__["_reference"] = SimpleNamespace(candidate_config=lambda mode, offset: audit.expected_candidate_recipe(mode))
        for mode in analysis.CANDIDATES:
            self.assertEqual(analysis.expected_recipe(mode), config(mode, 16, 2))

    def test_scene_block_bootstrap_is_order_invariant_and_not_sequence_resampling(self):
        names = sorted(r["name"] for r in self.raw["pretrained"]["per_sequence"])
        scenes = sorted({audit.target_scene(n) for n in names})
        values = np.array([i // 2 + (100 if i % 2 else -100) for i in range(16)], float)
        effects = np.stack([values - 1, values, values + 1])
        draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
        expected = np.quantile(np.arange(8)[draws].mean(1), [.025, .975])
        result = analysis.paired_statistics(effects, names, scenes, draws, "lpips")
        np.testing.assert_array_equal(result["scene_ci95"], expected)
        self.assertEqual(result["selector_seeds_better"], 0)
        permutation = np.random.default_rng(5).permutation(16)
        other = analysis.paired_statistics(effects[:, permutation], [names[i] for i in permutation], scenes, draws, "lpips")
        np.testing.assert_array_equal(other["scene_ci95"], result["scene_ci95"])

    def test_raw_sign_and_benefit_fields_and_legacy_operation_order(self):
        names = sorted(r["name"] for r in self.raw["pretrained"]["per_sequence"])
        scenes = sorted({audit.target_scene(n) for n in names})
        effects = np.random.default_rng(55).normal(size=(3, 16)) * .003
        draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
        sequence = effects.mean(0)
        scene = np.array([sequence[[audit.target_scene(n) == s for n in names]].mean() for s in scenes])
        ci = np.quantile(scene[draws].mean(1), [.025, .975])
        result = analysis.paired_statistics(effects, names, scenes, draws, "tlp_gap")
        np.testing.assert_array_equal(result["scene_ci95"], ci)
        self.assertEqual(result["difference"], float(sequence.mean()))
        benefit = result["benefit_oriented"]
        self.assertEqual(benefit["difference"], -result["difference"])
        np.testing.assert_array_equal(benefit["scene_ci95"], -ci[::-1])
        np.testing.assert_array_equal(result["per_selector_seed_difference"], effects.mean(1))
        np.testing.assert_array_equal(benefit["per_selector_seed_difference"], -effects.mean(1))
        positive = analysis.paired_statistics(effects, names, scenes, draws, "psnr_rgb")
        self.assertEqual(positive["benefit_oriented"]["difference"], positive["difference"])
        np.testing.assert_array_equal(positive["benefit_oriented"]["scene_ci95"], ci)

    def test_schema_seed_and_signed_temporal_gap_negative_cases(self):
        changes = [lambda r: r["per_sequence"][0]["methods"].pop("raw_node_s0"),
                   lambda r: r["per_sequence"].__setitem__(1, copy.deepcopy(r["per_sequence"][0])),
                   lambda r: r["means"]["base"].__setitem__("psnr_y", 999.),
                   lambda r: r["training"].pop("trajectory_s2"),
                   lambda r: r["training"]["raw_node_s0"].__setitem__("seed", 1),
                   lambda r: r["training"]["raw_node_s1"].__setitem__("sample_plan_sha256", sha("wrong plan")),
                   lambda r: r["training"]["same_frame_node_s0"].__setitem__("initial_state_sha256", sha("different initial"))]
        def bad_gap(raw):
            for row in raw["per_sequence"]:
                row["methods"]["raw_node_s0"]["tlp_gap"] = -1.
            raw["means"]["raw_node_s0"]["tlp_gap"] = -1.
        changes.append(bad_gap)
        with tempfile.TemporaryDirectory() as td:
            for change in changes:
                bad = copy.deepcopy(self.raw["pretrained"])
                change(bad)
                path = Path(td) / "SYNTHETIC_invalid.json"; path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    analysis.load_arm(path, self.splits, base_arm="pretrained", code=self.code)

    def test_cross_arm_gt_and_pairing_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            raw = copy.deepcopy(self.raw)
            raw["none"]["validation_pair_inventory"][0]["paired_frames"] = 34
            raw["none"]["validation_pair_inventory"][0]["centre_start"] = 9
            paths = self.write(Path(td), raw)
            with self.assertRaisesRegex(ValueError, "Arm target pairing"):
                analysis.analyze(paths["pretrained"], paths["none"], self.splits, code=self.code)
            raw = copy.deepcopy(self.raw)
            for row in raw["none"]["per_sequence"]:
                for value in row["methods"].values():
                    value["gt_tlp"] += .001
                    value["tlp_gap"] = abs(value["tlp"] - value["gt_tlp"])
            for method in analysis.EXPECTED_METHODS:
                for metric in ("gt_tlp", "tlp_gap"):
                    raw["none"]["means"][method][metric] = float(np.mean([r["methods"][method][metric] for r in raw["none"]["per_sequence"]]))
            paths = self.write(Path(td), raw)
            with self.assertRaisesRegex(ValueError, "Arm GT temporal"):
                analysis.analyze(paths["pretrained"], paths["none"], self.splits, code=self.code)


if __name__ == "__main__":
    unittest.main()
