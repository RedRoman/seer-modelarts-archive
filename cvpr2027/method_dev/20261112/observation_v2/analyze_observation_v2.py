"""Frozen, outcome-independent paired analysis for observation-v2 BVI VAL16.

No Torch/model/tensor/frame loading. All 31 methods and all declared contrasts
are retained. Three head-seed effects are averaged within each sequence before
resampling the eight complete scenes; intervals do not replicate base training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

CVPR = Path(__file__).resolve().parents[3]
REFERENCE_SHA = "1e61d95af69a5be89aef35ccb03fbf0af73df69bac6c9e807fa62fb85d33cbc7"
_reference = Path(__file__).with_name("reference_analyze_source_transfer.py")
if not _reference.exists():
    _reference = CVPR / "method_dev/20261111/analyze_source_transfer.py"
if hashlib.sha256(_reference.read_bytes()).hexdigest() != REFERENCE_SHA:
    raise ValueError("Frozen source-transfer reference auditor SHA differs")
_spec = importlib.util.spec_from_file_location("_observation_v2_reference_audit", _reference)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

METRICS, ALL_METRICS, SEEDS = audit.METRICS, audit.ALL_METRICS, (0, 1, 2)
CANDIDATES = ("raw_detail", "same_frame_detail", "mismatched_detail")
VARIANTS = {
    "raw_node": ("raw_detail", "node_only"),
    "same_frame_node": ("same_frame_detail", "node_only"),
    "mismatched_node": ("mismatched_detail", "node_only"),
    "trajectory": ("raw_detail", "conditioned"),
    "trajectory_fixed_edge": ("raw_detail", "fixed_edge"),
    "trajectory_no_cross": ("raw_detail", "no_cross"),
    "trajectory_scrambled_cross": ("raw_detail", "scrambled_cross"),
}
FIXED_METHODS = ("base", *(f"{c}/{f}" for c in CANDIDATES for f in ("initial_thirds", "fixed_average")))
LEARNED_METHODS = (*VARIANTS, "v1_raw_direct")
EXPECTED_METHODS = set(FIXED_METHODS) | {f"{v}_s{s}" for v in LEARNED_METHODS for s in SEEDS}
CONTRASTS = (
    ("raw_node", "same_frame_node"), ("raw_node", "mismatched_node"), ("raw_node", "base"),
    ("raw_node", "raw_detail/initial_thirds"), ("raw_node", "v1_raw_direct"),
    ("trajectory", "raw_node"), ("trajectory", "trajectory_fixed_edge"),
    ("trajectory", "trajectory_no_cross"), ("trajectory", "trajectory_scrambled_cross"),
    ("trajectory", "base"), ("trajectory", "raw_detail/initial_thirds"), ("trajectory", "v1_raw_direct"),
)
TRAIN_SOURCES = audit.TRAIN_SOURCE_KEYS | {
    "tools/train_observation_v2.py", "models/observation_controls.py",
    "models/trajectory_reconstruction.py", "tools/cache_adjacent_flows.py"}
EVAL_SOURCES = TRAIN_SOURCES | {"tools/eval_observation_v2.py", "tools/eval_source_selector.py",
                               "tools/eval_correction_selector.py"}
BOOTSTRAP_SEED, RESAMPLES = 0, 10000


def expected_plan(names, split, seed):
    """Reproduce the new clip-first plan independently of the Torch trainer."""
    fit = np.array([i for i, n in enumerate(names) if audit.source_scene(n) in split["fit"]])
    rng = np.random.default_rng(seed)
    clips = rng.choice(fit, size=1000, replace=True)
    return {"seed": seed, "steps": 1000, "crop": 128, "loss_frames": 8, "source_names": names,
            "batches": [{"clip_index": int(i), "frames": rng.choice(16, size=8, replace=False).tolist(),
                         "top": int(rng.integers(0, 17)) * 8, "left": int(rng.integers(0, 17)) * 8}
                        for i in clips]}


def expected_recipe(mode):
    recipe = audit.expected_candidate_recipe("raw_detail")
    directions = []
    for label, delta in (("past", -2), ("future", 2)):
        original = [t + delta for t in range(16)]
        requested = [t if mode == "same_frame_detail" else t - delta if mode == "mismatched_detail"
                     else t + delta for t in range(16)]
        actual = [min(max(t, 0), 15) for t in requested]
        directions.append({"direction": label, "original_offset": delta,
                           "original_source_requested": original,
                           "original_source_in_clip": [0 <= t < 16 for t in original],
                           "value_source_requested": requested, "value_source_indices": actual,
                           "value_index_clamped": [a != b for a, b in zip(requested, actual)],
                           "value_source_is_target": [s == t for t, s in enumerate(actual)]})
    return {"schema_version": 1, "mode": mode, "clip_len": 16, "offset": 2, "base_recipe": recipe,
            "direction_order": ["past", "future"], "directions": directions,
            "value_alignment": "none; target frame values" if mode == "same_frame_detail" else
                               "original direction's saved target-to-source flow, unchanged",
            "validity": "original direction's validity and reflect-LP9 support erosion; never relaxed",
            "mismatch_edge_policy": "opposite requested source index clamped to [0,T-1]; no wrapping",
            "gt_input": False, "target_domain_fit": False,
            "interpretation": "fixed-input candidate control; base_t may contain all-clip information; "
                              "no independence or novelty claim"}


def contrast_label(left, right):
    return left + "_minus_" + right


def _audit_preparation(raw, split, split_path, base_arm, code):
    require = audit.require
    meta = raw["prepared_manifest"]
    names = [f"{d}/{n}" for d in ("SDSD_indoor", "SDSD_outdoor") for n in split[d]["train"]]
    scenes = {audit.source_scene(n) for n in names}
    require(len(names) == len(set(names)) == 136 and len(scenes) == 68, "Source136/68 differs")
    for domain, count in (("SDSD_indoor", 46), ("SDSD_outdoor", 90)):
        entry = split[domain]
        groups = [{audit.source_scene(f"{domain}/{n}") for n in entry[k]} for k in ("train", "val", "test")]
        require(len(entry["train"]) == count and not (groups[0] & groups[1] or groups[0] & groups[2]
                or groups[1] & groups[2]), "Source scenes overlap held-out source splits")
    origin = meta["source_cache_provenance"]
    expected_split = audit.expected_scene_split(names)
    require(meta["schema_version"] == 1 and meta["domain"] == "sdsd" and meta["n"] == 136
            and meta["oracle_computed"] is False and meta["target_frames_read"] is False
            and meta["split"] == expected_split and len(expected_split["fit"]) == 51
            and len(expected_split["internal_holdout"]) == 17
            and meta["variants"] == {k: list(v) for k, v in VARIANTS.items()}, "Preparation scope/split/variants differ")
    require(origin["domain"] == "sdsd" and origin["source_names"] == names
            and origin["split_sha256"] == audit.digest(split_path)
            and origin["base_checkpoint_sha256"] == audit.BASE_SHAS[base_arm], "Source base/cache identity differs")
    saved = origin["base_saved_args"]
    require(saved["tag"] == f"e17_small_{base_arm}_s0" and saved["vjepa"] == base_arm
            and saved["seed"] == 0 and saved["iters"] == 20000 and saved.get("heldout") is None
            and saved.get("only") is None, "Historical E17 base identity differs")
    for key in ("base_state_sha256", "raft_state_sha256", "raft_weights_sha256"):
        require(audit.valid_sha(origin[key]), f"Missing original state SHA: {key}")
    require(origin["raft_weights_sha256"] == audit.RAFT_WEIGHTS_SHA and origin["encoder_eval"] is True
            and (audit.valid_sha(origin["encoder_weights_sha256"]) if base_arm == "pretrained"
                 else origin["encoder_weights_sha256"] is None), "RAFT/encoder provenance differs")
    audit.check_source_inventory(origin, names)
    audit.check_sources(origin["source_sha256"], audit.BASE_SOURCE_KEYS, code=code)
    audit.check_sources(meta["source_sha256"], TRAIN_SOURCES, code=code)
    audit.check_sources(raw["source_sha256"], EVAL_SOURCES, code=code)
    require(audit.valid_sha(raw["prepared_manifest_sha256"])
            and raw["prepared_manifest_sha256"] == audit.written_json_digest(meta), "Prepared manifest SHA differs")
    rows = meta["per_sequence"]
    require(len(rows) == 136 and [r["name"] for r in rows] == names, "Prepared source inventory incomplete")
    clips = []
    for i, row in enumerate(rows):
        require(row["scene"] == audit.source_scene(row["name"]) and row["cache_path"] == f"prepared_{i:03d}.pt"
                and all(audit.valid_sha(row[k]) for k in
                        ("cache_sha256", "source_cache_sha256", "adjacent_cache_sha256")), "Prepared lineage differs")
        clips.append({"name": row["name"], "scene": row["scene"], "sha256": row["source_cache_sha256"]})
    source_summary = audit.check_cache_clips(clips, names)
    source_prov_sha = audit.written_json_digest(origin)
    source_summary_sha = audit.written_json_digest(source_summary)
    ap = meta["adjacent_provenance"]
    require(ap["kind"] == "source_adjacent_flow_cache" and ap["source_cache"]["provenance"] == origin
            and ap["source_cache"]["summary"] == source_summary
            and ap["source_cache"]["provenance_sha256"] == source_prov_sha
            and ap["source_cache"]["summary_sha256"] == source_summary_sha
            and meta["source_cache_summary_sha256"] == source_summary_sha
            and audit.valid_sha(meta["adjacent_summary_sha256"]), "Adjacent/source nested lineage differs")
    require(ap["args"]["offsets"] == [-1, 1] and ap["args"]["flow_steps"] == 12 and ap["args"]["flow_batch"] == 2
            and ap["tool_sha256"] == audit.digest(Path(code) / "tools/cache_adjacent_flows.py")
            and ap["raft_weights_sha256"] == audit.RAFT_WEIGHTS_SHA
            and ap["raft_state_sha256_before"] == origin["raft_state_sha256"]
            and ap["gt_input"] is False and ap["target_input"] is False and ap["metric_computed"] is False,
            "Adjacent fixed recipe/state/scope differs")
    fit_frames = sum(audit.source_scene(n) in expected_split["fit"] for n in names) * 16
    require(set(meta["normalization"]) == set(CANDIDATES) and set(meta["candidate_recipes"]) == set(CANDIDATES),
            "Missing normalization or candidate recipe")
    for candidate in CANDIDATES:
        audit.check_normalization(meta["normalization"][candidate], fit_frames)
        require(meta["candidate_recipes"][candidate] == expected_recipe(candidate), "Fixed control recipe differs")
        require(meta["normalization"][candidate]["direct_mse_normalizer"]
                == meta["normalization"]["raw_detail"]["direct_mse_normalizer"], "Base-MSE loss unit differs")
    return meta, origin, names, clips, expected_split


def _audit_training(raw, meta, origin, names, clips, source_split, code):
    require = audit.require
    training = raw["training"]
    keys = {f"{v}_s{s}" for v in VARIANTS for s in SEEDS}
    require(set(training) == keys and set(raw["checkpoint_sha256"]) == keys | {f"v1_raw_direct_s{s}" for s in SEEDS}
            and all(audit.valid_sha(sha) for sha in raw["checkpoint_sha256"].values()), "Head coverage/hash differs")
    plan_shas = {s: audit.written_json_digest(expected_plan(names, source_split, s)) for s in SEEDS}
    init = {}
    for variant, (candidate, mode) in VARIANTS.items():
        for seed in SEEDS:
            p = training[f"{variant}_s{seed}"]
            a, config = p["args"], p["model_config"]
            require(p["variant"] == variant and p["seed"] == seed and p["candidate"] == candidate
                    and a["steps"] == 1000 and a["lr"] == 1e-3 and a["device"] == "cpu" and a["threads"] == 4
                    and a["clips_per_update"] == 1 and a["clip_len"] == 16 and a["loss_frames"] == 8
                    and a["source_crop"] == 128 and a["target_crop"] == 256
                    and a["optimizer"] == "Adam default betas/eps; no weight decay"
                    and a["loss"] == "source RGB MSE / fit-only mean base MSE", "V2 training budget differs")
            require(config["mode"] == mode and config["steps"] == 2 and config["offsets"] == [-1, 1]
                    and config["patch"] == 8 and config["coupling"] == 1.0 and config["backend"] == "CPU float32"
                    and config["anchor_precision"] == 1.0, "Graph configuration differs")
            require(p["prepared_manifest_sha256"] == raw["prepared_manifest_sha256"]
                    and p["source_names"] == names and p["source_scene_split"] == source_split
                    and p["source_split_sha256"] == origin["split_sha256"]
                    and p["base_checkpoint_sha256"] == origin["base_checkpoint_sha256"]
                    and p["source_sha256"] == meta["source_sha256"]
                    and p["normalization"] == meta["normalization"][candidate]
                    and p["candidate_recipe"] == meta["candidate_recipes"][candidate]
                    and p["selection"] == "Final1000 only; no intermediate source holdout or target selection",
                    "V2 source/normalization/selection lineage differs")
            require(p["sample_plan_sha256"] == plan_shas[seed] and audit.valid_sha(p["initial_state_sha256"]),
                    "V2 seed plan/initialization differs")
            require(seed not in init or init[seed] == p["initial_state_sha256"], "Variant initialization differs")
            init[seed] = p["initial_state_sha256"]
    require(len(set(init.values())) == 3, "Repeated V2 initialization seed")
    old = raw["v1_reference_training"]
    require(set(old) == {str(s) for s in SEEDS}, "V1 reference seed coverage differs")
    for seed in SEEDS:
        p = old[str(seed)]
        a = p["args"]
        require(a["candidate_mode"] == "raw_detail" and a["seed"] == seed and a["steps"] == 1000
                and a["modes"] == list(audit.LOSSES) and a["lr"] == 1e-3 and a["batch_frames"] == 8
                and a["device"] == "cpu" and p["cache_provenance"] == origin
                and p["split"] == source_split and p["cache_clips"] == clips
                and p["cache_provenance_sha256"] == audit.written_json_digest(origin)
                and p["cache_summary_sha256"] == audit.written_json_digest(audit.check_cache_clips(clips, names))
                and p["candidate_recipe"] == audit.expected_candidate_recipe("raw_detail"), "V1 source reference differs")
        audit.check_sources(p["source_sha256"], audit.TRAIN_SOURCE_KEYS, code=code)
        require(p["sample_plan_sha256"] == audit.written_json_digest(
                    audit.expected_sample_plan(names, source_split["fit"], seed)), "V1 reference plan differs")
        audit.check_normalization(p["normalization"], sum(audit.source_scene(n) in source_split["fit"] for n in names) * 16)
    return plan_shas, init


def load_arm(path, splits, *, base_arm, code):
    require = audit.require
    raw = json.loads(Path(path).read_text())
    split = json.loads(Path(splits).read_text())
    names = sorted("BVI/" + n for n in split["BVI_RLV"]["val"])
    scenes = sorted({audit.target_scene(n) for n in names})
    require(len(names) == len(set(names)) == 16 and len(scenes) == 8
            and all(sum(audit.target_scene(n) == s for n in names) == 2 for s in scenes), "VAL16/8scene split differs")
    require(raw["domain"] == "bvi" and raw["split"] == "val" and raw["n"] == 16
            and raw["split_sha256"] == audit.digest(splits), "Not the frozen BVI VAL16")
    require(raw["state_unchanged"] == {"base": True, "raft": True, "new_heads": True, "v1_heads": True},
            "Inference modified model state")
    require(raw["protocol"] == "center16/crop256; rounded uint8 RGB/Y PSNR, LPIPS, per-sequence absolute tLP gap"
            and raw["runtime"]["graph_backend"] == "CPU float32" and raw["runtime"]["threads"] == 4
            and raw["runtime"]["metric_device"] in ("npu:0", "cuda:0")
            and audit.valid_sha(raw["scoring_library"]["sha256"]), "Metric/backend protocol differs")
    rows = {r["name"]: r["methods"] for r in raw["per_sequence"]}
    require(len(raw["per_sequence"]) == 16 and sorted(rows) == names
            and set(raw["means"]) == EXPECTED_METHODS and all(set(r) == EXPECTED_METHODS for r in rows.values()),
            "Need all31 methods for all16 distinct sequences")
    inventory = raw["validation_pair_inventory"]
    require(len(inventory) == 16 and sorted(r["name"] for r in inventory) == names
            and all(r["matching_complete_basename_lists"] is True and isinstance(r["paired_frames"], int)
                    and r["paired_frames"] >= 16 and r["centre_start"] == (r["paired_frames"] - 16) // 2
                    for r in inventory), "Target frame pairing differs")
    meta, origin, source_names, clips, source_split = _audit_preparation(raw, split, splits, base_arm, code)
    plan_shas, init = _audit_training(raw, meta, origin, source_names, clips, source_split, code)
    values = {}
    for method in sorted(EXPECTED_METHODS):
        require(set(raw["means"][method]) == set(ALL_METRICS)
                and all(set(rows[n][method]) == set(ALL_METRICS) for n in names), "Metric schema differs")
        values[method] = {}
        for metric in ALL_METRICS:
            array = np.array([rows[n][method][metric] for n in names], dtype=float)
            require(np.isfinite(array).all() and np.isfinite(raw["means"][method][metric])
                    and np.isclose(array.mean(), raw["means"][method][metric], atol=1e-10, rtol=0),
                    f"Nonfinite or incorrect mean: {method}/{metric}")
            values[method][metric] = array
        require(np.allclose(values[method]["tlp_gap"], abs(values[method]["tlp"] - values[method]["gt_tlp"]),
                            atol=1e-12, rtol=0), "Temporal gap must be absolute distance to GT")
        require(np.array_equal(values[method]["gt_tlp"], values["base"]["gt_tlp"]), "GT temporal reference differs")
    return {"raw": raw, "values": values, "names": names, "scenes": scenes,
            "source_names": source_names, "source_split": source_split, "origin": origin,
            "plan_sha256": plan_shas, "initial_sha256": init, "input_sha256": audit.digest(path)}


def method_values(arm, label, metric):
    """Always [3,16], replicating fixed controls rather than pseudo-replicating scenes."""
    if label in FIXED_METHODS:
        return np.broadcast_to(arm["values"][label][metric], (3, 16))
    return np.stack([arm["values"][f"{label}_s{s}"][metric] for s in SEEDS])


def paired_statistics(effects, names, scenes, draws, metric):
    effects = np.asarray(effects, dtype=float)
    audit.require(effects.shape == (3, 16) and np.isfinite(effects).all(), "Need paired 3seed-by16 effects")
    sequence = effects.mean(0)
    seed_scene = np.stack([effects[:, [audit.target_scene(n) == s for n in names]].mean(1) for s in scenes], 1)
    # Match the legacy numerical operation order, not just its algebra:
    # average seeds FIRST, then the two conditions within each scene.
    scene = np.array([sequence[[audit.target_scene(n) == s for n in names]].mean() for s in scenes])
    seed_means = effects.mean(1)
    favorable = 1 if metric.startswith("psnr") else -1
    difference = float(sequence.mean())
    ci = np.quantile(scene[draws].mean(1), [.025, .975])
    benefit_ci = ci if favorable == 1 else -ci[::-1]
    return {"difference": difference, "scene_ci95": ci.tolist(),
            "per_selector_seed_difference": seed_means.tolist(),
            "selector_seeds_better": int((favorable * seed_means > 0).sum()),
            "seed_averaged_sequence_differences": dict(zip(names, sequence.tolist())),
            "seed_averaged_scene_differences": dict(zip(scenes, scene.tolist())),
            "per_selector_seed_scene_differences": {
                str(s): dict(zip(scenes, seed_scene[s].tolist())) for s in SEEDS},
            "per_selector_seed_sequence_differences": {
                str(s): dict(zip(names, effects[s].tolist())) for s in SEEDS},
            "benefit_oriented": {"sign_applied_to_raw_difference": favorable,
                                 "difference": favorable * difference, "scene_ci95": benefit_ci.tolist(),
                                 "per_selector_seed_difference": (favorable * seed_means).tolist(),
                                 "seed_averaged_sequence_differences": dict(zip(names, (favorable * sequence).tolist())),
                                 "seed_averaged_scene_differences": dict(zip(scenes, (favorable * scene).tolist()))}}


def arm_statistics(arm, draws):
    absolute, individual = {}, {}
    names, scenes = arm["names"], arm["scenes"]
    for label in (*FIXED_METHODS, *LEARNED_METHODS):
        absolute[label] = {}
        for metric in ALL_METRICS:
            array = method_values(arm, label, metric)
            means = array.mean(1)
            absolute[label][metric] = {"mean": float(array.mean()), "per_selector_seed": means.tolist(),
                                      "seed_range": [float(means.min()), float(means.max())],
                                      "fixed_control_repeated_for_pairing_only": label in FIXED_METHODS,
                                      "scene_means": {s: float(array[:, [audit.target_scene(n) == s for n in names]].mean())
                                                      for s in scenes}}
    for method in sorted(EXPECTED_METHODS):
        individual[method] = {metric: {"mean": float(array.mean()), "per_sequence": dict(zip(names, array.tolist())),
                                      "per_scene": {s: float(array[[audit.target_scene(n) == s for n in names]].mean())
                                                    for s in scenes}}
                              for metric, array in arm["values"][method].items()}
    comparisons = {contrast_label(left, right): {
        metric: paired_statistics(method_values(arm, left, metric) - method_values(arm, right, metric),
                                  names, scenes, draws, metric) for metric in METRICS}
        for left, right in CONTRASTS}
    return {"absolute": absolute, "individual_methods": individual, "comparisons": comparisons,
            "input_sha256": arm["input_sha256"], "base_checkpoint_sha256": arm["origin"]["base_checkpoint_sha256"],
            "audit": {"source_scene_split": arm["source_split"], "sample_plan_sha256": arm["plan_sha256"],
                      "initial_state_sha256": arm["initial_sha256"],
                      "prepared_manifest_sha256": arm["raw"]["prepared_manifest_sha256"],
                      "checkpoint_sha256": arm["raw"]["checkpoint_sha256"],
                      "runtime": arm["raw"]["runtime"], "scoring_library": arm["raw"]["scoring_library"],
                      "source_sha256": arm["raw"]["source_sha256"]}}


def analyze(pretrained, none, splits=CVPR / "results/splits_cvpr.json", *, code=CVPR / "code_full"):
    arms = {"pretrained": load_arm(pretrained, splits, base_arm="pretrained", code=code),
            "none": load_arm(none, splits, base_arm="none", code=code)}
    p, n = arms.values()
    require = audit.require
    require(p["names"] == n["names"] and p["scenes"] == n["scenes"]
            and p["raw"]["validation_pair_inventory"] == n["raw"]["validation_pair_inventory"], "Arm target pairing differs")
    for key in ("plan", "source_names", "paired_frame_inventory", "split_sha256", "args",
                "raft_weights_sha256", "raft_state_sha256"):
        require(p["origin"][key] == n["origin"][key], f"Arm source data/recipe differs: {key}")
    require(p["source_split"] == n["source_split"] and p["plan_sha256"] == n["plan_sha256"]
            and p["initial_sha256"] == n["initial_sha256"], "Arm added-head split/plan/initialization differs")
    require(p["raw"]["source_sha256"] == n["raw"]["source_sha256"]
            and p["raw"]["runtime"] == n["raw"]["runtime"]
            and p["raw"]["scoring_library"]["sha256"] == n["raw"]["scoring_library"]["sha256"],
            "Arm code/runtime/scoring library differs")
    gt_difference = float(abs(p["values"]["base"]["gt_tlp"] - n["values"]["base"]["gt_tlp"]).max())
    require(gt_difference <= 1e-8, "Arm GT temporal reference differs")
    # Exactly the legacy scene-bootstrap draws, generated ONCE and reused for
    # every arm/metric/contrast. No seed or sequence pseudo-replication.
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 8, size=(RESAMPLES, 8))
    within = {label: arm_statistics(arm, draws) for label, arm in arms.items()}
    between = {}
    for label in (*FIXED_METHODS, *LEARNED_METHODS):
        between["absolute_pretrained_minus_none/" + label] = {
            metric: paired_statistics(method_values(p, label, metric) - method_values(n, label, metric),
                                      p["names"], p["scenes"], draws, metric) for metric in METRICS}
    for left, right in CONTRASTS:
        between["contrast_effect_pretrained_minus_none/" + contrast_label(left, right)] = {
            metric: paired_statistics((method_values(p, left, metric) - method_values(p, right, metric))
                                      - (method_values(n, left, metric) - method_values(n, right, metric)),
                                      p["names"], p["scenes"], draws, metric) for metric in METRICS}
    return {"scope": "Exploratory source SDSD136 training to BVI VAL16; all prespecified methods/contrasts",
            "n_sequences": 16, "n_scenes": 8, "head_seeds": list(SEEDS), "base_seeds_per_arm": [0],
            "within_arms": within, "between_arms": between,
            "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "interval": "unadjusted percentile95",
                          "sampling": "mean three paired head-seed effects per sequence, then two conditions per scene; resample eight whole scenes",
                          "draws_shared_across_all_contrasts": True, "conditional_on_fitted_models": True,
                          "fitted_head_seeds_per_base": 3, "fitted_bases_per_arm": 1, "scene_resampling_only": True},
            "metric_direction": {"psnr_rgb": "higher", "psnr_y": "higher", "lpips": "lower", "tlp_gap": "lower"},
            "temporal_gap": "absolute per-sequence |tLP(prediction)-tLP(GT)|, before seed/scene averaging",
            "contrasts": [[left, right] for left, right in CONTRASTS], "all_methods": sorted(EXPECTED_METHODS),
            "none_gt_tlp_max_abs_vs_pretrained": gt_difference,
            "split_sha256": audit.digest(splits), "analyzer_sha256": audit.digest(__file__),
            "reference_auditor_sha256": REFERENCE_SHA, "numpy_version": str(np.__version__),
            "limitations": ["One historical base per arm; three added-head seeds do not replicate base training",
                            "Base pixel initialization/training histories differ; between-arm effects are not isolated causal video-prior evidence",
                            "Scene bootstrap intervals are exploratory and unadjusted across the full declared comparison family",
                            "Same-frame controls remove explicit neighbour values only at construction; base_t is all-clip",
                            "Node-only and graph modes have different initial functions; inactive edge parameters are not equal active capacity",
                            "V1 uses an earlier frame-batch/whole-crop training recipe; comparison is a frozen reference, not a single-mechanism isolation",
                            "Extra adjacent RAFT and CPU graph work are real costs; no equal-latency claim",
                            "BVI VAL informed prior project development; no target-blind-history or SOTA conclusion"]}


def table(result):
    lines = ["# Observation-v2 complete paired analysis", "", result["scope"], "",
             "Differences are left minus right; PSNR higher is favorable, LPIPS and absolute tLP gap lower is favorable.", ""]
    for arm, data in result["within_arms"].items():
        lines += [f"## {arm}", "", "| Method | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |",
                  "|---|---:|---:|---:|---:|"]
        for method, metrics in data["absolute"].items():
            lines.append("| " + method + " | " + " | ".join(f"{metrics[k]['mean']:.8f}" for k in METRICS) + " |")
        lines += ["", "| Contrast | Metric | Difference | Scene95%CI | Favorable head seeds /3 |",
                  "|---|---|---:|---|---:|"]
        for contrast, metrics in data["comparisons"].items():
            for metric, row in metrics.items():
                lo, hi = row["scene_ci95"]
                lines.append(f"| {contrast} | {metric} | {row['difference']:+.8f} | [{lo:+.8f},{hi:+.8f}] | {row['selector_seeds_better']} |")
        lines.append("")
    lines += ["## Between historical base arms", "", "These comparisons do not isolate a causal video-prior effect.", "",
              "| Difference of arm values/effects | Metric | Difference | Scene95%CI |", "|---|---|---:|---|"]
    for contrast, metrics in result["between_arms"].items():
        for metric, row in metrics.items():
            lo, hi = row["scene_ci95"]
            lines.append(f"| {contrast} | {metric} | {row['difference']:+.8f} | [{lo:+.8f},{hi:+.8f}] |")
    lines += ["", "Bootstrap: 10,000 complete-scene resamples, RNG0, seed effects averaged before resampling; unadjusted percentile95.", ""]
    lines.extend("- " + item for item in result["limitations"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--none", required=True, type=Path)
    parser.add_argument("--splits", type=Path, default=CVPR / "results/splits_cvpr.json")
    parser.add_argument("--code", type=Path, default=CVPR / "code_full")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    audit.require(args.out.suffix == ".json" and not args.out.exists() and not args.out.with_suffix(".md").exists(),
                  "Use new JSON and Markdown output paths")
    result = analyze(args.pretrained, args.none, args.splits, code=args.code)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.out.with_suffix(".md").open("x") as handle:
        handle.write(table(result))
    print(json.dumps({"out": str(args.out), "sha256": audit.digest(args.out)}))


if __name__ == "__main__":
    main()
