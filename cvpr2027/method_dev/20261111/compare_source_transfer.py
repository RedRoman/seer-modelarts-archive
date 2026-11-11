"""Compare two audited source-transfer arms and an unmatched STCD reference.

All declared methods are retained. No checkpoint ranking or target fitting.
Scene intervals are conditional on the available trained models, not training
replication. Historical base initialization differs between the two arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_source_transfer as audit


def paired_effect(left, right, names, draws):
    scenes = sorted({audit.target_scene(n) for n in names})
    effect = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    audit.require(effect.shape == (16,) and np.isfinite(effect).all(), "Invalid paired effect")
    grouped = np.array([effect[[audit.target_scene(n) == s for n in names]].mean() for s in scenes])
    return {"difference": float(effect.mean()),
            "scene_ci95": np.quantile(grouped[draws].mean(1), [.025, .975]).tolist(),
            "sequence_differences": dict(zip(names, effect.tolist())),
            "scene_differences": dict(zip(scenes, grouped.tolist()))}


def method_values(raw, names, label, metric):
    rows = {r["name"]: r["methods"] for r in raw["per_sequence"]}
    if label == "base" or label.endswith(("/fixed_average", "/initial_thirds")):
        return np.array([rows[n][label][metric] for n in names])
    family, loss = label.split("/")
    return np.array([[rows[n][f"{family}_s{s}/{loss}"][metric] for n in names]
                     for s in (0, 1, 2)]).mean(0)


def compare(pretrained, none, stcd=None):
    paths = {"pretrained": pretrained, "none": none}
    audited = {arm: audit.analyze(path, base_arm=arm) for arm, path in paths.items()}
    raw = {arm: json.loads(Path(path).read_text()) for arm, path in paths.items()}
    p, n = raw["pretrained"], raw["none"]
    names = sorted(r["name"] for r in p["per_sequence"])
    audit.require(p["validation_pair_inventory"] == n["validation_pair_inventory"], "Target inventory differs")
    for key in ("plan", "source_names", "paired_frame_inventory", "split_sha256", "args",
                "raft_weights_sha256", "raft_state_sha256"):
        audit.require(key in p["base_cache_provenance"] and key in n["base_cache_provenance"]
                      and p["base_cache_provenance"][key] == n["base_cache_provenance"][key],
                      f"Source data/plan differs: {key}")
    audit.require(p["selector_training"]["restored_s0"]["split"]
                  == n["selector_training"]["restored_s0"]["split"], "Source scene split differs")
    for seed in (0, 1, 2):
        for key in ("initial_state_sha256", "sample_plan_sha256"):
            audit.require(p["selector_training"][f"restored_s{seed}"][key]
                          == n["selector_training"][f"restored_s{seed}"][key], "Added-head initialization/plan differs")
    gt = method_values(p, names, "base", "gt_tlp")
    none_gt_difference = float(abs(gt - method_values(n, names, "base", "gt_tlp")).max())
    audit.require(none_gt_difference <= 1e-8,
                  "GT temporal scores differ between arms")
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    comparisons = {"pretrained_base_minus_none_base": {
        metric: paired_effect(method_values(p, names, "base", metric),
                              method_values(n, names, "base", metric), names, draws)
        for metric in audit.METRICS}}
    for family in audit.FAMILIES:
        for loss in audit.LOSSES:
            label = f"{family}/{loss}"
            absolute, interaction = {}, {}
            for metric in audit.METRICS:
                pv, nv = (method_values(raw[arm], names, label, metric) for arm in paths)
                pb, nb = (method_values(raw[arm], names, "base", metric) for arm in paths)
                absolute[metric] = paired_effect(pv, nv, names, draws)
                interaction[metric] = paired_effect(pv - pb, nv - nb, names, draws)
            comparisons[f"pretrained_minus_none/{label}"] = absolute
            comparisons[f"correction_gain_pretrained_minus_none/{label}"] = interaction
    reference = None
    if stcd is not None:
        baseline = json.loads(Path(stcd).read_text())
        rows = {r["name"]: r for r in baseline["per_sequence"]}
        audit.require(len(baseline["per_sequence"]) == 16 and sorted(rows) == names
                      and baseline["domain"] == "bvi" and baseline["split"] == "val"
                      and baseline["split_sha256"] == audited["pretrained"]["split_sha256"],
                      "STCD coverage/split differs")
        audit.require(baseline["provenance"]["checkpoint_sha256"]
                      == "dc50957303b2ac15d2f71a32c3d4231c47c5d3f8dc21daa52bccf48faea49cda"
                      and baseline["runtime"]["chunk"] == 1 and baseline["runtime"]["device"] == "npu:0",
                      "STCD released checkpoint/runtime differs")
        scoring = baseline["scoring_source_sha256"]
        for suffix, local in (("/tools/eval_stcd_transfer.py", audit.CVPR / "code_full/tools/eval_stcd_transfer.py"),
                              ("/tools/eval_e4.py", audit.CVPR / "code_full/tools/eval_e4.py"),
                              ("/data/common.py", audit.CVPR / "code_full/data/common.py"),
                              ("/baselines/common/eval_lib.py", Path(__file__).resolve().parent / "source_transfer_v1/references/runtime_sources/common_eval_lib.py")):
            matches = [sha for path, sha in scoring.items() if path.endswith(suffix)]
            audit.require(matches == [audit.digest(local)], f"STCD scoring source differs: {suffix}")
        baseline_values = {metric: np.array([rows[name][metric] for name in names], dtype=float)
                           for metric in audit.ALL_METRICS}
        for metric, values in baseline_values.items():
            audit.require(np.isfinite(values).all()
                          and np.isclose(values.mean(), baseline["means"][metric], atol=1e-10, rtol=0),
                          "Invalid STCD metric/mean")
        audit.require(np.allclose(baseline_values["tlp_gap"],
                                  abs(baseline_values["tlp"] - baseline_values["gt_tlp"]), atol=1e-12, rtol=0)
                      and np.allclose(gt, baseline_values["gt_tlp"], atol=1e-8, rtol=0), "STCD GT or gap differs")
        target_inventory = {r["name"]: r for r in p["validation_pair_inventory"]}
        audit.require(len(baseline["validation_pair_inventory"]) == 16
                      and sorted(r["name"] for r in baseline["validation_pair_inventory"]) == names,
                      "STCD inventory incomplete")
        for row in baseline["validation_pair_inventory"]:
            other = target_inventory[row["name"]]
            audit.require(row["paired_frame_count"] == other["paired_frames"]
                          and row["centre_start"] == other["centre_start"], "STCD frame/crop selection differs")
        reference = {"label": baseline["method"], "comparison_scope": baseline["comparison_scope"],
                     "input_sha256": audit.digest(stcd), "means": baseline["means"],
                     "gt_tlp_max_abs_vs_pretrained": float(abs(gt - baseline_values["gt_tlp"]).max()),
                     "provenance": baseline["provenance"], "paired_comparisons": {}}
        for arm in paths:
            for label in ("base", *(f"{f}/{loss}" for f in audit.FAMILIES for loss in audit.LOSSES)):
                reference["paired_comparisons"][f"{arm}/{label}_minus_STCD"] = {
                    metric: paired_effect(method_values(raw[arm], names, label, metric),
                                          baseline_values[metric], names, draws) for metric in audit.METRICS}
    return {"scope": "Exploratory SDSD-to-BVI source-only training; all declared arms; no target selection",
            "base_seeds_per_arm": [0], "selector_seeds": [0, 1, 2], "n_scenes": 8,
            "absolute": {arm: result["absolute"] for arm, result in audited.items()},
            "comparisons": comparisons, "stcd_reference": reference,
            "input_sha256": {arm: audit.digest(path) for arm, path in paths.items()},
            "none_gt_tlp_max_abs_vs_pretrained": none_gt_difference,
            "analyzer_sha256": audit.digest(__file__), "arm_analyzer_sha256": audit.digest(audit.__file__),
            "bootstrap": "Average selector seeds within sequence, then resample eight entire scenes; 10k RNG0, percentile95, unadjusted exploratory intervals",
            "limitations": ["One trained base per arm; selector seeds do not replicate base training",
                            "Historical base pixel initializations differ, so the arm contrast is not isolated causal pretraining evidence",
                            "STCD uses released SDSD-indoor training; Seer used SDSD indoor+outdoor and extra foundation pretraining",
                            "BVI VAL informed earlier project development; no target-blind-history or SOTA conclusion"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--none", required=True, type=Path)
    parser.add_argument("--stcd", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    audit.require(args.out.suffix == ".json" and not args.out.exists(), "Use a new JSON output path")
    result = compare(args.pretrained, args.none, args.stcd)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": audit.digest(args.out)}))


if __name__ == "__main__":
    main()
