"""Aggregate selector seeds 0/1/2 on the same BVI VAL16 and one E12 base.

Average paired sequence effects across selector models FIRST, then bootstrap
whole scenes. The 48 seed-by-sequence observations are never independent units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_correction_val import (ALL_METRICS, COMPARISONS, CVPR, METHODS,
                                    METRICS, analyze, digest, require, scene_condition)


def aggregate(paths, splits=CVPR / "results/splits_cvpr.json"):
    require(len(paths) == 3, "Exactly three actual VAL JSONs are required")
    analyzed = [analyze(path, splits) for path in paths]
    raw = {a["training_seed"]: json.loads(Path(path).read_text()) for path, a in zip(paths, analyzed)}
    records = {a["training_seed"]: a for a in analyzed}
    require(set(records) == {0, 1, 2} and len(raw) == 3, "Expected selector seeds 0, 1 and 2 once each")
    reference = raw[0]["selector_training"]
    origin = raw[0]["base_cache_provenance"]
    require(origin["base_saved_args"].get("seed") == 0
            and origin["base_saved_args"].get("tag") == "e12_bvi_pretrained_s0",
            "This analysis requires the one fixed E12 BVI base seed0")
    names = records[0]["sequence_names"]
    for seed, report in raw.items():
        trained = report["selector_training"]
        require(report["base_cache_provenance"] == origin, "Base/cache provenance changed across selector seeds")
        require(report["source_sha256"] == raw[0]["source_sha256"], "VAL source changed across seeds")
        for key in ("split", "normalization", "features", "architecture", "source_sha256",
                    "cache_provenance_sha256", "cache_summary_sha256"):
            require(trained[key] == reference[key], f"Matched selector setting changed: {key}")
        require({k: v for k, v in trained["args"].items() if k != "seed"}
                == {k: v for k, v in reference["args"].items() if k != "seed"},
                "Selector budgets/optimizer/device differ beyond seed")
        require(records[seed]["sequence_names"] == names, "VAL names differ across seeds")
        require(report["validation_pair_inventory"] == raw[0]["validation_pair_inventory"],
                "Validation frame pairing changed across seeds")
    for key in ("initial_state_sha256", "sample_plan_sha256"):
        require(len({raw[s]["selector_training"][key] for s in (0, 1, 2)}) == 3,
                f"Independent selector seeds unexpectedly reuse {key}")

    indexed = {seed: {row["name"]: row for row in report["per_sequence"]} for seed, report in raw.items()}
    fixed_differences, fixed_mean_absolute_differences, fixed_sequence_differences = {}, {}, {}
    for method in ("base", "fixed_average", "initial_thirds"):
        fixed_differences[method] = {}
        fixed_mean_absolute_differences[method] = {}
        fixed_sequence_differences[method] = {}
        for metric in ALL_METRICS:
            differences = np.array([abs(indexed[s][n]["methods"][method][metric]
                                        - indexed[0][n]["methods"][method][metric])
                                    for s in (1, 2) for n in names])
            maximum = float(differences.max())
            if method == "base" or metric == "gt_tlp":
                require(maximum == 0, f"Base/GT output changed: {method}/{metric}")
            fixed_differences[method][metric] = maximum
            fixed_mean_absolute_differences[method][metric] = float(differences.mean())
            fixed_sequence_differences[method][metric] = {
                str(s): {n: indexed[s][n]["methods"][method][metric] - indexed[0][n]["methods"][method][metric]
                         for n in names} for s in (1, 2)}
    scenes = sorted({scene_condition(name)[0] for name in names})
    groups = [np.array([i for i, name in enumerate(names) if scene_condition(name)[0] == scene]) for scene in scenes]
    sizes = np.array([len(group) for group in groups])
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    absolute = {}
    for method in METHODS:
        absolute[method] = {}
        for metric in METRICS:
            per_seed = [records[s]["absolute"][method][metric] for s in (0, 1, 2)]
            absolute[method][metric] = {"mean": float(np.mean(per_seed)),
                                        "range": [min(per_seed), max(per_seed)],
                                        "per_selector_seed": dict(zip(("0", "1", "2"), per_seed))}
    comparisons = {}
    for left, right in COMPARISONS:
        label = f"{left}_minus_{right}"
        comparisons[label] = {}
        for metric in METRICS:
            effects = np.array([[records[s]["comparisons"][label][metric]["per_sequence_differences"][n]
                                 for n in names] for s in (0, 1, 2)])
            averaged = effects.mean(0)
            sums = np.array([averaged[group].sum() for group in groups])
            bootstrap = sums[draws].sum(1) / sizes[draws].sum(1)
            means = effects.mean(1)
            favorable = 1 if metric.startswith("psnr") else -1
            comparisons[label][metric] = {
                "difference": float(averaged.mean()), "scene_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
                "per_selector_seed_difference": dict(zip(("0", "1", "2"), means.tolist())),
                "selector_seed_difference_range": [float(means.min()), float(means.max())],
                "selector_seeds_better": int(np.sum(favorable * means > 0)),
                "sequences_better_after_seed_average": int(np.sum(favorable * averaged > 0)),
                "scenes_better_after_seed_average": int(np.sum(favorable * sums > 0)),
                "seed_averaged_sequence_differences": dict(zip(names, averaged.tolist())),
                "seed_averaged_scene_differences": dict(zip(scenes, (sums / sizes).tolist()))}
    drift_to_gain = {}
    for method in ("fixed_average", "initial_thirds"):
        drift_to_gain[method] = {}
        for metric in ("psnr_rgb", "psnr_y"):
            gain = abs(comparisons["direct_mse_minus_base"][metric]["difference"])
            drift_to_gain[method][metric] = {
                "reference_absolute_direct_minus_base_gain": gain,
                "max_drift_fraction_of_gain": fixed_differences[method][metric] / gain if gain else None,
                "mean_drift_fraction_of_gain": fixed_mean_absolute_differences[method][metric] / gain if gain else None}
    return {"scope": "Exploratory frozen BVI VAL16; selector seeds0/1/2; one fixed E12 base seed0",
            "selector_seeds": [0, 1, 2], "base_training_seeds": [0], "n_sequences": 16, "n_scenes": 8,
            "bootstrap": {"samples": 10000, "rng_seed": 0,
                          "protocol": "Average each sequence's paired effect over three selector models, then shared whole-scene draws; total sums/counts",
                          "interval_scope": "Conditional on these three fitted selectors, one base and their actual per-run candidate computations; unadjusted percentile95; no training-seed resampling or selection correction",
                          "independent_units": "Eight scenes, never 48 seed-by-sequence observations"},
            "absolute": absolute, "comparisons": comparisons, "fixed_method_max_differences": fixed_differences,
            "fixed_method_mean_absolute_differences_vs_seed0": fixed_mean_absolute_differences,
            "fixed_method_per_sequence_differences_vs_seed0": fixed_sequence_differences,
            "fixed_control_psnr_drift_relative_to_direct_gain": drift_to_gain,
            "candidate_repeatability": "Fixed-average/initial-thirds metric drift is reported, not assumed zero. Candidates are shared within each run; cross-run changes are not attributable solely to selector initialization. Their numerical source is not isolated by this analysis.",
            "base_checkpoint_sha256": origin["base_checkpoint_sha256"],
            "inputs": {str(s): records[s]["input"] for s in (0, 1, 2)},
            "selector_checkpoint_sha256": {str(s): records[s]["selector_checkpoint_sha256"] for s in (0, 1, 2)},
            "source_sha256": raw[0]["source_sha256"], "split_sha256": digest(splits),
            "analyzer_sha256": digest(__file__),
            "single_seed_analyzer_sha256": digest(Path(__file__).with_name("analyze_correction_val.py")),
            "numpy_version": np.__version__}


def table(result):
    lines = ["# Correction selector: three selector seeds on BVI VAL16", "", result["scope"], "",
             "The base remains E12 seed0. These are three selector fits, not three independent base trainings.", "",
             "| Method | Metric | Mean | Range across selector seeds |", "|---|---|---:|---|"]
    for method, metrics in result["absolute"].items():
        for metric, value in metrics.items():
            lo, hi = value["range"]
            lines.append(f"| {method} | {metric} | {value['mean']:.8f} | [{lo:.8f}, {hi:.8f}] |")
    lines += ["", "| Contrast | Metric | Mean difference | Scene 95% CI | Favorable selector seeds / 3 |",
              "|---|---|---:|---|---:|"]
    for label, metrics in result["comparisons"].items():
        for metric, value in metrics.items():
            lo, hi = value["scene_ci95"]
            lines.append(f"| {label} | {metric} | {value['difference']:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {value['selector_seeds_better']} |")
    lines += ["", "| Fixed control | Metric | Maximum absolute drift vs seed0 | Mean absolute drift (seeds1/2 vs seed0) |",
              "|---|---|---:|---:|"]
    for method, metrics in result["fixed_method_max_differences"].items():
        for metric in METRICS:
            mean_drift = result["fixed_method_mean_absolute_differences_vs_seed0"][method][metric]
            lines.append(f"| {method} | {metric} | {metrics[metric]:.8f} | {mean_drift:.8f} |")
    lines += ["", "| Fixed control | Metric | Max drift / direct-vs-base gain | Mean drift / gain |",
              "|---|---|---:|---:|"]
    for method, metrics in result["fixed_control_psnr_drift_relative_to_direct_gain"].items():
        for metric, value in metrics.items():
            maximum, mean = value["max_drift_fraction_of_gain"], value["mean_drift_fraction_of_gain"]
            if maximum is not None:
                lines.append(f"| {method} | {metric} | {maximum:.6%} | {mean:.6%} |")
    lines += ["", "First average each sequence's paired effect across selector seeds, then resample eight complete scenes (10,000 draws, RNG0).",
              "Intervals are unadjusted and conditional on these three selectors, one base and the actual candidate computations. Repeated sequences are not counted as 48 independent observations.",
              result["candidate_repeatability"],
              "PSNR positive is favorable; LPIPS/tLP gap negative is favorable. VAL selection, base-seed uncertainty and SOTA claims are outside this analysis."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    parser.add_argument("--splits", type=Path, default=CVPR / "results/splits_cvpr.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.out.suffix == ".json", "--out must be JSON")
    require(not args.out.exists() and not args.out.with_suffix(".md").exists(), "Refusing to overwrite analysis")
    result = aggregate(args.inputs, args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(table(result))
    print(json.dumps({"output": str(args.out), "sha256": digest(args.out), "scope": result["scope"]}))


if __name__ == "__main__":
    main()
