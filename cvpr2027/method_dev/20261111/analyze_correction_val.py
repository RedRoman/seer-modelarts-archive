"""Paired scene bootstrap for the frozen correction-selector BVI VAL16 run.

Reads result/split JSON only. This is exploratory validation of one trained
seed, not a test-set result, equivalence test, or SOTA claim. No model runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

CVPR = Path(__file__).resolve().parents[2]
MODES = ("risk_full", "direct_mse", "direct_mse_normalized")
METHODS = ("base", "fixed_average", "initial_thirds", *MODES)
METRICS = ("psnr_rgb", "psnr_y", "lpips", "tlp_gap")
ALL_METRICS = (*METRICS, "tlp", "gt_tlp")
EVALUATOR_SHA = "5d99715af4ba3084c342a0b6086e278da969fc2669515eef6b8046b74d204853"
COMPARISONS = tuple((name, "base") for name in (*MODES, "fixed_average")) + (
    ("risk_full", "direct_mse"), ("risk_full", "direct_mse_normalized"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_condition(name):
    match = re.fullmatch(r"BVI/([^/\\]+)__(?:(10|20)fps|fps(10|20))", name)
    require(match is not None, f"Invalid BVI sequence name: {name}")
    return match[1], match[2] or match[3]


def analyze(path, splits=CVPR / "results/splits_cvpr.json"):
    raw = json.loads(Path(path).read_text())
    split = json.loads(Path(splits).read_text())["BVI_RLV"]
    names = sorted(f"BVI/{name}" for name in split["val"])
    pairs = [scene_condition(name) for name in names]
    scenes = sorted({scene for scene, _ in pairs})
    counts = Counter(pairs)
    require(len(names) == len(set(names)) == 16 and len(scenes) == 8
            and all(counts[(s, c)] == 1 for s in scenes for c in ("10", "20")),
            "Expected frozen VAL16: eight scenes, each with conditions 10 and 20")
    require(raw.get("domain") == "bvi" and raw.get("split") == "val" and raw.get("n") == 16,
            "Expected a BVI VAL16 report")
    rows = raw["per_sequence"]
    require(len(rows) == 16 and sorted(row["name"] for row in rows) == names,
            "Result sequence names differ from the frozen validation split")
    indexed = {row["name"]: row for row in rows}
    require(set(raw["means"]) == set(METHODS), "Unexpected method coverage in means")
    values = {}
    for row in rows:
        require(set(row["methods"]) == set(METHODS), "Unexpected per-sequence method coverage")
        require(all(set(score) == set(ALL_METRICS) for score in row["methods"].values()),
                "Unexpected metric coverage")
    for method in METHODS:
        values[method] = {}
        for metric in ALL_METRICS:
            array = np.array([indexed[name]["methods"][method][metric] for name in names], dtype=np.float64)
            require(np.isfinite(array).all(), f"Nonfinite {method}/{metric}")
            require(np.isclose(array.mean(), raw["means"][method][metric], atol=1e-10, rtol=0),
                    f"Reported mean differs from rows: {method}/{metric}")
            values[method][metric] = array
        require(np.allclose(values[method]["tlp_gap"],
                            np.abs(values[method]["tlp"] - values[method]["gt_tlp"]), atol=1e-12, rtol=0),
                f"Temporal gap is not abs(pred-GT): {method}")
    gt_difference = max(float(np.max(np.abs(values[m]["gt_tlp"] - values["base"]["gt_tlp"])))
                        for m in METHODS)
    require(gt_difference <= 1e-7, "GT temporal reference differs between methods")

    origin, trained = raw["base_cache_provenance"], raw["selector_training"]
    require(origin["split_sha256"] == digest(splits), "Frozen split SHA mismatch")
    require(trained["cache_provenance"] == origin, "Training/cache provenance differs")
    require(trained["args"]["steps"] == 1000 and trained["args"]["modes"] == list(MODES),
            "Expected three matched last-step1000 selectors")
    fit, internal = set(trained["split"]["fit"]), set(trained["split"]["internal_holdout"])
    train = {scene_condition(f"BVI/{name}")[0] for name in split["train"]}
    require(len(fit) == 18 and len(internal) == 6 and not fit & internal
            and fit | internal == train and not train & set(scenes), "Training/VAL scene partition mismatch")
    require(set(raw["selector_checkpoint_sha256"]) == set(MODES), "Missing selector checkpoint SHA")
    require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in raw["selector_checkpoint_sha256"].values()),
            "Invalid recorded selector SHA")
    require(raw["source_sha256"]["tools/eval_correction_selector.py"] == EVALUATOR_SHA,
            "Unexpected evaluator version")
    for relative, sha in raw["source_sha256"].items():
        for manifest in (origin["source_sha256"], trained["source_sha256"]):
            if relative in manifest:
                require(sha == manifest[relative], f"Inconsistent recorded source SHA: {relative}")
    inventory = raw["validation_pair_inventory"]
    require(len(inventory) == 16 and sorted(row["name"] for row in inventory) == names,
            "Paired-frame inventory differs from VAL16")
    require(all(row["matching_complete_basename_lists"] is True and row["paired_frames"] >= 16
                and row["centre_start"] == (row["paired_frames"] - 16) // 2 for row in inventory),
            "Invalid recorded paired-frame preflight")

    groups = [np.array([i for i, pair in enumerate(pairs) if pair[0] == scene]) for scene in scenes]
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    group_sizes = np.array([len(group) for group in groups])
    comparisons = {}
    for left, right in COMPARISONS:
        output = comparisons[f"{left}_minus_{right}"] = {}
        for metric in METRICS:
            delta = values[left][metric] - values[right][metric]
            sums = np.array([delta[group].sum() for group in groups])
            replicates = sums[draws].sum(1) / group_sizes[draws].sum(1)
            sign = 1 if metric.startswith("psnr") else -1
            output[metric] = {
                "difference": float(delta.mean()), "scene_ci95": np.quantile(replicates, [.025, .975]).tolist(),
                "sequences_better": int(np.sum(sign * delta > 0)),
                "scenes_better": int(np.sum(sign * sums > 0)),
                "per_sequence_differences": dict(zip(names, delta.tolist())),
                "per_scene_differences": dict(zip(scenes, (sums / group_sizes).tolist()))}
    return {"scope": "Exploratory BVI VAL16, eight scenes, one trained seed; not TEST/SOTA/equivalence evidence",
            "training_seed": trained["args"]["seed"], "n_sequences": 16, "n_scenes": 8,
            "bootstrap": {"samples": 10000, "rng_seed": 0,
                          "protocol": "Shared whole-scene draws across metrics/contrasts; total paired sums / sequence counts",
                          "intervals": "Unadjusted percentile 95%; conditional on this seed; no selection correction"},
            "sequence_names": names, "scene_conditions": {s: [n for n in names if scene_condition(n)[0] == s] for s in scenes},
            "gt_temporal_max_difference": gt_difference,
            "absolute": {m: {k: float(values[m][k].mean()) for k in METRICS} for m in METHODS},
            "comparisons": comparisons, "input": {"path": str(path), "sha256": digest(path)},
            "split_sha256": digest(splits), "analyzer_sha256": digest(__file__), "numpy_version": np.__version__,
            "selector_checkpoint_sha256": raw["selector_checkpoint_sha256"],
            "evaluation_source_sha256": raw["source_sha256"],
            "provenance_scope": "JSON/source/checkpoint hash records cross-checked; no checkpoint or frame bytes loaded"}


def table(result):
    lines = ["# Correction selector: exploratory BVI VAL16", "", result["scope"], "",
             "| Method | PSNR-RGB ↑ | PSNR-Y ↑ | LPIPS ↓ | tLP gap ↓ |",
             "|---|---:|---:|---:|---:|"]
    for method, row in result["absolute"].items():
        lines.append(f"| {method} | {row['psnr_rgb']:.6f} | {row['psnr_y']:.6f} | {row['lpips']:.8f} | {row['tlp_gap']:.8f} |")
    lines += ["", "Differences are left minus right. Intervals resample eight whole scenes (10,000 draws, RNG0).",
              "PSNR positive is favorable; LPIPS/tLP gap negative is favorable. Intervals are unadjusted and conditional on one training seed.", "",
              "| Contrast | Metric | Difference | Scene 95% CI | Better sequences / 16 |",
              "|---|---|---:|---|---:|"]
    for comparison, metrics in result["comparisons"].items():
        for metric, row in metrics.items():
            lo, hi = row["scene_ci95"]
            lines.append(f"| {comparison} | {metric} | {row['difference']:+.8f} | [{lo:+.8f}, {hi:+.8f}] | {row['sequences_better']} |")
    lines += ["", "No oracle is scored on VAL. The earlier six-scene selector holdout was internal to TRAIN; this report uses the separate frozen VAL scenes.",
              "The temporal gap is abs(tLP(pred)-tLP(GT)); a smaller value alone does not establish better fidelity or rule out oversmoothing.",
              "Raw per-sequence/scene effects and input hashes are retained in the adjacent JSON."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=CVPR / "results/splits_cvpr.json")
    parser.add_argument("--out", type=Path, required=True, help="New JSON file; Markdown table uses the same stem")
    args = parser.parse_args()
    require(args.out.suffix == ".json", "--out must end in .json")
    require(not args.out.exists() and not args.out.with_suffix(".md").exists(), "Refusing to overwrite analysis")
    result = analyze(args.input, args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(table(result))
    print(json.dumps({"analysis": str(args.out), "sha256": digest(args.out), "absolute": result["absolute"]}, indent=2))


if __name__ == "__main__":
    main()
