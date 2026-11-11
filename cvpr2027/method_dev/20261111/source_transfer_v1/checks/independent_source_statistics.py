"""Independent arithmetic audit of real source-transfer output; imports no project code."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def run(raw_path, analysis_path):
    raw = json.loads(raw_path.read_text())
    published = json.loads(analysis_path.read_text())
    assert published["input_sha256"] == sha(raw_path)
    families = ("restored", "restored_detail", "raw_detail")
    losses = ("direct_mse", "direct_mse_normalized")
    metrics = ("psnr_rgb", "psnr_y", "lpips", "tlp_gap")
    all_metrics = (*metrics, "tlp", "gt_tlp")
    rows = {r["name"]: r["methods"] for r in raw["per_sequence"]}
    names = sorted(rows)
    assert len(raw["per_sequence"]) == len(names) == 16
    # BVI scene identity is the complete prefix before __; do not strip _2.
    scenes = sorted({n.split("__")[0] for n in names})
    groups = [[i for i, n in enumerate(names) if n.split("__")[0] == s] for s in scenes]
    assert len(scenes) == 8 and all(len(g) == 2 for g in groups)
    fixed = [f"{f}/{k}" for f in families for k in ("fixed_average", "initial_thirds")]
    heads = [f"{f}_s{s}/{loss}" for f in families for s in range(3) for loss in losses]
    methods = ["base", *fixed, *heads]
    assert set(raw["means"]) == set(methods)
    assert all(set(rows[n]) == set(methods) for n in names)
    values = {method: {metric: [float(rows[n][method][metric]) for n in names]
                       for metric in all_metrics} for method in methods}
    numeric_errors = []
    comparisons_checked = 0

    def close(actual, expected, label):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        assert difference <= 1e-12, (label, difference)
        numeric_errors.append({"field": label, "max_absolute_error": difference})

    method_means = {}
    for method in methods:
        method_means[method] = {}
        for metric in all_metrics:
            computed = mean(values[method][metric])
            close(computed, raw["means"][method][metric], f"raw means/{method}/{metric}")
            method_means[method][metric] = computed
        assert values[method]["gt_tlp"] == values["base"]["gt_tlp"]
        close(values[method]["tlp_gap"],
              [abs(a - b) for a, b in zip(values[method]["tlp"], values[method]["gt_tlp"])],
              f"absolute temporal gap/{method}")

    absolute = {method: {metric: {"mean": method_means[method][metric]} for metric in metrics}
                for method in ["base", *fixed]}
    for family in families:
        for loss in losses:
            key = f"{family}/{loss}"
            absolute[key] = {}
            for metric in metrics:
                seed_means = [method_means[f"{family}_s{s}/{loss}"][metric] for s in range(3)]
                absolute[key][metric] = {"mean": mean(seed_means), "per_seed": seed_means,
                                         "range": [min(seed_means), max(seed_means)]}
    assert set(absolute) == set(published["absolute"])
    for method, entries in absolute.items():
        for metric, entry in entries.items():
            assert set(entry) == set(published["absolute"][method][metric])
            for field, value in entry.items():
                close(value, published["absolute"][method][metric][field], f"absolute/{method}/{metric}/{field}")

    # Enumerate the declared design independently, including both fixed controls.
    specs = []
    for family in families:
        for loss in losses:
            specs.append((family, "base", loss))
            specs.extend((family, f"{family}/{control}", loss)
                         for control in ("fixed_average", "initial_thirds"))
            if family != "restored":
                specs.append((family, "restored", loss))
            if family == "raw_detail":
                specs.append((family, "restored_detail", loss))
    assert len(specs) == 24
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    contrasts = {}
    for family, reference, loss in specs:
        label = f"{family}_minus_{reference.split('/')[-1]}/{loss}"
        assert label not in contrasts
        contrasts[label] = {}
        for metric in metrics:
            per_seed = []
            for seed in range(3):
                left = values[f"{family}_s{seed}/{loss}"][metric]
                right_name = reference if reference in ["base", *fixed] else f"{reference}_s{seed}/{loss}"
                right = values[right_name][metric]
                per_seed.append([a - b for a, b in zip(left, right)])
            sequence_effects = [mean(per_seed[seed][i] for seed in range(3)) for i in range(16)]
            scene_effects = [mean(sequence_effects[i] for i in group) for group in groups]
            scene_array = np.asarray(scene_effects, dtype=np.float64)
            # Each sampled scene contains both conditions. Because all clusters
            # have size two, mean(scene means) equals ratio(total sum,total count).
            resampled = scene_array[draws].sum(axis=1) / 8
            interval = np.percentile(resampled, [2.5, 97.5]).tolist()
            seed_effects = [mean(v) for v in per_seed]
            direction = 1 if metric in ("psnr_rgb", "psnr_y") else -1
            item = {"difference": mean(sequence_effects), "scene_ci95": interval,
                    "per_selector_seed_difference": seed_effects,
                    "selector_seeds_better": sum(direction * value > 0 for value in seed_effects),
                    "seed_averaged_sequence_differences": dict(zip(names, sequence_effects)),
                    "seed_averaged_scene_differences": {
                        scene.removeprefix("BVI/"): value for scene, value in zip(scenes, scene_effects)}}
            expected = published["comparisons"][label][metric]
            for field in ("difference", "scene_ci95", "per_selector_seed_difference", "selector_seeds_better"):
                close(item[field], expected[field], f"contrast/{label}/{metric}/{field}")
            for field in ("seed_averaged_sequence_differences", "seed_averaged_scene_differences"):
                assert set(item[field]) == set(expected[field]), (label, metric, field)
                close(list(item[field].values()), [expected[field][k] for k in item[field]],
                      f"contrast/{label}/{metric}/{field}")
            contrasts[label][metric] = item
            comparisons_checked += 1
    assert set(contrasts) == set(published["comparisons"])
    return {"status": "PASS", "scope": "Independent local arithmetic audit; original metrics only, no model or project imports",
            "raw_sha256": sha(raw_path), "analysis_sha256": sha(analysis_path),
            "script_sha256": sha(__file__), "numpy_version": np.__version__,
            "n_sequences": 16, "n_scenes": 8, "selector_seeds": [0, 1, 2],
            "method_means_recomputed": method_means, "absolute": absolute, "comparisons": contrasts,
            "checks": {"methods": len(methods), "raw_metric_means": len(methods) * len(all_metrics),
                       "contrasts": len(contrasts), "contrast_metric_pairs": comparisons_checked,
                       "bootstrap_endpoints": comparisons_checked * 2,
                       "all_six_fixed_controls_present": set(fixed).issubset(absolute),
                       "exact_common_gt_reference": True, "fields_compared": len(numeric_errors),
                       "maximum_numeric_difference": max(r["max_absolute_error"] for r in numeric_errors),
                       "numeric_tolerance": 1e-12},
            "protocol": "Mean three selector effects for each sequence; resample eight paired-condition scenes; shared 10000 RNG0 draws; percentile95; no multiple-comparison adjustment",
            "limits": "Conditional on one base seed and three fitted heads. Source development knew prior BVI VAL findings; no target-blind history, causal pretraining or SOTA assertion."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.raw, args.analysis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "sha256": sha(args.out), **result["checks"]}))
