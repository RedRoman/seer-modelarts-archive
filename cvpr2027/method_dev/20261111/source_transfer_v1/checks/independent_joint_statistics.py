"""Recompute the real P/N and STCD comparisons without importing project analyzers."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def avg(values):
    values = list(values)
    return math.fsum(values) / len(values)


def run(p_path, n_path, stcd_path, joint_path):
    paths = {"pretrained": p_path, "none": n_path}
    data = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    stcd, expected = json.loads(stcd_path.read_text()), json.loads(joint_path.read_text())
    rows = {arm: {r["name"]: r["methods"] for r in item["per_sequence"]} for arm, item in data.items()}
    names = sorted(rows["pretrained"])
    assert len(names) == 16 and sorted(rows["none"]) == names
    scenes = sorted({name.split("__")[0] for name in names})
    grouping = [[i for i, name in enumerate(names) if name.split("__")[0] == scene] for scene in scenes]
    assert len(scenes) == 8 and all(len(group) == 2 for group in grouping)
    assert expected["input_sha256"] == {arm: sha(path) for arm, path in paths.items()}
    assert expected["stcd_reference"]["input_sha256"] == sha(stcd_path)
    p, n = data["pretrained"], data["none"]
    source_keys = ("source_names", "plan", "paired_frame_inventory", "split_sha256", "args",
                   "raft_weights_sha256", "raft_state_sha256")
    for key in source_keys:
        assert p["base_cache_provenance"][key] == n["base_cache_provenance"][key], key
    assert p["base_cache_provenance"]["raft_weights_sha256"] == (
        "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27")
    assert p["validation_pair_inventory"] == n["validation_pair_inventory"]
    for group in p["selector_training"]:
        for key in ("split", "initial_state_sha256", "sample_plan_sha256", "candidate_recipe", "features"):
            assert p["selector_training"][group][key] == n["selector_training"][group][key], (group, key)
    families = ("restored", "restored_detail", "raw_detail")
    losses = ("direct_mse", "direct_mse_normalized")
    metrics = ("psnr_rgb", "psnr_y", "lpips", "tlp_gap")
    fixed = [f"{family}/{name}" for family in families for name in ("fixed_average", "initial_thirds")]
    learned = [f"{family}/{loss}" for family in families for loss in losses]
    errors = []

    def close(actual, reference, label):
        a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
        error = float(np.max(np.abs(a - b))) if a.size else 0.0
        assert error <= 1e-12, (label, error)
        errors.append(error)

    def vector(arm, label, metric):
        if label in ["base", *fixed]:
            return [float(rows[arm][name][label][metric]) for name in names]
        family, loss = label.split("/")
        return [avg(float(rows[arm][name][f"{family}_s{seed}/{loss}"][metric]) for seed in range(3))
                for name in names]

    gt = vector("pretrained", "base", "gt_tlp")
    assert gt == vector("none", "base", "gt_tlp")
    assert expected["none_gt_tlp_max_abs_vs_pretrained"] == 0.0
    stcd_rows = {row["name"]: row for row in stcd["per_sequence"]}
    assert len(stcd["per_sequence"]) == 16 and sorted(stcd_rows) == names
    assert [stcd_rows[name]["gt_tlp"] for name in names] == gt
    assert expected["stcd_reference"]["gt_tlp_max_abs_vs_pretrained"] == 0.0
    assert stcd["provenance"]["checkpoint_sha256"] == (
        "dc50957303b2ac15d2f71a32c3d4231c47c5d3f8dc21daa52bccf48faea49cda")
    assert stcd["split_sha256"] == p["base_cache_provenance"]["split_sha256"]
    target_inventory = {r["name"]: r for r in p["validation_pair_inventory"]}
    for entry in stcd["validation_pair_inventory"]:
        other = target_inventory[entry["name"]]
        assert entry["paired_frame_count"] == other["paired_frames"]
        assert entry["centre_start"] == other["centre_start"]
    stcd_means = {}
    for metric in (*metrics, "tlp", "gt_tlp"):
        stcd_means[metric] = avg(stcd_rows[name][metric] for name in names)
        close(stcd_means[metric], stcd["means"][metric], f"STCD raw mean/{metric}")
        close(stcd_means[metric], expected["stcd_reference"]["means"][metric], f"STCD joint mean/{metric}")
    for name in names:
        close(stcd_rows[name]["tlp_gap"], abs(stcd_rows[name]["tlp"] - stcd_rows[name]["gt_tlp"]),
              f"STCD temporal gap/{name}")
    absolute = {}
    for arm in paths:
        absolute[arm] = {}
        for method in ["base", *fixed, *learned]:
            absolute[arm][method] = {}
            for metric in metrics:
                entry = {"mean": avg(vector(arm, method, metric))}
                if method in learned:
                    family, loss = method.split("/")
                    seed_values = [avg(rows[arm][name][f"{family}_s{seed}/{loss}"][metric] for name in names)
                                   for seed in range(3)]
                    entry.update(per_seed=seed_values, range=[min(seed_values), max(seed_values)])
                for key, value in entry.items():
                    close(value, expected["absolute"][arm][method][metric][key],
                          f"joint absolute/{arm}/{method}/{metric}/{key}")
                absolute[arm][method][metric] = entry
        assert set(absolute[arm]) == set(expected["absolute"][arm])

    draws = np.random.default_rng(0).integers(0, 8, (10000, 8))

    def contrast(left, right):
        effect = [a - b for a, b in zip(left, right)]
        cluster_means = [avg(effect[i] for i in group) for group in grouping]
        bootstrap = np.asarray(cluster_means)[draws].sum(axis=1) / 8
        return {"difference": avg(effect), "scene_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                "sequence_differences": dict(zip(names, effect)),
                "scene_differences": {s.removeprefix("BVI/"): v for s, v in zip(scenes, cluster_means)}}

    def compare_item(item, reference, label):
        assert set(item) == set(reference)
        for key in ("difference", "scene_ci95"):
            close(item[key], reference[key], f"{label}/{key}")
        for key in ("sequence_differences", "scene_differences"):
            assert set(item[key]) == set(reference[key])
            close(list(item[key].values()), [reference[key][k] for k in item[key]], f"{label}/{key}")

    comparisons = {"pretrained_base_minus_none_base": {
        metric: contrast(vector("pretrained", "base", metric), vector("none", "base", metric))
        for metric in metrics}}
    for method in learned:
        comparisons[f"pretrained_minus_none/{method}"] = {}
        comparisons[f"correction_gain_pretrained_minus_none/{method}"] = {}
        for metric in metrics:
            pv, nv = vector("pretrained", method, metric), vector("none", method, metric)
            pb, nb = vector("pretrained", "base", metric), vector("none", "base", metric)
            comparisons[f"pretrained_minus_none/{method}"][metric] = contrast(pv, nv)
            comparisons[f"correction_gain_pretrained_minus_none/{method}"][metric] = contrast(
                [v - base for v, base in zip(pv, pb)], [v - base for v, base in zip(nv, nb)])
    assert len(comparisons) == 13 and set(comparisons) == set(expected["comparisons"])
    for name, entries in comparisons.items():
        for metric, entry in entries.items():
            compare_item(entry, expected["comparisons"][name][metric], f"joint/{name}/{metric}")
    stcd_comparisons = {}
    for arm in paths:
        for method in ["base", *learned]:
            label = f"{arm}/{method}_minus_STCD"
            stcd_comparisons[label] = {}
            for metric in metrics:
                entry = contrast(vector(arm, method, metric), [stcd_rows[name][metric] for name in names])
                compare_item(entry, expected["stcd_reference"]["paired_comparisons"][label][metric],
                             f"STCD/{label}/{metric}")
                stcd_comparisons[label][metric] = entry
    assert len(stcd_comparisons) == 14
    assert set(stcd_comparisons) == set(expected["stcd_reference"]["paired_comparisons"])
    return {"status": "PASS", "scope": "Independent local arithmetic and shared-input audit; no project analyzer import",
            "inputs_sha256": {"pretrained": sha(p_path), "none": sha(n_path), "stcd": sha(stcd_path),
                              "joint_analysis": sha(joint_path)}, "script_sha256": sha(__file__),
            "absolute": absolute, "comparisons": comparisons,
            "stcd_reference": {"means": stcd_means, "paired_comparisons": stcd_comparisons},
            "checks": {"joint_contrasts": 13, "stcd_comparisons": 14, "metrics_per_comparison": 4,
                       "ci_endpoints": 216, "fields_compared": len(errors),
                       "maximum_numeric_difference": max(errors), "numeric_tolerance": 1e-12,
                       "pretrained_none_gt_max_abs": 0.0, "pretrained_stcd_gt_max_abs": 0.0,
                       "source_raft_weights_equal": True, "source_raft_state_equal": True,
                       "source_input_plan_pair_inventory_equal": True, "same_target_pair_inventory": True,
                       "matched_selector_initial_state_and_sample_plan": True,
                       "all_fixed_controls_absolute_retained": True},
            "raft_weights_sha256": p["base_cache_provenance"]["raft_weights_sha256"],
            "raft_state_sha256": p["base_cache_provenance"]["raft_state_sha256"],
            "bootstrap": "Average selector seeds per sequence then resample eight whole two-condition scenes; shared10k RNG0; unadjusted percentile95",
            "limits": ["Each arm has one historical base checkpoint; base initializations were not matched",
                       "Interaction describes correction gains, not an isolated causal effect of video pretraining",
                       "STCD has released SDSD-indoor-only training; training-data coverage and extra pretraining are unmatched",
                       "BVI VAL was known during earlier project development; no target-blind-history or SOTA assertion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pretrained", "none", "stcd", "joint", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pretrained, args.none, args.stcd, args.joint)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": str(args.out), "sha256": sha(args.out), **result["checks"]}))
