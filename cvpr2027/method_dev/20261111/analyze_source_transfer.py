"""Audit a source-trained three-family/three-selector-seed BVI VAL16 run.

No model, frame or checkpoint loading. Average paired sequence effects over
selector seeds before resampling eight scenes; one fixed E17 base remains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

CVPR = Path(__file__).resolve().parents[2]
FAMILIES = ("restored", "restored_detail", "raw_detail")
LOSSES = ("direct_mse", "direct_mse_normalized")
METRICS = ("psnr_rgb", "psnr_y", "lpips", "tlp_gap")
ALL_METRICS = (*METRICS, "tlp", "gt_tlp")
BASE_SHAS = {"pretrained": "5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740",
             "none": "fa3c0c9e720a28b56f01069f7a2cd407bc5b4d752a2a1d4482f1bd63557b5516"}
RAFT_WEIGHTS_SHA = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
BACKEND = "CPU float32 for all families, source and target"
BASE_SOURCE_KEYS = frozenset((
    "tools/cache_source_corrections.py", "tools/cache_correction_oracle.py",
    "tools/eval_e4.py", "tools/checkpoint_config.py", "models/pixel_branch_v2.py",
    "models/vjepa_backbone.py", "models/token_projection.py", "data/common.py",
    "data/planned_bvi.py", "tools/train_method_pilot.py", "tools/train_correction_selector.py"))
TRAIN_SOURCE_KEYS = BASE_SOURCE_KEYS | {
    "tools/train_source_selector.py", "models/correction_selector.py",
    "models/correction_risk.py", "models/observation_candidates.py"}
EVAL_SOURCE_KEYS = BASE_SOURCE_KEYS | {
    "tools/eval_source_selector.py", "tools/eval_correction_selector.py",
    "tools/train_source_selector.py", "models/observation_candidates.py", "models/correction_selector.py"}
FEATURE_NAMES = ([f"{field}_{c}" for field in
                  ("low_mean", "base_mean", "past_correction_mean", "future_correction_mean") for c in "rgb"]
                 + ["low_rgb_variance", "base_rgb_variance", "gram_00", "gram_01", "gram_11",
                    "past_valid_fraction", "future_valid_fraction"])
NORMALIZATION_SCOPE = "Fit scenes only; population feature std; uncentered target RMS; mean base MSE"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def written_json_digest(value):
    """Rebuild the frozen writer's UTF-8 bytes, including order and final newline."""
    return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def check_sources(mapping, expected, *, code=CVPR / "code_full"):
    require(isinstance(mapping, dict), "Source SHA manifest must be an object")
    root = Path(code).resolve()
    for relative, sha in mapping.items():
        require(isinstance(relative, str) and "\\" not in relative, "Unsafe source path")
        path = PurePosixPath(relative)
        require(not path.is_absolute() and relative == path.as_posix()
                and all(part not in ("", ".", "..") for part in relative.split("/")), "Unsafe source path")
        resolved = (root / relative).resolve()
        require(resolved.is_relative_to(root) and resolved.is_file(), f"Unsafe/missing source: {relative}")
        require(valid_sha(sha) and digest(resolved) == sha, f"Runtime/repository source differs: {relative}")
    require(set(mapping) == set(expected), "Missing or unexpected frozen source hash keys")


def expected_scene_split(names):
    """Independent reconstruction; do not import the Torch-dependent trainer."""
    scenes = sorted({source_scene(n) for n in names})
    order = np.random.default_rng(20261114).permutation(len(scenes))
    nfit = 3 * len(scenes) // 4
    return {"seed": 20261114, "fit": [scenes[int(i)] for i in order[:nfit]],
            "internal_holdout": [scenes[int(i)] for i in order[nfit:]],
            "scope": "75/25 source TRAIN scenes; internal to base training; no target data or model selection"}


def expected_candidate_recipe(family):
    require(family in FAMILIES, "Unknown candidate family")
    return {"schema_version": 1, "mode": family, "offset": 2,
            "lowpass": {"type": "box", "kernel": 9, "padding": "reflect", "time_filtering": False},
            "photometric_gain": {"minimum": 0.25, "maximum": 16.0, "denominator_epsilon": 1e-4,
                                 "fit": "none; observable per-source lowpass RGB ratio"},
            "source_values": "low-light RGB detail" if family == "raw_detail" else "restored RGB",
            "candidate_clip": "source calibrated RGB and final detail candidate clamp to [0,1]",
            "detail_mask": "reflect-minpool over lowpass support; restored uses original valid mask",
            "alignment": "saved restored-RGB target-to-source flow; bilinear gather; align_corners=False",
            "interpretation": "Observable candidate controls; no independence or novelty claim"}


def check_parity(rows, names, label):
    require(isinstance(rows, list) and len(rows) == len(names)
            and sorted(r["name"] for r in rows) == sorted(names), f"{label} parity names differ")
    for row in rows:
        require(set(row) == {"name", "max_abs", "mean_abs"}, f"{label} parity schema differs")
        maximum, mean = row["max_abs"], row["mean_abs"]
        require(isinstance(maximum, (int, float)) and isinstance(mean, (int, float))
                and np.isfinite([maximum, mean]).all() and 0 <= mean <= maximum <= 2e-5,
                f"{label} candidate parity outside [0,2e-5]: {row['name']}")


def check_source_inventory(origin, names):
    args = {"data_seed": 20261113, "crop": 256, "clip_len": 16, "offset": 2,
            "flow_steps": 12, "flow_batch": 2, "patch_size": 8}
    require(origin["args"] == args, "Source crop/flow/data plan configuration differs")
    rng = np.random.default_rng(20261113)
    plan = [{"sequence_index": i, "augmentation_seed": int(rng.integers(0, 2**63 - 1))}
            for i in range(len(names))]
    require(origin["plan"] == plan, "Source augmentation plan differs")
    rows = origin["paired_frame_inventory"]
    require(len(rows) == len(names) and [r["name"] for r in rows] == names,
            "Source paired-frame inventory order/names differ")
    for row in rows:
        pairs = row["complete_basename_pairs"]
        require(isinstance(pairs, list) and len(pairs) == row["paired_frames"] and len(pairs) >= 16,
                "Incomplete source basename pairing inventory")
        require(all(isinstance(pair, list) and len(pair) == 2
                    and all(isinstance(n, str) and re.fullmatch(r"[0-9]+\.npy", n) for n in pair)
                    for pair in pairs), "Invalid source frame basenames")
        for column in zip(*pairs):
            require(list(column) == sorted(set(column)), "Source frames not distinct/sorted")
        require(row["first_last_pairs"] == [pairs[0], pairs[-1]]
                and row["matching_complete_basename_lists"] is all(a == b for a, b in pairs)
                and row["pairing"] == "sorted index; SDSD _2 passes retain different original frame numbers",
                "Source pairing metadata differs")
        pair_sha = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
        require(row["basename_pairs_sha256"] == pair_sha, "Source basename mapping SHA differs")


def check_cache_clips(clips, names):
    require(isinstance(clips, list) and len(clips) == len(names)
            and [r["name"] for r in clips] == names, "Source cache clip inventory differs")
    for row in clips:
        require(set(row) == {"name", "scene", "sha256"} and row["scene"] == source_scene(row["name"])
                and valid_sha(row["sha256"]), "Invalid source clip scene/hash")
    # The frozen cache writer emits precisely this summary. Reconstruct it to
    # cross-check the saved summary SHA without loading 11 GB of tensor files.
    return {"n": len(names), "per_sequence": [
        {"name": row["name"], "scene": row["scene"], "cache_path": f"clip_{i:03d}.pt",
         "cache_sha256": row["sha256"]} for i, row in enumerate(clips)],
        "base_state_unchanged": True, "raft_state_unchanged": True, "oracle_computed": False}


def expected_sample_plan(names, fit_scenes, seed):
    frames = [{"clip_index": i, "frame": t, "scene": source_scene(n), "name": n}
              for i, n in enumerate(names) for t in range(16)]
    fit_indices = np.array([i for i, row in enumerate(frames) if row["scene"] in fit_scenes], dtype=np.int64)
    batches = np.random.default_rng(seed).choice(fit_indices, size=(1000, 8), replace=True)
    return {"seed": seed, "batches": batches.tolist(), "frame_index": frames}


def check_normalization(norm, fit_frames):
    require(norm["fit_frames"] == fit_frames and norm["fit_patches"] == fit_frames * 1024
            and norm["source"] == NORMALIZATION_SCOPE, "Fit-only stats scope differs")
    for key, count, positive in (("feature_mean", 19, False), ("feature_std", 19, True),
                                 ("target_scale", 2, True)):
        value = np.asarray(norm[key], dtype=float)
        require(value.shape == (count,) and np.isfinite(value).all()
                and (not positive or (value > 0).all()), f"Invalid normalization {key}")
    for key in ("energy_floor", "direct_mse_normalizer"):
        value = norm[key]
        require(isinstance(value, (int, float)) and np.isfinite(value) and value > 0,
                f"Invalid normalization {key}")


def source_scene(name):
    require(re.fullmatch(r"SDSD_(?:indoor|outdoor)/[^/\\]+", name) is not None, "Invalid source name")
    return re.sub(r"_2$", "", name)


def target_scene(name):
    match = re.fullmatch(r"BVI/([^/\\]+)__(?:10fps|20fps|fps10|fps20)", name)
    require(match is not None, "Invalid BVI target name")
    return match[1]


def analyze(path, splits=CVPR / "results/splits_cvpr.json", *, base_arm="pretrained"):
    require(base_arm in BASE_SHAS, "Unknown base arm")
    raw = json.loads(Path(path).read_text())
    split = json.loads(Path(splits).read_text())
    names = sorted("BVI/" + n for n in split["BVI_RLV"]["val"])
    scenes = sorted({target_scene(n) for n in names})
    require(len(names) == 16 and len(set(names)) == 16 and len(scenes) == 8
            and all(sum(target_scene(n) == s for n in names) == 2 for s in scenes), "Frozen VAL16 differs")
    require(raw["domain"] == "bvi" and raw["split"] == "val" and raw["n"] == 16, "Not BVI VAL16")
    require(raw["state_unchanged"] == {"base": True, "raft": True, "all_selectors": True}, "State changed")
    require(raw["candidate_backend"] == BACKEND, "Backend differs")
    rows = {r["name"]: r["methods"] for r in raw["per_sequence"]}
    require(len(raw["per_sequence"]) == 16 and sorted(rows) == names, "Missing/duplicate target rows")
    origin = raw["base_cache_provenance"]
    source_names = [f"{domain}/{n}" for domain in ("SDSD_indoor", "SDSD_outdoor") for n in split[domain]["train"]]
    source_scenes = {source_scene(n) for n in source_names}
    require(len(source_names) == len(set(source_names)) == 136 and len(source_scenes) == 68, "Source136/68 differs")
    for domain, count in (("SDSD_indoor", 46), ("SDSD_outdoor", 90)):
        entry = split[domain]
        sets = [{source_scene(f"{domain}/{n}") for n in entry[part]} for part in ("train", "val", "test")]
        require(len(entry["train"]) == count and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]),
                "Frozen source split scene isolation differs")
    require(origin["domain"] == "sdsd" and origin["source_names"] == source_names
            and origin["split_sha256"] == digest(splits) and origin["base_checkpoint_sha256"] == BASE_SHAS[base_arm],
            "Unexpected or target-trained base/cache")
    require(origin["base_saved_args"]["tag"] == f"e17_small_{base_arm}_s0"
            and origin["base_saved_args"]["seed"] == 0 and origin["base_saved_args"]["iters"] == 20000
            and origin["base_saved_args"]["vjepa"] == base_arm
            and origin["base_saved_args"].get("heldout") is None and origin["base_saved_args"].get("only") is None,
            "E17 base identity differs")
    for key in ("base_state_sha256", "raft_weights_sha256", "raft_state_sha256"):
        require(valid_sha(origin[key]), f"Missing/invalid source state/weight hash: {key}")
    require(origin["raft_weights_sha256"] == RAFT_WEIGHTS_SHA, "Predeclared RAFT-small weights differ")
    require(origin["encoder_eval"] is True and
            (valid_sha(origin["encoder_weights_sha256"]) if base_arm == "pretrained"
             else origin["encoder_weights_sha256"] is None), "Encoder provenance differs")
    check_source_inventory(origin, source_names)
    check_sources(origin["source_sha256"], BASE_SOURCE_KEYS)
    check_sources(raw["source_sha256"], EVAL_SOURCE_KEYS)
    expected_split = expected_scene_split(source_names)
    require(len(expected_split["fit"]) == 51 and len(expected_split["internal_holdout"]) == 17,
            "Source51/17 reconstruction differs")
    expected_groups = {f"{f}_s{s}" for f in FAMILIES for s in (0, 1, 2)}
    training = raw["selector_training"]
    require(set(training) == expected_groups, "Need all nine family/seed groups")
    expected_methods = {"base"} | {f"{f}/{fixed}" for f in FAMILIES for fixed in ("fixed_average", "initial_thirds")} | {
        f"{group}/{loss}" for group in expected_groups for loss in LOSSES}
    require(set(raw["means"]) == expected_methods
            and all(set(r) == expected_methods for r in rows.values()), "Need all25 predeclared methods")
    require(set(raw["selector_checkpoint_sha256"]) == {f"{g}/{loss}" for g in expected_groups for loss in LOSSES}
            and all(valid_sha(sha) for sha in raw["selector_checkpoint_sha256"].values()),
            "Selector checkpoint hash coverage differs")
    reference = training["restored_s0"]
    check_sources(reference["source_sha256"], TRAIN_SOURCE_KEYS)
    cache_summary = check_cache_clips(reference["cache_clips"], source_names)
    provenance_sha, summary_sha = written_json_digest(origin), written_json_digest(cache_summary)
    plan_shas = {seed: written_json_digest(expected_sample_plan(source_names, expected_split["fit"], seed))
                 for seed in (0, 1, 2)}
    for family in FAMILIES:
        for seed in (0, 1, 2):
            p = training[f"{family}_s{seed}"]
            args = p["args"]
            require(args["candidate_mode"] == family and args["seed"] == seed and args["steps"] == 1000
                    and args["modes"] == list(LOSSES) and args["lr"] == 1e-3 and args["batch_frames"] == 8
                    and args["optimizer"] == "Adam default betas/eps, no weight decay" and args["device"] == "cpu",
                    "Training budget/family/seed differs")
            require(p["cache_provenance"] == origin and p["split"] == expected_split
                    and p["cache_clips"] == reference["cache_clips"] and p["source_sha256"] == reference["source_sha256"]
                    and p["cache_provenance_sha256"] == provenance_sha and p["cache_summary_sha256"] == summary_sha,
                    "Matched source data/split/code differs")
            require(p["candidate_backend"] == BACKEND and p["features"] == FEATURE_NAMES
                    and p["candidate_recipe"] == expected_candidate_recipe(family), "Candidate recipe/backend/features differ")
            require(p["base_may_have_seen_internal_holdout"] is True
                    and p["initial_weights"] == [1 / 3, 1 / 3] and p["initial_none_weight"] == 1 / 3
                    and p["selection"] == "No target data, intermediate holdout or best checkpoint selection",
                    "Training initialization/selection scope differs")
            check_parity(p["restored_cpu_vs_cache_parity"], source_names if family == "restored" else [], "Source")
            fit, internal = set(p["split"]["fit"]), set(p["split"]["internal_holdout"])
            require(len(fit) == 51 and len(internal) == 17 and not fit & internal and fit | internal == source_scenes,
                    "Source51/17 scene isolation differs")
            fit_frames = sum(source_scene(n) in fit for n in source_names) * 16
            check_normalization(p["normalization"], fit_frames)
            require(p["normalization"]["direct_mse_normalizer"] == reference["normalization"]["direct_mse_normalizer"],
                    "Shared base fit-MSE loss normalizer differs")
            if seed:
                require(p["normalization"] == training[f"{family}_s0"]["normalization"], "Family stats changed by seed")
                require(p["restored_cpu_vs_cache_parity"] == training[f"{family}_s0"]["restored_cpu_vs_cache_parity"],
                        "Source reconstruction parity changed by selector seed")
            for key in ("initial_state_sha256", "sample_plan_sha256"):
                require(valid_sha(p[key]) and p[key] == training[f"restored_s{seed}"][key], "Family initialization/plan mismatch")
            require(p["sample_plan_sha256"] == plan_shas[seed], "Fixed fit-only sample plan SHA differs")
    for key in ("initial_state_sha256", "sample_plan_sha256"):
        require(len({training[f"restored_s{s}"][key] for s in (0, 1, 2)}) == 3, "Repeated selector seed")
    inventory = raw["validation_pair_inventory"]
    require(len(inventory) == 16 and sorted(r["name"] for r in inventory) == names
            and all(r["matching_complete_basename_lists"] is True and isinstance(r["paired_frames"], int)
                    and r["paired_frames"] >= 16
                    and r["centre_start"] == (r["paired_frames"] - 16) // 2 for r in inventory), "Target frame pairing differs")
    check_parity(raw["restored_cpu_vs_npu_parity"], names, "Target")

    values = {m: {k: np.array([rows[n][m][k] for n in names], dtype=float) for k in ALL_METRICS}
              for m in expected_methods}
    for method, metrics in values.items():
        require(set(raw["means"][method]) == set(ALL_METRICS)
                and all(set(rows[name][method]) == set(ALL_METRICS) for name in names), "Metric schema differs")
        for metric, a in metrics.items():
            require(np.isfinite(a).all() and np.isclose(a.mean(), raw["means"][method][metric], atol=1e-10, rtol=0),
                    f"Nonfinite or incorrect reported mean: {method}/{metric}")
        require(np.allclose(metrics["tlp_gap"], np.abs(metrics["tlp"] - metrics["gt_tlp"]), atol=1e-12, rtol=0),
                "Incorrect temporal gap")
        require(np.array_equal(metrics["gt_tlp"], values["base"]["gt_tlp"]), "GT reference differs")
    fixed_methods = ["base"] + [f"{f}/{fixed}" for f in FAMILIES for fixed in ("fixed_average", "initial_thirds")]
    absolute = {m: {k: {"mean": float(values[m][k].mean())} for k in METRICS} for m in fixed_methods}
    for family in FAMILIES:
        for loss in LOSSES:
            label = f"{family}/{loss}"
            absolute[label] = {}
            for metric in METRICS:
                per_seed = [float(values[f"{family}_s{s}/{loss}"][metric].mean()) for s in (0, 1, 2)]
                absolute[label][metric] = {"mean": float(np.mean(per_seed)), "range": [min(per_seed), max(per_seed)],
                                          "per_seed": per_seed}
    draws = np.random.default_rng(0).integers(0, 8, size=(10000, 8))
    comparisons = {}
    pairs = [(f, "base", loss) for f in FAMILIES for loss in LOSSES]
    pairs += [(f, "restored", loss) for f in FAMILIES[1:] for loss in LOSSES]
    pairs += [("raw_detail", "restored_detail", loss) for loss in LOSSES]
    pairs += [(f, f"{f}/{fixed}", loss) for f in FAMILIES
              for fixed in ("fixed_average", "initial_thirds") for loss in LOSSES]
    for left, right, loss in pairs:
        label = f"{left}_minus_{right.rsplit('/', 1)[-1]}/{loss}"
        comparisons[label] = {}
        for metric in METRICS:
            effects = np.array([values[f"{left}_s{s}/{loss}"][metric]
                                - values[right if right in fixed_methods else f"{right}_s{s}/{loss}"][metric]
                                for s in (0, 1, 2)])
            averaged = effects.mean(0)
            scene_means = np.array([averaged[[target_scene(n) == scene for n in names]].mean() for scene in scenes])
            ci = np.quantile(scene_means[draws].mean(1), [.025, .975])
            means = effects.mean(1)
            sign = 1 if metric.startswith("psnr") else -1
            comparisons[label][metric] = {"difference": float(averaged.mean()), "scene_ci95": ci.tolist(),
                "per_selector_seed_difference": means.tolist(), "selector_seeds_better": int((sign * means > 0).sum()),
                "seed_averaged_sequence_differences": dict(zip(names, averaged.tolist())),
                "seed_averaged_scene_differences": dict(zip(scenes, scene_means.tolist()))}
    return {"scope": "Exploratory SDSD-only restoration/selector training to BVI VAL16; one base seed0, three selector seeds",
            "n_sequences": 16, "n_scenes": 8, "selector_seeds": [0, 1, 2], "base_seeds": [0], "base_arm": base_arm,
            "absolute": absolute, "comparisons": comparisons,
            "bootstrap": "Average three selector effects per sequence, then resample eight complete two-condition scenes,10k,RNG0; unadjusted percentile95; conditional on fitted models",
            "input_sha256": digest(path), "analyzer_sha256": digest(__file__), "split_sha256": digest(splits),
            "base_checkpoint_sha256": origin["base_checkpoint_sha256"], "source_sha256": raw["source_sha256"],
            "audit": {"cache_provenance_sha256": provenance_sha, "cache_summary_sha256": summary_sha,
                      "source_scene_split": expected_split, "sample_plan_sha256": plan_shas,
                      "cache_verification": "Embedded inventory and frozen JSON serialization SHA verified; tensor files not loaded",
                      "normalization_verification": "Fit scene/count, finite values and cross-seed equality checked; statistics not refit from tensors",
                      "candidate_backend": BACKEND, "candidate_parity_max_abs_tolerance": 2e-5,
                      "fixed_controls_retained": fixed_methods[1:]},
            "numpy_version": np.__version__, "interpretation": "No target training/normalization or oracle. BVI VAL informed prior development; no target-blind-history or SOTA claim."}


def table(result):
    lines = ["# Source-trained temporal correction on BVI VAL16", "", result["scope"], "",
             "| Arm | PSNR-RGB | PSNR-Y | LPIPS | tLP gap |", "|---|---:|---:|---:|---:|"]
    for arm, metrics in result["absolute"].items():
        lines.append("| " + arm + " | " + " | ".join(f"{metrics[k]['mean']:.8f}" for k in METRICS) + " |")
    lines += ["", "| Paired contrast | Metric | Mean difference | Scene 95% CI | Favorable seeds / 3 |",
              "|---|---|---:|---|---:|"]
    for name, metrics in result["comparisons"].items():
        for metric, row in metrics.items():
            lo, hi = row["scene_ci95"]
            lines.append(f"| {name} | {metric} | {row['difference']:+.8f} | [{lo:+.8f},{hi:+.8f}] | {row['selector_seeds_better']} |")
    lines += ["", result["bootstrap"], "", result["interpretation"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=CVPR / "results/splits_cvpr.json")
    parser.add_argument("--base-arm", choices=tuple(BASE_SHAS), default="pretrained")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.out.suffix == ".json" and not args.out.exists() and not args.out.with_suffix(".md").exists(),
            "Use a new JSON output path")
    result = analyze(args.input, args.splits, base_arm=args.base_arm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(table(result))
    print(json.dumps({"out": str(args.out), "sha256": digest(args.out), "absolute": result["absolute"]}))


if __name__ == "__main__":
    main()
