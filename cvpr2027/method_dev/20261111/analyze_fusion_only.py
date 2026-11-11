"""Two-arm frozen-base SDSD VAL analysis for pilot v2 (600 optimizer steps).

There is no baseline training arm. Each trained fusion arm is compared with its
fixed step-zero reconstruction; prior-minus-local is the matched arm contrast.
Only result JSONs and source bytes are opened, never checkpoints or images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from analyze_pilots import (CVPR, METRICS, SOURCES, digest, make_plan,
                            paired_summary, require, scene_key, verify_sources)

MODES = ("local", "prior")
PACKAGE_SHA256 = "25948bb0ef40c46199cdbb1957826516c992494fbc155637d289c262fffc5351"
TASK = CVPR.parents[1] / "seer_method_dev_20261111"


def verify_archive(manifest, path):
    """Hash committed runtime bytes without extracting or importing the archive."""
    require(set(manifest) == set(SOURCES), "Unexpected v2 source manifest coverage")
    sha = digest(path)
    require(sha == PACKAGE_SHA256, "Source archive is not the frozen pilot-v2 package")
    with tarfile.open(path, "r:gz") as archive:
        for relative, expected in manifest.items():
            matches = [member for member in archive.getmembers() if member.name == relative]
            require(len(matches) == 1 and matches[0].isfile(), f"Missing/duplicate/non-file source: {relative}")
            with archive.extractfile(matches[0]) as stream:
                require(hashlib.sha256(stream.read()).hexdigest() == expected,
                        f"Frozen archive source mismatch: {relative}")
    return {"mode": "frozen_archive", "archive": str(path), "archive_sha256": sha}


def analyze(runs, step=600, *, splits=CVPR / "results/splits_cvpr.json",
            source_archive=None, code_root=CVPR / "code_full", code_revision=None, boot=10000):
    sources = []

    def read(path):
        sources.append({"path": str(path), "sha256": digest(path)})
        return json.loads(path.read_text())

    split = read(splits)
    train_names = [f"{domain}/{name}" for domain in ("SDSD_indoor", "SDSD_outdoor")
                   for name in split[domain]["train"]]
    names = sorted(f"{domain}/{name}" for domain in ("SDSD_indoor", "SDSD_outdoor")
                   for name in split[domain]["val"])
    require(len(train_names) == len(set(train_names)) == 136, "Expected frozen SDSD train136")
    require(len(names) == len(set(names)) == 16, "Expected frozen SDSD VAL16")
    require(not {scene_key(n) for n in train_names} & {scene_key(n) for n in names}, "Train/VAL scene overlap")
    metadata = {}
    for mode in MODES:
        base = runs / f"{mode}_s0"
        meta = metadata[mode] = read(base / "provenance.json")
        args = meta["args"]
        require(args["variant"] == mode and args["seed"] == 0 and args["domain"] == "sdsd"
                and args["trainable"] == "fusion" and args["steps"] == 600,
                f"{mode}: expected seed0 SDSD fusion-only 600-step run")
        require(meta["trainable_base"] == 0 and meta["trainable_fusion"] > 0,
                f"{mode}: base must be frozen and fusion trainable")
        require(meta["verified_frozen_base_sha256"] == meta["base_pixel_state_sha256"],
                f"{mode}: trained base state differs from initialization")
        require(meta["split_sha256"] == digest(splits), f"{mode}: split hash mismatch")
        plan = read(base / "data_plan.json")
        require(meta["data_plan_sha256"] == digest(base / "data_plan.json"), f"{mode}: actual plan hash mismatch")
        require(plan["sequence_names"] == train_names and plan["data_seed"] == args["data_seed"],
                f"{mode}: train order/data seed mismatch")
        require(plan["samples"] == make_plan(136, args["steps"] * args["accum"], args["data_seed"]),
                f"{mode}: explicit sample/augmentation plan does not reproduce")
        require(meta["first_batch_name"] == train_names[plan["samples"][0]["sequence_index"]],
                f"{mode}: first batch name differs from plan")
        require(set(meta["source_sha256"]) == set(SOURCES), f"{mode}: source coverage mismatch")
        require(meta["fusion_config"]["fusion_mode"] == mode, f"{mode}: guidance mode mismatch")
        for key in ("init_sha256", "base_pixel_state_sha256", "verified_frozen_base_sha256",
                    "fusion_initial_state_sha256", "split_sha256", "data_plan_sha256", "first_batch_sha256"):
            value = meta[key]
            require(isinstance(value, str) and len(value) == 64 and
                    all(c in "0123456789abcdef" for c in value), f"{mode}: invalid {key}")
    for key in ("init_sha256", "base_pixel_state_sha256", "verified_frozen_base_sha256",
                "fusion_initial_state_sha256", "split_sha256", "data_plan_sha256", "first_batch_sha256",
                "first_batch_name", "source_sha256", "base_reconstruction", "trainable_fusion", "torch_version"):
        require(metadata["local"][key] == metadata["prior"][key], f"Matched metadata differs: {key}")
    for key in ("trainable", "domain", "steps", "accum", "eval_every", "lr", "fusion_lr", "seed", "data_seed", "base_git"):
        require(metadata["local"]["args"][key] == metadata["prior"]["args"][key], f"Matched run setting differs: {key}")
    local_config = {k: v for k, v in metadata["local"]["fusion_config"].items() if k != "fusion_mode"}
    prior_config = {k: v for k, v in metadata["prior"]["fusion_config"].items() if k != "fusion_mode"}
    require(local_config == prior_config, "Fusion configurations differ beyond guidance mode")
    expected = {"arch": "v2", "branch": "small", "arm": "pretrained", "clip_len": 16,
                "xattn": "global", "kv": "per_tubelet", "layers": "single", "token_proj": "none", "taps": [5, 11, 17, 23]}
    require(all(metadata["local"]["base_reconstruction"].get(k) == v for k, v in expected.items()),
            "Unsupported initial pixel/backbone configuration")
    require(not (source_archive is not None and code_revision is not None), "Choose archive or Git revision, not both")
    if code_revision is not None:
        source_validation = verify_sources(metadata["local"]["source_sha256"], code_root, code_revision)
    else:
        source_validation = verify_archive(metadata["local"]["source_sha256"],
                                           source_archive or TASK / "source_pilot_v2.tgz")
    schedule = metadata["local"]["args"]
    require(0 < step <= 600 and schedule["eval_every"] > 0, "Requested step outside v2 budget")
    steps = sorted({0, step} | set(range(schedule["eval_every"], step + 1, schedule["eval_every"])))
    values, gt = {}, {}
    for current in steps:
        values[current] = {}
        for mode in MODES:
            report = read(runs / f"{mode}_s0/val_step_{current:06d}.json")
            require(report["split"] == "val" and report["variant"] == mode and
                    report["step"] == current and report["n"] == 16, f"{mode}/{current}: invalid report identity")
            require(sorted(row["name"] for row in report["per_sequence"]) == names,
                    f"{mode}/{current}: not the frozen VAL16 names")
            indexed = {row["name"]: row for row in report["per_sequence"]}
            arrays = {metric: np.array([indexed[name][metric] for name in names], dtype=np.float64)
                      for metric in (*METRICS, "tlp", "gt_tlp")}
            require(all(np.isfinite(v).all() for v in arrays.values()), f"{mode}/{current}: nonfinite metric")
            require(all(np.isclose(v.mean(), report["metrics"][k], rtol=0, atol=1e-10)
                        for k, v in arrays.items()), f"{mode}/{current}: mean differs from raw rows")
            require(np.allclose(arrays["tlp_gap"], np.abs(arrays["tlp"] - arrays["gt_tlp"]), rtol=0, atol=1e-12),
                    f"{mode}/{current}: temporal gap formula differs")
            values[current][mode] = {k: arrays[k] for k in METRICS}
            gt[(current, mode)] = arrays["gt_tlp"]
    initial_delta = {k: float(np.max(np.abs(values[0]["prior"][k] - values[0]["local"][k]))) for k in METRICS}
    require(initial_delta["psnr_y"] < .02 and initial_delta["lpips"] < .001, "Initialization metrics diverge")
    gt_delta = max(float(np.max(np.abs(v - gt[(0, "local")]))) for v in gt.values())
    require(gt_delta < 1e-5, "GT temporal reference changed")
    scenes = sorted({scene_key(name) for name in names})
    require(len(scenes) == 8, "Expected eight VAL scenes")
    groups = [np.array([i for i, name in enumerate(names) if scene_key(name) == scene]) for scene in scenes]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 16, (boot, 16))
    scene_draws = rng.integers(0, 8, (boot, 8))
    curves = {}
    for current in steps:
        absolute = {mode: {k: float(v.mean()) for k, v in values[current][mode].items()} for mode in MODES}
        gains = {mode: {k: paired_summary(values[current][mode][k] - values[0][mode][k], groups, draws, scene_draws, k)
                        for k in METRICS} for mode in MODES}
        contrast = {k: paired_summary(values[current]["prior"][k] - values[current]["local"][k],
                                      groups, draws, scene_draws, k) for k in METRICS}
        curves[str(current)] = {"absolute": absolute, "gain_vs_fixed_init": gains, "prior_minus_local": contrast}
    return {"stage": "Exploratory single-seed frozen-base fusion-only SDSD VAL screen; no test/SOTA claim",
            "training_arms": list(MODES), "baseline_training_arm": None,
            "gain_reference": "Each arm's fixed, zero-initialized fusion step0; no baseline continuation is fabricated.",
            "step": step, "validated_steps": steps, "n_sequences": 16, "n_scenes": 8, "training_seeds": [0],
            "bootstrap_samples": boot, "rng_seed": 0,
            "bootstrap_protocol": "RNG0 sequence draws then scene draws, reused across all metrics/steps. Whole-scene total difference divided by sequence count.",
            "interval_scope": "Unadjusted descriptive intervals conditional on this trained seed; no training-seed or selection correction.",
            "initial_metric_max_differences": initial_delta, "gt_temporal_max_difference": gt_delta,
            "verified_frozen_base_sha256": metadata["local"]["verified_frozen_base_sha256"],
            "provenance_scope": "Recorded checkpoint/first-batch/frozen-state hashes are compared; model/data bytes are not opened. Actual plans, split and source bytes are hashed.",
            "source_validation": source_validation, "matched_metadata": metadata,
            "curves": curves, **curves[str(step)], "sequence_names": names,
            "scene_members": {s: [names[i] for i in group] for s, group in zip(scenes, groups)},
            "sources": sources, "analyzer_sha256": digest(Path(__file__)),
            "shared_analysis_sha256": digest(Path(__file__).with_name("analyze_pilots.py")), "numpy_version": np.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", "--root", dest="runs", type=Path, default=TASK / "pilot_v2/artifacts/runs")
    parser.add_argument("--step", type=int, default=600)
    parser.add_argument("--splits", type=Path, default=CVPR / "results/splits_cvpr.json")
    parser.add_argument("--code-root", type=Path, default=CVPR / "code_full")
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument("--source-archive", type=Path)
    provenance.add_argument("--code-revision")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = analyze(args.runs, args.step, splits=args.splits, source_archive=args.source_archive,
                     code_root=args.code_root, code_revision=args.code_revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"step": result["step"], "absolute": result["absolute"],
                      "gain_vs_fixed_init": {m: {k: {q: x for q, x in v.items() if q != "per_sequence_differences"}
                                                 for k, v in values.items()} for m, values in result["gain_vs_fixed_init"].items()},
                      "prior_minus_local": {k: {q: x for q, x in v.items() if q != "per_sequence_differences"}
                                             for k, v in result["prior_minus_local"].items()}}, indent=2))


if __name__ == "__main__":
    main()
