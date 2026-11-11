"""Matched VAL-only pilot analysis; never opens test metrics or images.

Intervals are descriptive, unadjusted, and conditional on one training seed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

CVPR = Path(__file__).resolve().parents[2]
MODES = ("baseline", "local", "prior")
METRICS = ("psnr_y", "psnr_rgb", "lpips", "tlp_gap")
SOURCES = ("tools/train_method_pilot.py", "models/pixel_branch_v2.py",
           "models/vjepa_backbone.py", "models/pixel_branch_fusion.py",
           "models/temporal_detail_fusion.py", "data/common.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(manifest, code_root, code_revision=None):
    """Read current files or committed bytes without checking out/mutating Git."""
    if code_root is None:
        require(code_revision is None, "--code-revision requires --code-root")
        return {"mode": "not_checked", "requested_revision": None, "resolved_revision": None}
    code_root = Path(code_root).resolve()
    require(set(manifest) == set(SOURCES), "Unexpected source manifest paths")
    resolved, prefix = None, None
    if code_revision is not None:
        def git(*arguments):
            return subprocess.run(["git", "-C", str(code_root), *arguments],
                                  check=True, capture_output=True, timeout=30).stdout
        repo_root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        prefix = code_root.relative_to(repo_root).as_posix()
        resolved = git("rev-parse", "--verify", "--end-of-options",
                       str(code_revision) + "^{commit}").decode().strip()
        require(len(resolved) == 40 and all(c in "0123456789abcdef" for c in resolved),
                "Git did not resolve a full commit SHA")
    for relative, expected in manifest.items():
        raw = (git("show", f"{resolved}:{prefix}/{relative}") if resolved is not None
               else (code_root / relative).read_bytes())
        require(hashlib.sha256(raw).hexdigest() == expected,
                f"Published source hash mismatch: {relative} at {resolved or 'working tree'}")
    return {"mode": "git_revision" if resolved is not None else "working_tree",
            "requested_revision": code_revision, "resolved_revision": resolved,
            "repository_code_prefix": prefix}


def scene_key(name):
    return name[:-2] if name.endswith("_2") else name


def make_plan(n_sequences, n_samples, seed):
    # Same explicit CPU RNG schedule as the training runner.
    rng = np.random.default_rng(seed)
    plan = []
    while len(plan) < n_samples:
        for index in rng.permutation(n_sequences):
            if len(plan) == n_samples:
                break
            plan.append({"sequence_index": int(index),
                         "augmentation_seed": int(rng.integers(0, 2**63 - 1))})
    return plan


def paired_summary(d, groups, draws, scene_draws, metric):
    counts = np.array([len(g) for g in groups])
    sums = np.array([d[g].sum() for g in groups])
    cluster = sums[scene_draws].sum(1) / counts[scene_draws].sum(1)
    return {"difference": float(d.mean()),
            "sequence_ci95": np.percentile(d[draws].mean(1), [2.5, 97.5]).tolist(),
            "scene_ci95": np.percentile(cluster, [2.5, 97.5]).tolist(),
            "sequences_better": int(np.sum(d > 0 if metric.startswith("psnr") else d < 0)),
            "per_sequence_differences": d.tolist()}


def analyze(runs, step=400, *, splits=CVPR / "results/splits_cvpr.json",
            code_root=CVPR / "code_full", code_revision=None, boot=10000):
    source_files = []

    def read(path):
        source_files.append({"path": str(path), "sha256": digest(path)})
        return json.loads(path.read_text())

    split = read(splits)
    train_names = [f"{domain}/{name}" for domain in ("SDSD_indoor", "SDSD_outdoor")
                   for name in split[domain]["train"]]
    names = sorted(f"{domain}/{name}" for domain in ("SDSD_indoor", "SDSD_outdoor")
                   for name in split[domain]["val"])
    require(len(train_names) == len(set(train_names)) == 136, "Expected unique frozen train136")
    require(len(names) == len(set(names)) == 16, "Expected unique frozen VAL16")
    require(not {scene_key(n) for n in names} & {scene_key(n) for n in train_names},
            "Train/VAL scene overlap")
    metadata, plans = {}, {}
    for mode in MODES:
        base = runs / f"{mode}_s0"
        metadata[mode] = read(base / "provenance.json")
        plans[mode] = read(base / "data_plan.json")
        meta, plan = metadata[mode], plans[mode]
        require(meta["args"]["variant"] == mode and meta["args"]["seed"] == 0,
                f"{mode}: wrong variant/seed")
        require(meta["split_sha256"] == digest(splits), f"{mode}: split file hash mismatch")
        require(meta["data_plan_sha256"] == digest(base / "data_plan.json"),
                f"{mode}: data plan file hash mismatch")
        require(plan["sequence_names"] == train_names, f"{mode}: training sequence order differs")
        require(plan["data_seed"] == meta["args"]["data_seed"], f"{mode}: data seed differs")
        expected = make_plan(136, meta["args"]["steps"] * meta["args"]["accum"], plan["data_seed"])
        require(plan["samples"] == expected, f"{mode}: sample schedule does not reproduce")
        require(meta["first_batch_name"] == train_names[plan["samples"][0]["sequence_index"]],
                f"{mode}: first batch is not the first planned sequence")
        require(set(meta["source_sha256"]) == set(SOURCES), f"{mode}: source manifest coverage differs")
        require(meta["trainable_base"] > 0, f"{mode}: missing trainable base")
        for field in ("init_sha256", "base_pixel_state_sha256", "split_sha256",
                      "data_plan_sha256", "first_batch_sha256"):
            value = meta[field]
            require(isinstance(value, str) and len(value) == 64 and
                    all(c in "0123456789abcdef" for c in value), f"{mode}: invalid {field}")
    for field in ("init_sha256", "base_pixel_state_sha256", "split_sha256", "data_plan_sha256",
                  "first_batch_sha256", "first_batch_name", "source_sha256", "base_reconstruction",
                  "trainable_base", "torch_version"):
        require(all(metadata[mode][field] == metadata["baseline"][field] for mode in MODES),
                f"Matched provenance differs: {field}")
    for field in ("steps", "accum", "eval_every", "lr", "fusion_lr", "seed", "data_seed", "base_git"):
        require(all(metadata[mode]["args"][field] == metadata["baseline"]["args"][field]
                    for mode in MODES), f"Matched optimizer/data setting differs: {field}")
    require(metadata["baseline"]["fusion_config"] is None and metadata["baseline"]["trainable_fusion"] == 0,
            "Baseline unexpectedly has fusion")
    require(metadata["local"]["fusion_initial_state_sha256"] == metadata["prior"]["fusion_initial_state_sha256"]
            and metadata["local"]["fusion_initial_state_sha256"] is not None, "Fusion initial state mismatch")
    require(metadata["local"]["trainable_fusion"] == metadata["prior"]["trainable_fusion"] > 0,
            "Fusion parameter counts differ")
    configs = []
    for mode in ("local", "prior"):
        config = dict(metadata[mode]["fusion_config"])
        require(config.pop("fusion_mode") == mode, f"{mode}: fusion mode differs")
        configs.append(config)
    require(configs[0] == configs[1], "Fusion configurations differ beyond mode")
    config = metadata["baseline"]["base_reconstruction"]
    expected_config = {"arch": "v2", "branch": "small", "arm": "pretrained", "clip_len": 16,
                       "kv": "per_tubelet", "layers": "single", "xattn": "global",
                       "taps": [5, 11, 17, 23], "token_proj": "none"}
    require(all(config.get(k) == v for k, v in expected_config.items()), "Init is not the intended default E17 model")
    source_validation = verify_sources(metadata["baseline"]["source_sha256"], code_root, code_revision)
    schedule = metadata["baseline"]["args"]
    require(0 <= step <= schedule["steps"], "Requested step outside training budget")
    steps = sorted({0, step} | set(range(schedule["eval_every"], step + 1, schedule["eval_every"])))
    values, gt_temporal = {}, {}
    for current in steps:
        values[current] = {}
        for mode in MODES:
            report = read(runs / f"{mode}_s0/val_step_{current:06d}.json")
            require(report["split"] == "val" and report["n"] == 16 and
                    report["step"] == current and report["variant"] == mode,
                    f"{mode}/{current}: wrong split/count/step/variant")
            require(sorted(row["name"] for row in report["per_sequence"]) == names,
                    f"{mode}/{current}: sequence coverage differs from frozen VAL16")
            indexed = {row["name"]: row for row in report["per_sequence"]}
            array = {metric: np.array([indexed[name][metric] for name in names], dtype=np.float64)
                     for metric in (*METRICS, "tlp", "gt_tlp")}
            require(all(np.isfinite(v).all() for v in array.values()), f"{mode}/{current}: nonfinite metric")
            require(all(np.isclose(v.mean(), report["metrics"][k], rtol=0, atol=1e-10)
                        for k, v in array.items()), f"{mode}/{current}: summary mean does not match rows")
            require(np.allclose(array["tlp_gap"], np.abs(array["tlp"] - array["gt_tlp"]), rtol=0, atol=1e-12),
                    f"{mode}/{current}: wrong per-sequence temporal gap")
            values[current][mode] = {k: array[k] for k in METRICS}
            gt_temporal[(current, mode)] = array["gt_tlp"]
    initial = values[0]
    max_initial_difference = {mode: {metric: float(np.max(np.abs(initial[mode][metric] - initial["baseline"][metric])))
                                     for metric in METRICS} for mode in MODES}
    # Keep cross-host CANN rounding differences; never subtract them from the
    # primary final-step contrast. Thresholds were specified before results.
    for mode in MODES:
        require(max_initial_difference[mode]["psnr_y"] < .02 and
                max_initial_difference[mode]["lpips"] < .001, f"Initialization metrics diverge: {max_initial_difference}")
    gt_max_difference = max(float(np.max(np.abs(v - gt_temporal[(0, "baseline")])))
                            for v in gt_temporal.values())
    require(gt_max_difference < 1e-5, "GT temporal metrics changed across arms/steps")
    scenes = sorted({scene_key(name) for name in names})
    require(len(scenes) == 8, "Expected frozen VAL eight scenes")
    groups = [np.array([i for i, name in enumerate(names) if scene_key(name) == scene]) for scene in scenes]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 16, (boot, 16))
    scene_draws = rng.integers(0, len(groups), (boot, len(groups)))
    curves = {}
    for current in steps:
        current_values = values[current]
        contrasts, changes = {}, {}
        for a, b in (("local", "baseline"), ("prior", "baseline"), ("prior", "local")):
            contrasts[f"{a}_minus_{b}"] = {
                metric: paired_summary(current_values[a][metric] - current_values[b][metric],
                                       groups, draws, scene_draws, metric) for metric in METRICS}
        for mode in MODES:
            changes[mode] = {metric: paired_summary(current_values[mode][metric] - initial[mode][metric],
                                                    groups, draws, scene_draws, metric) for metric in METRICS}
        curves[str(current)] = {
            "absolute": {mode: {k: float(v.mean()) for k, v in current_values[mode].items()} for mode in MODES},
            "change_from_initial": {mode: {k: changes[mode][k]["difference"] for k in METRICS} for mode in MODES},
            "change_from_initial_intervals": changes, "contrasts": contrasts}
    return {"stage": "exploratory single-seed method-development screen; not test/SOTA evidence",
            "step": step, "validated_steps": steps, "n": 16, "scenes": len(groups), "training_seeds": [0],
            "bootstrap_samples": boot, "rng_seed": 0,
            "bootstrap_protocol": "One RNG0: sequence draws then scene draws; reuse for all metrics/steps/contrasts. Whole-scene total difference divided by sampled sequence count.",
            "interval_scope": "Unadjusted intervals conditional on one trained seed; do not estimate seed variation or correct validation-based selection.",
            "initial_metric_max_differences": max_initial_difference,
            "gt_temporal_max_difference": gt_max_difference,
            "provenance_checks": {"recorded_inputs_match": True, "actual_data_plan_hash_and_schedule_match": True,
                                  "published_source_hashes_match": code_root is not None,
                                  "input_content_scope": "Checkpoint/first-batch hashes compared as recorded; checkpoint/data image bytes are not opened."},
            "source_validation": source_validation,
            "matched_metadata": metadata, "curves": curves, **curves[str(step)],
            "sequence_names": names, "scene_members": {s: [names[i] for i in g] for s, g in zip(scenes, groups)},
            "sources": source_files, "analyzer_sha256": digest(Path(__file__)), "numpy_version": np.__version__}


def self_test():
    # Unequal cluster sizes distinguish sequence weighting from scene means.
    d = np.array([1., 3., 3., 3.])
    groups = [np.array([0]), np.array([1, 2, 3])]
    draws = np.array([[0, 1, 2, 3]])
    scene_draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    summary = paired_summary(d, groups, draws, scene_draws, "psnr_y")
    np.testing.assert_allclose(summary["difference"], 2.5)
    np.testing.assert_allclose(summary["scene_ci95"], np.percentile([1., 2.5, 2.5, 3.], [2.5, 97.5]))
    np.testing.assert_allclose(summary["sequence_ci95"], [2.5, 2.5])
    require(summary["sequences_better"] == 4, "PSNR direction")
    require(paired_summary(-d, groups, draws, scene_draws, "lpips")["sequences_better"] == 4, "LPIPS direction")
    require(scene_key("SDSD_indoor/pair1_2") != scene_key("SDSD_outdoor/pair1_2"), "Domain grouping")
    require(make_plan(3, 8, 0) == make_plan(3, 8, 0), "Plan repeatability")
    require([r["sequence_index"] for r in make_plan(3, 8, 0)[:3]] == [2, 0, 1], "Plan permutation")
    print("PASS: paired estimator, unequal scene weighting, metric directions, grouping, plan RNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", "--root", dest="runs", type=Path,
                        default=CVPR.parents[1] / "seer_method_dev_20261111/artifacts/runs")
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--splits", type=Path, default=CVPR / "results/splits_cvpr.json")
    parser.add_argument("--code-root", type=Path, default=CVPR / "code_full")
    parser.add_argument("--code-revision", help="Verify committed source bytes with git show, e.g. 8a9dcfa for pilot v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error("--out is required unless --self-test")
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = analyze(args.runs, args.step, splits=args.splits, code_root=args.code_root,
                     code_revision=args.code_revision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"step": result["step"], "absolute": result["absolute"],
                      "change_from_initial": result["change_from_initial"],
                      "contrasts": {name: {metric: {k: v for k, v in data.items() if k != "per_sequence_differences"}
                                             for metric, data in comparison.items()}
                                    for name, comparison in result["contrasts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
