#!/usr/bin/env python3
"""Reproduce the exploratory VAL16 mean x basis analysis; no accelerator needed.

The original global RNG stream and JSON schema are deliberately preserved.
Run from any directory: python /path/to/cvpr2027/tools/analyze_mean_basis_val.py --check
NumPy is the only non-standard dependency. This checks archived metadata and
local source bytes, not checkpoint contents or a new model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

CVPR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = CVPR / "results/protocol_audit_20261111/mean_basis_val"
DEFAULT_BASIS = CVPR / "results/cvpr_converged/probes/token_basis_train.json"
DEFAULT_SPLITS = CVPR / "results/splits_cvpr.json"
DEFAULT_MANIFEST = CVPR / "results/protocol_audit_20261111/checkpoint_manifest_31051.json"
CONDITION_LABELS = {
    "identity": {"mean": "unchanged", "basis": "identity", "rank": "full"},
    "clip_mean_only": {"mean": "clip", "basis": "none", "rank": 0},
    "clip_mean_clip_basis": {"mean": "clip", "basis": "clip_centered", "rank": 4},
    "clip_mean_training_basis": {"mean": "clip", "basis": "training", "rank": 4},
    "training_mean_clip_basis": {"mean": "training", "basis": "clip_centered", "rank": 4},
    "training_mean_training_basis": {"mean": "training", "basis": "training", "rank": 4},
}
CONDITIONS = tuple(CONDITION_LABELS)
CONTRASTS = [(c, "identity") for c in CONDITIONS if c != "identity"] + [
    ("clip_mean_clip_basis", "clip_mean_training_basis"),
    ("training_mean_clip_basis", "training_mean_training_basis"),
    ("clip_mean_clip_basis", "training_mean_clip_basis"),
    ("clip_mean_training_basis", "training_mean_training_basis"),
    ("clip_mean_clip_basis", "clip_mean_only"),
]
SOURCE_FILES = {
    "tools/mean_basis_control.py", "tools/eval_e4.py", "tools/checkpoint_config.py",
    "models/vjepa_backbone.py", "models/pixel_branch_v2.py", "models/pixel_branch_v8.py",
}
UPSTREAM_SUFFIX = "modelzoo/vjepa2-main/app/vjepa_2_1/models/vision_transformer.py"
UPSTREAM_SHA256 = "d2932eabeba684d8f558302a13cfd4be70a0170ee5112f5a794652d0a29089b9"
EXPECTED_RECONSTRUCTION = {
    "version": 2, "mode": "saved", "arch": "v2", "trained_arm": "pretrained",
    "arm": "pretrained", "clip_len": 16, "branch": "small", "xattn": "global",
    "kv": "per_tubelet", "layers": "single", "taps": [5, 11, 17, 23],
    "token_proj": "none", "token_proj_k": 4, "random_encoder_seed": None,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def hash_string(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            f"Invalid SHA256: {label}")


def validate_reports(reports, *, basis_path=DEFAULT_BASIS, splits_path=DEFAULT_SPLITS,
                     manifest_path=DEFAULT_MANIFEST, source_root=CVPR / "code_full",
                     upstream_source=None):
    """Fail closed on incomplete pairs, changed inputs, or configuration drift."""
    require(len(reports) == 2, "Exactly two reports (seed 0, seed 1) are required")
    basis, splits, manifest = map(read_json, (basis_path, splits_path, manifest_path))
    expected_names = sorted(f"{domain}/{n}" for domain in ("SDSD_indoor", "SDSD_outdoor")
                            for n in splits[domain]["val"])
    require(len(expected_names) == len(set(expected_names)) == 16, "Expected frozen SDSD VAL16")
    require(basis["split"] == "train" and basis["taps"] == [5, 11, 17, 23]
            and basis["rank"] == 64 and basis["clips"] == 60, "Training basis metadata mismatch")
    means, components = np.asarray(basis["mean"]), np.asarray(basis["components"])
    require(means.shape == (4, 1024) and components.shape == (4, 64, 1024), "Basis shape mismatch")
    require(np.isfinite(means).all() and np.isfinite(components).all(), "Nonfinite basis")
    manifest_hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    normalized_sources = []
    fingerprints, tokens = [], []
    for seed, report in enumerate(reports):
        prefix = f"seed {seed}: "
        required = {"seed": seed, "requested_split": "val", "split": "val", "heldout": "sdsd",
                    "arm": "pretrained", "iters": 20000, "n_sequences": 16, "n_frames": 16,
                    "crop": 256, "tile": 0,
                    "analysis": "post-hoc validation mechanism control; no test-set claim",
                    "selection": "all 16 frozen validation sequences; no tuning or selection",
                    "adaptive_basis": "CPU float32 right singular vectors of clip-centered token field, once per clip/tap",
                    "projection_formula": "mean + (Z - mean) @ V.T @ V; V stores basis vectors as rows"}
        for key, expected in required.items():
            require(report.get(key) == expected, prefix + f"Unexpected {key}")
        require(report["reconstruction"] == EXPECTED_RECONSTRUCTION, prefix + "Reconstruction mismatch")
        names = report["sequence_names"]
        require(len(names) == 16 and sorted(names) == expected_names, prefix + "Sequence names mismatch/duplicate")
        require(report["basis_sha256"] == sha256(basis_path), prefix + "Basis SHA256 mismatch")
        require(report["split_sha256"] == sha256(splits_path), prefix + "Split SHA256 mismatch")
        ckpt = f"runs_e17/e17_small_pretrained_s{seed}/final_cpu.pt"
        require(report["ckpt"].endswith("/" + ckpt), prefix + "Unexpected checkpoint path")
        require(report["checkpoint_sha256"] == manifest_hashes[ckpt], prefix + "Checkpoint manifest mismatch")
        sources, external = {}, []
        for path, digest in report["source_sha256"].items():
            hash_string(digest, path)
            if "/code_full/" in path:
                relative = path.split("/code_full/", 1)[1]
                require(relative in SOURCE_FILES and relative not in sources, prefix + "Unexpected/duplicate source")
                require(sha256(Path(source_root) / relative) == digest, prefix + f"Source changed: {relative}")
                sources[relative] = digest
            else:
                require(path.endswith("/" + UPSTREAM_SUFFIX) and digest == UPSTREAM_SHA256,
                        prefix + "Unexpected upstream source provenance")
                external.append(digest)
        require(set(sources) == SOURCE_FILES and external == [UPSTREAM_SHA256], prefix + "Missing source hashes")
        normalized_sources.append(sources)
        require(set(report["conditions"]) == set(CONDITIONS), prefix + "Expected exactly six conditions")
        token_map = {}
        for condition, labels in CONDITION_LABELS.items():
            result = report["conditions"][condition]
            require(result["mean_basis"] == labels, prefix + f"Wrong mean/basis labels: {condition}")
            rows = result["per_sequence"]
            require(len(rows) == 16 and sorted(row["name"] for row in rows) == expected_names,
                    prefix + f"Unpaired/duplicate condition rows: {condition}")
            token_map[condition] = {}
            for row in rows:
                require(row["kind"] == "video" and row["frames"] == 16, prefix + "Frame selection changed")
                hash_string(row["token_sha256"], row["name"])
                token_map[condition][row["name"]] = row["token_sha256"]
            for metric in ("psnr_rgb", "psnr_y", "lpips"):
                values = np.asarray([row[metric] for row in rows], dtype=float)
                require(np.isfinite(values).all(), prefix + f"Nonfinite {metric}")
                require(np.isclose(float(values.mean()), result["aggregate"][metric], rtol=0, atol=1e-12),
                        prefix + f"Aggregate disagrees with rows: {condition}/{metric}")
        fps = report["sequence_fingerprints"]
        require(len(fps) == 16 and sorted(f["name"] for f in fps) == expected_names,
                prefix + "Missing/duplicate sequence fingerprints")
        fingerprint_map = {}
        for f in fps:
            for key in ("clip_mean_sha256", "clip_basis_sha256"):
                hash_string(f[key], f["name"] + "/" + key)
            fingerprint_map[f["name"]] = (f["clip_mean_sha256"], f["clip_basis_sha256"])
        fingerprints.append(fingerprint_map)
        tokens.append(token_map)
    require(normalized_sources[0] == normalized_sources[1], "Source bytes differ across seeds")
    require(fingerprints[0] == fingerprints[1], "Clip means/adaptive bases differ across seeds")
    require(tokens[0] == tokens[1], "Condition token fingerprints differ across seeds")
    if upstream_source is not None:
        require(sha256(upstream_source) == UPSTREAM_SHA256, "Local upstream source SHA256 mismatch")
    return expected_names


def build_analysis(reports):
    """Exact original estimator and draw order; callers must validate inputs first."""
    names = sorted(reports[0]["sequence_names"])
    rows = [{c: {row["name"]: row for row in report["conditions"][c]["per_sequence"]}
             for c in CONDITIONS} for report in reports]
    clusters = {}
    for index, name in enumerate(names):
        clusters.setdefault(re.sub(r"_2$", "", name), []).append(index)
    require(len(clusters) == 8 and all(len(v) == 2 for v in clusters.values()), "Expected eight two-pass scenes")
    cluster_indices = list(clusters.values())
    out = {
        "scope": "Exploratory validation only, all 16 frozen sequences, seeds 0 and 1; unadjusted paired bootstrap intervals, no success or equivalence gate",
        "bootstrap_replicates": 10000,
        "bootstrap_rng_seed": 0,
        "cluster_sensitivity": "Resample scene clusters with replacement, retain every member sequence of each selected cluster; sequence-weighted statistic",
        "clusters": {k: [names[i] for i in v] for k, v in clusters.items()},
        "conditions": {}, "contrasts": {},
    }
    for condition in CONDITIONS:
        out["conditions"][condition] = {}
        for metric in ("psnr_y", "lpips"):
            per_seed = [float(np.mean([r[condition][n][metric] for n in names])) for r in rows]
            out["conditions"][condition][metric] = {"mean": float(np.mean(per_seed)), "per_seed": per_seed}
    # One GLOBAL RNG. Do not reset inside either loop or reorder contrasts/metrics.
    # For each metric, consume all sequence draws before all scene draws.
    rng = np.random.default_rng(0)
    for a, b in CONTRASTS:
        key = a + "_minus_" + b
        out["contrasts"][key] = {}
        for metric in ("psnr_y", "lpips"):
            per_seed = np.array([[r[a][n][metric] - r[b][n][metric] for n in names] for r in rows])
            diff = per_seed.mean(axis=0)
            sequence_boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
            cluster_boot = []
            for _ in range(10000):
                choices = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
                indices = np.concatenate([cluster_indices[i] for i in choices])
                cluster_boot.append(diff[indices].mean())
            out["contrasts"][key][metric] = {
                "mean_difference": float(diff.mean()),
                "per_seed_differences": per_seed.mean(axis=1).tolist(),
                "sequence_ci95": np.percentile(sequence_boot, [2.5, 97.5]).tolist(),
                "scene_cluster_ci95": np.percentile(cluster_boot, [2.5, 97.5]).tolist(),
                "same_sign_across_seeds": bool(np.all(per_seed.mean(axis=1) > 0) or np.all(per_seed.mean(axis=1) < 0)),
            }
    return out


def load_validated_analysis(input_dir=DEFAULT_INPUT, **validation_paths):
    reports = [read_json(Path(input_dir) / f"mean_basis_val_s{s}.json") for s in (0, 1)]
    validate_reports(reports, **validation_paths)
    return build_analysis(reports)


def serialize(analysis):
    return json.dumps(analysis, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, help="Default: INPUT_DIR/analysis.json")
    parser.add_argument("--basis", type=Path, default=DEFAULT_BASIS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=CVPR / "code_full")
    parser.add_argument("--upstream-source", type=Path, help="Optional local upstream VJEPA source for byte verification")
    parser.add_argument("--check", action="store_true", help="Recompute and require byte-identical existing JSON; write nothing")
    args = parser.parse_args()
    out = args.out or args.input_dir / "analysis.json"
    analysis = load_validated_analysis(args.input_dir, basis_path=args.basis, splits_path=args.splits,
                                     manifest_path=args.manifest, source_root=args.source_root,
                                     upstream_source=args.upstream_source)
    content = serialize(analysis)
    if args.check:
        require(out.read_text() == content, f"Analysis differs from recomputation: {out}")
    else:
        out.write_text(content)
    print(f"{'Verified' if args.check else 'Wrote'} {out}: SHA256 {sha256(out)}")
    print("Validated 16 paired sequences, 8 scenes, seeds 0/1, 6 conditions; local sources, basis and split hashes match.")
    print("Checkpoint hashes match archived manifest; checkpoint contents were not reloaded.")
    if args.upstream_source is None:
        print("Upstream VJEPA hash matches archived provenance only; its source is not vendored.")
    print("Exploratory validation; unadjusted intervals, no equivalence gate; rank4-vs-identity LPIPS seed signs differ.")


if __name__ == "__main__":
    main()
