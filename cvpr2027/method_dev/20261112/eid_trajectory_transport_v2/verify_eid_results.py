"""Independent NumPy/stdlib verifier for EID-TT v2 experimental artifacts.

Recomputes sequence means, multi-seed per-sequence averages, absolute tLP gaps,
per-seed directional sign agreement, exact baseline anchor parity against
baseline_replay_v1, and 10,000-draw scene-cluster bootstrap confidence intervals
(both vs anchor and vs fixed_temporal_mean_0.5 across all _mean3 variants)
from raw scalar JSON outputs alone. Requires zero PyTorch dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recompute_scene_bootstrap(
    rows: List[dict],
    method_key: str,
    ref_key: str,
    metric_keys: Tuple[str, ...],
    draws: int = 10000,
    seed: int = 20261112,
) -> Dict[str, dict]:
    scenes = sorted({r["scene"] for r in rows})
    by_scene = {s: [r for r in rows if r["scene"] == s] for s in scenes}
    n_scenes = len(scenes)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_scenes, size=(draws, n_scenes))
    result = {}
    for mk in metric_keys:
        deltas = np.array(
            [np.mean([r["methods"][method_key][mk] - r["methods"][ref_key][mk] for r in by_scene[s]]) for s in scenes],
            dtype=np.float64,
        )
        boot = deltas[idx].mean(axis=1)
        result[mk] = {
            "mean_delta": float(deltas.mean()),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
        }
    return result


def verify_target_report(path: Path, baseline_replay_dir: Path | None = None) -> dict:
    data = json.loads(path.read_text())
    rows = data["per_sequence"]
    _require(len(rows) == data["n_sequences"] and len(rows) > 0, f"Invalid sequence count in {path}")
    _require(data.get("all_states_unchanged") is True, f"State-unchanged check failed in {path}")

    metrics = ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")
    all_methods = list(rows[0]["methods"].keys())
    checks_passed = 0

    for r in rows:
        for m in all_methods:
            md = r["methods"][m]
            for k in metrics:
                _require(np.isfinite(md[k]), f"Nonfinite {k} for {m} on {r['name']}")
            if not m.endswith("_mean3"):
                gap = abs(md["tlp"] - md["gt_tlp"])
                _require(abs(gap - md["tlp_gap"]) < 1e-12, f"tLP gap arithmetic mismatch on {r['name']} {m}")
            checks_passed += 1

        # Check 3-seed mean arithmetic
        for m in all_methods:
            if m.endswith("_mean3"):
                prefix = m[:-6]
                seed_tags = [f"{prefix}_s{s}" for s in (0, 1, 2) if f"{prefix}_s{s}" in r["methods"]]
                if seed_tags:
                    for k in metrics:
                        recomp = float(np.mean([r["methods"][st][k] for st in seed_tags]))
                        _require(abs(recomp - r["methods"][m][k]) < 1e-12, f"3-seed mean mismatch on {m} {k}")
                        checks_passed += 1

    # Recompute overall means
    for m in all_methods:
        for k in metrics:
            recomp_mean = float(np.mean([r["methods"][m][k] for r in rows]))
            _require(abs(recomp_mean - data["means"][m][k]) < 1e-12, f"Aggregate mean mismatch on {m} {k}")
            checks_passed += 1

    # Recompute bootstrap CIs for ALL _mean3 variants vs anchor (and vs fixed_temporal_mean_0.5 if present)
    mean3_methods = [m for m in all_methods if m.endswith("_mean3")]
    for m3 in mean3_methods:
        recomp_boot = recompute_scene_bootstrap(rows, m3, "anchor", ("psnr_y", "psnr_rgb", "lpips", "tlp_gap"))
        stored_boot = data["scene_bootstrap_vs_anchor"][m3]
        for k, vals in recomp_boot.items():
            for bk in ("mean_delta", "ci95_low", "ci95_high"):
                _require(abs(vals[bk] - stored_boot[k][bk]) < 1e-11, f"Bootstrap mismatch {m3}/{k}/{bk} in {path}")
                checks_passed += 1

        if "scene_bootstrap_vs_fixed_temporal_mean" in data and m3 in data["scene_bootstrap_vs_fixed_temporal_mean"]:
            recomp_fix = recompute_scene_bootstrap(
                rows, m3, "fixed_temporal_mean_0.5", ("psnr_y", "psnr_rgb", "lpips", "tlp_gap")
            )
            stored_fix = data["scene_bootstrap_vs_fixed_temporal_mean"][m3]
            for k, vals in recomp_fix.items():
                for bk in ("mean_delta", "ci95_low", "ci95_high"):
                    _require(
                        abs(vals[bk] - stored_fix[k][bk]) < 1e-11,
                        f"Fixed-mean bootstrap mismatch {m3}/{k}/{bk} in {path}",
                    )
                    checks_passed += 1

    # Check exact parity with historical baseline_replay_v1 on BVI VAL16 if available
    anchor_parity = None
    if baseline_replay_dir is not None and data["domain"] == "bvi" and data["split"] == "val":
        bfile = baseline_replay_dir / f"{data['backbone']}_bvi_val16.json"
        if bfile.is_file():
            bdata = json.loads(bfile.read_text())
            bmap = {r["name"]: r for r in bdata["per_sequence"]}
            max_diff = 0.0
            for r in rows:
                ref_r = bmap[r["name"]]
                _require(r["lq_rgb_u8_sha256"] == bdata["inputs"][rows.index(r)]["lq_sha256"], "LQ SHA mismatch")
                _require(r["gt_rgb_u8_sha256"] == bdata["inputs"][rows.index(r)]["gt_sha256"], "GT SHA mismatch")
                _require(r["methods"]["anchor"]["gt_tlp"] == ref_r["gt_tlp"], "GT tLP mismatch")
                for k in ("psnr_y", "psnr_rgb", "lpips", "tlp_gap"):
                    diff = abs(r["methods"]["anchor"][k] - ref_r[k])
                    max_diff = max(max_diff, diff)
                    _require(diff < 1e-5, f"Anchor vs baseline_replay_v1 drift {diff} on {r['name']} {k}")
                    checks_passed += 1
            anchor_parity = {"verified_against": str(bfile), "max_metric_abs_diff": max_diff}

    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "domain": data["domain"],
        "split": data["split"],
        "backbone": data["backbone"],
        "n_sequences": data["n_sequences"],
        "n_scenes": data["n_scenes"],
        "checks_passed": checks_passed,
        "anchor_parity": anchor_parity,
        "anchor_means": data["means"]["anchor"],
        "eid_full_mean3": data["means"].get("eid_full_mean3"),
    }


def verify_source_internal17_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = data["internal17_per_sequence"]
    _require(len(rows) == 34 and data["internal17_n_clips"] == 34 and data["internal17_n_scenes"] == 17, "Invalid internal17 size")
    metrics = ("psnr_y", "psnr_rgb", "decoupled_psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")
    all_methods = list(rows[0]["methods"].keys())
    checks_passed = 0

    for r in rows:
        for m in all_methods:
            md = r["methods"][m]
            for k in metrics:
                _require(np.isfinite(md[k]), f"Nonfinite {k} on {r['name']} {m}")
            if not m.endswith("_mean3"):
                _require(abs(abs(md["tlp"] - md["gt_tlp"]) - md["tlp_gap"]) < 1e-12, "internal17 tLP gap mismatch")
            checks_passed += 1
        for m in all_methods:
            if m.endswith("_mean3"):
                prefix = m[:-6]
                seed_tags = [f"{prefix}_s{s}" for s in (0, 1, 2) if f"{prefix}_s{s}" in r["methods"]]
                for k in metrics:
                    recomp = float(np.mean([r["methods"][st][k] for st in seed_tags]))
                    _require(abs(recomp - r["methods"][m][k]) < 1e-12, f"internal17 3-seed mean mismatch {m}/{k}")
                    checks_passed += 1

    for m in all_methods:
        for k in metrics:
            recomp = float(np.mean([r["methods"][m][k] for r in rows]))
            _require(abs(recomp - data["internal17_means"][m][k]) < 1e-12, f"internal17 aggregate mismatch {m}/{k}")
            checks_passed += 1

    for m3 in [m for m in all_methods if m.endswith("_mean3")]:
        rb = recompute_scene_bootstrap(rows, m3, "anchor", ("psnr_y", "psnr_rgb", "decoupled_psnr_rgb", "lpips", "tlp_gap"))
        sb = data["internal17_bootstrap_vs_anchor"][m3]
        for k, vals in rb.items():
            for bk in ("mean_delta", "ci95_low", "ci95_high"):
                _require(abs(vals[bk] - sb[k][bk]) < 1e-11, f"internal17 bootstrap mismatch {m3}/{k}/{bk}")
                checks_passed += 1

    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "n_clips": 34,
        "n_scenes": 17,
        "checks_passed": checks_passed,
        "anchor_means": data["internal17_means"]["anchor"],
        "eid_full_mean3": data["internal17_means"].get("eid_full_mean3"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, default=None)
    parser.add_argument("--baseline-replay-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _require(not args.out.exists(), f"Verification output already exists: {args.out}")
    target_files = sorted(args.results_dir.glob("eval_*.json"))
    _require(len(target_files) > 0, f"No eval_*.json files found in {args.results_dir}")

    summaries = [verify_target_report(p, args.baseline_replay_dir) for p in target_files]
    source_ver = verify_source_internal17_summary(args.source_summary) if args.source_summary is not None else None
    total_checks = int(sum(s["checks_passed"] for s in summaries)) + (source_ver["checks_passed"] if source_ver else 0)

    receipt = {
        "schema_version": 2,
        "verified": True,
        "files_verified": len(summaries) + (1 if source_ver else 0),
        "total_scalar_and_hash_checks": total_checks,
        "source_internal17_verification": source_ver,
        "reports": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {"verified": True, "files": receipt["files_verified"], "checks": total_checks},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
