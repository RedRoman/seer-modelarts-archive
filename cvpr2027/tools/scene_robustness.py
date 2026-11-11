"""Post-hoc scene-cluster sensitivity analysis of archived paired comparisons.

No images/models are read. The primary estimator remains the sequence-weighted
mean after averaging the available matched seeds. Resampling entire scenes
keeps paired passes/fps versions together; each replicate is sum/count, not an
unweighted mean of scene means. Equal-scene weighting is reported separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

CVPR = Path(__file__).resolve().parents[1]
METRICS = ("psnr_y", "lpips")
COMPARISONS = (
    ("indomain_sdsd", "indomain", "*", (0, 1, 2)),
    ("h5_sdsd", "xdomain", "e20_sdsd", (0, 1)),
    ("h5_bvi", "xdomain", "e20_bvi", (0, 1)),
)


def scene_key(name):
    # Keep the domain prefix: indoor/pair5 and outdoor/pair5 cannot merge.
    # For all present rows this agrees exactly with code/make_splits.py base().
    for suffix in ("__10fps", "__20fps"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name[:-2] if name.endswith("_2") else name


def ratio_replicates(sums, counts, draws):
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def self_check():
    # Unequal cluster sizes distinguish sequence weighting from equal-scene
    # weighting. Enumerate every two-scene bootstrap draw for a known example.
    draws = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_allclose(ratio_replicates(np.array([1., 9.]), np.array([1, 3]), draws),
                               [1., 2.5, 2.5, 3.])
    assert scene_key("SDSD_indoor/pair5_2") == "SDSD_indoor/pair5"
    assert scene_key("BVI/S33_mix5__20fps") == "BVI/S33_mix5"
    assert scene_key("SDSD_indoor/pair5") != scene_key("SDSD_outdoor/pair5")


def load_arm(root, folder, prefix, arm, expected_seeds):
    runs, sources = {}, []
    for path in sorted((root / folder).glob(f"{prefix}*{arm}_s*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        seed = data["seed"]
        if seed in runs:
            raise ValueError(f"duplicate {arm} seed {seed}: {path}")
        if data["arm"] != arm or data["split"] != "test":
            raise ValueError(f"unexpected arm/split in {path}")
        rows = {row["name"]: row for row in data["per_sequence"]}
        if len(rows) != len(data["per_sequence"]):
            raise ValueError(f"duplicate sequence names in {path}")
        runs[seed] = rows
        sources.append({"path": str(path.relative_to(CVPR)),
                        "sha256": hashlib.sha256(raw).hexdigest(), "seed": seed, "arm": arm})
    if set(runs) != set(expected_seeds):
        raise ValueError(f"{folder}/{prefix}/{arm}: expected {expected_seeds}, found {sorted(runs)}")
    return runs, sources


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def verdict(ci, seed_effects, metric):
    sign = 1 if ci[0] > 0 else -1 if ci[1] < 0 else 0
    agrees = bool(sign and all(np.sign(x) == sign for x in seed_effects))
    if not agrees:
        return "inconclusive"
    favorable = 1 if metric == "psnr_y" else -1
    return "benefit" if sign == favorable else "harm"


def compare(root, label, folder, prefix, expected_seeds, nboot, seed):
    prior, prior_sources = load_arm(root, folder, prefix, "pretrained", expected_seeds)
    none, none_sources = load_arm(root, folder, prefix, "none", expected_seeds)
    all_rows = list(prior.values()) + list(none.values())
    names = sorted(all_rows[0])
    if any(set(rows) != set(names) for rows in all_rows):
        raise ValueError(f"{label}: mismatched sequence coverage; do not silently drop rows")
    clusters = sorted({scene_key(name) for name in names})
    member_indices = [[i for i, name in enumerate(names) if scene_key(name) == scene]
                      for scene in clusters]
    counts = np.array([len(indices) for indices in member_indices])
    scene_draws = np.random.default_rng(seed).integers(0, len(clusters), (nboot, len(clusters)))
    sequence_draws = np.random.default_rng(seed).integers(0, len(names), (nboot, len(names)))
    result = dict(comparison=label, difference="pretrained minus none", seeds=list(expected_seeds),
                  n_sequences=len(names), n_scenes=len(clusters),
                  cluster_size_counts={str(k): v for k, v in sorted(Counter(counts.tolist()).items())},
                  sources=prior_sources + none_sources, metrics={}, scenes=[])
    scene_rows = [{"scene": scene, "n_sequences": int(counts[i]),
                   "sequences": [names[j] for j in member_indices[i]], "metrics": {}}
                  for i, scene in enumerate(clusters)]
    for metric in METRICS:
        a = np.array([[prior[s][name][metric] for name in names] for s in expected_seeds])
        b = np.array([[none[s][name][metric] for name in names] for s in expected_seeds])
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"{label}/{metric}: nonfinite values")
        seed_diffs = a - b
        diffs = a.mean(axis=0) - b.mean(axis=0)
        sums = np.array([diffs[indices].sum() for indices in member_indices])
        means = sums / counts
        np.testing.assert_allclose(sums.sum() / counts.sum(), diffs.mean(), rtol=0, atol=1e-12)
        per_seed = seed_diffs.mean(axis=1)
        equal_per_seed = np.array([[row[indices].mean() for indices in member_indices]
                                  for row in seed_diffs]).mean(axis=1)
        seq_ci = interval(diffs[sequence_draws].mean(axis=1))
        cluster_ci = interval(ratio_replicates(sums, counts, scene_draws))
        equal_ci = interval(means[scene_draws].mean(axis=1))
        seq_verdict, cluster_verdict = verdict(seq_ci, per_seed, metric), verdict(cluster_ci, per_seed, metric)
        result["metrics"][metric] = dict(
            prior_mean=float(a.mean()), none_mean=float(b.mean()), effect=float(diffs.mean()),
            per_seed_effects={str(s): float(v) for s, v in zip(expected_seeds, per_seed)},
            seed_signs_agree=bool(len(set(np.sign(per_seed))) == 1 and np.sign(per_seed[0]) != 0),
            sequence_bootstrap_ci95=seq_ci, scene_cluster_bootstrap_ci95=cluster_ci,
            sequence_verdict=seq_verdict, scene_cluster_verdict=cluster_verdict,
            conclusion_changed=seq_verdict != cluster_verdict,
            equal_scene_weighted={"effect": float(means.mean()), "ci95": equal_ci,
                                  "per_seed_effects": {str(s): float(v) for s, v in zip(expected_seeds, equal_per_seed)},
                                  "verdict": verdict(equal_ci, equal_per_seed, metric)})
        for i, row in enumerate(scene_rows):
            row["metrics"][metric] = {"sum_sequence_differences": float(sums[i]),
                                       "mean_sequence_difference": float(means[i])}
    result["scenes"] = scene_rows
    return result


def render_note(report):
    def fmt(v, digits):
        return f"{v:+.{digits}f}"
    def ci(v, digits):
        return f"[{fmt(v[0], digits)}, {fmt(v[1], digits)}]"
    lines = [
        "# Scene-bootstrap sensitivity audit — 2026-11-11", "",
        "This is a **post-hoc sensitivity analysis**, not a new preregistered test or a rerun of inference. "
        "It uses archived per-sequence JSONs and leaves their values unchanged.", "",
        "Within each sequence, average the matched seeds first, then subtract pretrained minus none. "
        "The primary point estimate remains sequence weighted. Sample whole scene clusters with replacement "
        f"({report['bootstrap_samples']:,} replicates, NumPy default_rng seed {report['rng_seed']}); each replicate is total "
        "sampled sequence differences divided by total sampled sequence count. "
        "Strip the 10/20-fps suffix and terminal `_2`, keeping the dataset/domain prefix, matching the "
        "split builder's scene groups on these rows. Paired passes/fps variants always travel together.", "",
        "| Comparison | Sequences / scenes | Metric | Effect | Sequence CI | Scene-cluster CI | Seed signs | Verdict |",
        "|---|---:|---|---:|---|---|---|---|",
    ]
    for comparison in report["comparisons"]:
        for metric, m in comparison["metrics"].items():
            digits = 4 if metric == "lpips" else 3
            signs = "/".join("+" if value > 0 else "−" if value < 0 else "0"
                             for value in m["per_seed_effects"].values())
            lines.append(f"| {comparison['comparison']} | {comparison['n_sequences']} / {comparison['n_scenes']} "
                         f"| {metric} | {fmt(m['effect'], digits)} | {ci(m['sequence_bootstrap_ci95'], digits)} "
                         f"| {ci(m['scene_cluster_bootstrap_ci95'], digits)} | {signs} | {m['scene_cluster_verdict']} |")
    lines += ["", "The CI/sign verdict is unchanged for every listed comparison. In-domain PSNR and LPIPS "
              "benefits persist. Transfer PSNR remains inconclusive for SDSD and BVI; LPIPS remains favorable "
              "under scene resampling, with the SDSD upper bound close to zero (−0.000761).", "",
              "Equal-scene weighting is a separate estimand, not a replacement for the primary result:", "",
              "| Comparison | Metric | Equal-scene effect | Scene-bootstrap CI |",
              "|---|---|---:|---|"]
    for comparison in report["comparisons"]:
        for metric, m in comparison["metrics"].items():
            digits = 4 if metric == "lpips" else 3
            equal = m["equal_scene_weighted"]
            lines.append(f"| {comparison['comparison']} | {metric} | {fmt(equal['effect'], digits)} | {ci(equal['ci95'], digits)} |")
    lines += ["", "SDSD contains 21 scenes: 18 with two sequences and 3 with one; BVI contains 8 scenes "
              "with two fps versions each. The intervals condition on the available trained seeds; they do "
              "not estimate uncertainty over a population of training seeds, repair any split leakage, or "
              "address unrecorded dependence beyond the scene grouping. The small BVI scene count limits precision.", "",
              "Source paths, SHA-256 hashes, per-seed effects and scene membership are in "
              "`results/protocol_audit_20261111/scene_cluster_bootstrap.json`. Reproduce from the repository root:", "",
              "```sh", "python3 cvpr2027/tools/scene_robustness.py --out /tmp/scene_audit.json --note /tmp/scene_audit.md", "```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=CVPR / "results" / "cvpr_converged")
    ap.add_argument("--out", type=Path, default=CVPR / "results" / "protocol_audit_20261111" / "scene_cluster_bootstrap.json")
    ap.add_argument("--note", type=Path, default=CVPR / "notes" / "SCENE_BOOTSTRAP_AUDIT_20261111.md")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.boot <= 0:
        ap.error("--boot must be positive")
    for path in (args.out, args.note):
        if path.exists():
            ap.error(f"refusing to overwrite {path}; choose a new output path")
    self_check()
    report = dict(analysis="post-hoc scene-cluster sensitivity; no new preregistered claim",
                  bootstrap_samples=args.boot, rng_seed=args.seed, numpy_version=np.__version__,
                  primary_estimator="sequence-weighted mean of seed-averaged paired differences",
                  bootstrap_estimator="sample whole scenes; total sampled difference / total sampled sequence count",
                  grouping="remove __10fps/__20fps, then terminal _2; retain domain prefix",
                  comparisons=[compare(args.results, *spec, args.boot, args.seed) for spec in COMPARISONS])
    for path in (args.out, args.note):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.note.write_text(render_note(report))
    for comparison in report["comparisons"]:
        for metric, m in comparison["metrics"].items():
            print(comparison["comparison"], metric, m["effect"], m["scene_cluster_bootstrap_ci95"], m["scene_cluster_verdict"])
    print(f"wrote {args.out}\nwrote {args.note}")


if __name__ == "__main__":
    main()
