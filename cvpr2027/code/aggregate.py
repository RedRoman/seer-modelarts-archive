"""Turn per-sequence JSONs into the pre-registered decisions.

Ground rule 4 of notes/CVPR_PREREG.md: arm differences are paired over sequences
on seed-averaged outputs, with a 10k-sample bootstrap 95% CI, and an effect
counts only if the CI excludes 0 *and* every seed shows the same sign. Both
halves are reported here, so a result that passes one and fails the other is
visible rather than averaged away.

  python tools/aggregate.py --dir results/e4_eval_val --compare pretrained none
  python tools/aggregate.py --dir results/e1_eval_val --key arm \
      --compare pretrained none --group heldout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HIGHER_BETTER = {"psnr_rgb", "psnr_y", "psnr_y_gtmean", "input_psnr_y"}
TEMPORAL = ("tlp", "mabd", "flicker", "warp_err")


def load(d):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        j = json.load(open(f))
        j["_file"] = os.path.basename(f)
        runs.append(j)
    return runs


def seed_average(runs):
    """{(group, arm): {sequence: {metric: value averaged over seeds}}}"""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seeds = defaultdict(set)
    for r in runs:
        g = r.get("heldout") or r.get("domain") or "all"
        arm = r.get("arm") or r.get("method")
        seeds[(g, arm)].add(r.get("seed"))
        for row in r["per_sequence"]:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "frames":
                    acc[(g, arm)][row["name"]][k].append(v)
    out = {}
    for key, seqs in acc.items():
        out[key] = {n: {k: float(np.mean(v)) for k, v in m.items()}
                    for n, m in seqs.items()}
    return out, {k: sorted(x for x in v if x is not None) for k, v in seeds.items()}


def per_seed_sign(runs, group, a, b, metric):
    """Sign of (a - b) within each seed, for the all-seeds-agree half of the rule."""
    by_seed = defaultdict(dict)
    for r in runs:
        g = r.get("heldout") or r.get("domain") or "all"
        if g != group:
            continue
        arm = r.get("arm") or r.get("method")
        if arm in (a, b):
            by_seed[r.get("seed")][arm] = {x["name"]: x for x in r["per_sequence"]}
    signs = []
    for s, arms in sorted(by_seed.items(), key=lambda kv: (kv[0] is None, kv[0])):
        if a not in arms or b not in arms:
            continue
        names = [n for n in arms[a] if n in arms[b] and metric in arms[a][n]]
        if not names:
            continue
        dm = np.mean([arms[a][n][metric] - arms[b][n][metric] for n in names])
        signs.append(np.sign(dm))
    return signs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), required=True,
                    help="report A minus B")
    ap.add_argument("--metrics", nargs="*",
                    default=["psnr_y", "lpips", "psnr_y_gtmean"])
    ap.add_argument("--temporal", action="store_true",
                    help="also report the temporal metrics as |pred - gt| gaps")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = load(args.dir)
    if not runs:
        raise SystemExit(f"no json under {args.dir}")
    avg, seeds = seed_average(runs)
    a, b = args.compare
    rng = np.random.default_rng(args.seed)
    groups = sorted({g for (g, arm) in avg})

    metrics = list(args.metrics)
    if args.temporal:
        metrics += [f"d_{k}" for k in TEMPORAL]

    print(f"{args.dir}   {a} minus {b}   "
          f"(bootstrap {args.boot}, paired over sequences, seed-averaged)")
    for g in groups:
        if (g, a) not in avg or (g, b) not in avg:
            print(f"\n[{g}] missing arm ({a} or {b}) -- skipped")
            continue
        sa, sb = avg[(g, a)], avg[(g, b)]
        names = sorted(set(sa) & set(sb))
        print(f"\n[{g}]  n={len(names)}  seeds {a}={seeds[(g, a)]} {b}={seeds[(g, b)]}")
        for met in metrics:
            if met.startswith("d_"):
                base = met[2:]
                def val(d, n):      # distance to the ground truth's own level
                    if base not in d[n] or f"gt_{base}" not in d[n]:
                        return None
                    return abs(d[n][base] - d[n][f"gt_{base}"])
                higher_better = False
            else:
                def val(d, n, m=met):
                    return d[n].get(m)
                higher_better = met in HIGHER_BETTER
            va = np.array([val(sa, n) for n in names if val(sa, n) is not None])
            vb = np.array([val(sb, n) for n in names if val(sb, n) is not None])
            if len(va) != len(vb) or len(va) == 0:
                continue
            diff = va - vb
            bs = np.array([rng.choice(diff, len(diff), replace=True).mean()
                           for _ in range(args.boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            better = (lo > 0) if higher_better else (hi < 0)
            worse = (hi < 0) if higher_better else (lo > 0)
            signs = per_seed_sign(runs, g, a, b, met[2:] if met.startswith("d_") else met)
            agree = len(set(signs)) <= 1 and len(signs) > 0
            verdict = ("better" if better else "worse" if worse else "n.s.")
            if verdict != "n.s." and not agree:
                verdict += " (seeds disagree)"
            print(f"  {met:<16} {a}={va.mean():8.4f}  {b}={vb.mean():8.4f}  "
                  f"d={diff.mean():+8.4f} [{lo:+.4f},{hi:+.4f}]  "
                  f"seeds={len(signs)}  {verdict}")


if __name__ == "__main__":
    main()
