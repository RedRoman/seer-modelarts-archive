"""tools/reaggregate_ar2b.py — AR2.B re-aggregation per AS3 spec (2026-05-02).

Bucket sequences by motion_hq (GT frame diff, the "true" motion proxy)
NOT motion_lq (which conflates real motion with low-light flicker/noise).

Use existing CSVs:
  /cache/zihaohe/checkpoint/audit_llvestcd/per_sequence_metrics.csv  (v9.0_best, v9.0b_mix_best, v9.1a_best)
  /cache/zihaohe/checkpoint/motion_audit_v3/per_seq_motion_audit.csv  (v9.0-noJEPA_best, motion scores)

For each ckpt × bucket: report raw RGB-PSNR + lift-over-LQ.
Also: per-seq regression of lift ~ {motion_hq, motion_lq, LQ_baseline}.

Output: /cache/zihaohe/checkpoint/ar2b_reagg/
  ar2b_summary.md   (full per-bucket table + regression + sensitivity vs motion_lq)
  ar2b_per_seq.csv  (combined per-seq with all ckpts + motion + bucket assignments)
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def maybe_float(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except Exception:
        return None


def bucket_by_score(seq_score_pairs, n_buckets=3, labels=("static", "mild", "dynamic")):
    """Equal-size buckets by score (lowest → static)."""
    sorted_pairs = sorted(seq_score_pairs, key=lambda x: x[1])
    n = len(sorted_pairs)
    bucket = {}
    for i, (name, _) in enumerate(sorted_pairs):
        idx = min(i * n_buckets // n, n_buckets - 1)
        bucket[name] = labels[idx]
    return bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_csv", default="/cache/zihaohe/checkpoint/audit_llvestcd/per_sequence_metrics.csv")
    ap.add_argument("--motion_csv", default="/cache/zihaohe/checkpoint/motion_audit_v3/per_seq_motion_audit.csv")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/ar2b_reagg")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    audit = load_csv(args.audit_csv)
    motion = load_csv(args.motion_csv)

    # Build seq → {ckpt: psnr, motion_lq, motion_hq, LQ_psnr}
    seq_data = {}

    # First, process motion_csv (has all our seqs + motion scores + LQ + v9.0_best + noJEPA)
    for r in motion:
        if r["dataset"] != "indoor":
            continue
        seq = r["seq"]
        seq_data[seq] = {
            "motion_lq": maybe_float(r["motion_lq"]),
            "motion_hq": maybe_float(r["motion_hq"]),
            "LQ": maybe_float(r["LQ_rgb_psnr"]),
            "v9.0_best": maybe_float(r["v9.0_best_rgb_psnr"]),
            "v9.0-noJEPA_best": maybe_float(r["v9.0-noJEPA_best_rgb_psnr"]),
        }

    # Then merge in audit_csv (has v9.0b_mix and v9.1a)
    for r in audit:
        seq = r["seq"]
        if seq not in seq_data:
            continue  # skip non-overlapping
        seq_data[seq]["v9.0b_mix_best"] = maybe_float(r.get("v9.0b_mix_best_rgb_psnr_seqmean"))
        seq_data[seq]["v9.1a_best"] = maybe_float(r.get("v9.1a_best_rgb_psnr_seqmean"))

    # Compute lifts over LQ
    ckpts = ["v9.0_best", "v9.0b_mix_best", "v9.1a_best", "v9.0-noJEPA_best"]
    for seq, d in seq_data.items():
        lq = d["LQ"]
        for c in ckpts:
            if d.get(c) is not None and lq is not None:
                d[f"lift_{c}"] = d[c] - lq

    # Bucket by motion_hq (primary) and motion_lq (sensitivity)
    seq_motion_hq = [(s, d["motion_hq"]) for s, d in seq_data.items() if d["motion_hq"] is not None]
    seq_motion_lq = [(s, d["motion_lq"]) for s, d in seq_data.items() if d["motion_lq"] is not None]
    bucket_hq = bucket_by_score(seq_motion_hq, n_buckets=3)
    bucket_lq = bucket_by_score(seq_motion_lq, n_buckets=3)

    # Aggregate per-bucket per-ckpt
    def per_bucket_agg(bucket_map):
        agg = defaultdict(lambda: defaultdict(list))
        for seq, b in bucket_map.items():
            d = seq_data[seq]
            agg[b]["LQ"].append(d["LQ"])
            for c in ckpts:
                if d.get(c) is not None:
                    agg[b][c].append(d[c])
                if d.get(f"lift_{c}") is not None:
                    agg[b][f"lift_{c}"].append(d[f"lift_{c}"])
        out = {}
        for b in ("static", "mild", "dynamic"):
            n_seqs = len(agg[b]["LQ"])
            out[b] = {"n": n_seqs}
            for k, vals in agg[b].items():
                if vals:
                    out[b][k] = float(np.mean(vals))
        return out

    agg_hq = per_bucket_agg(bucket_hq)
    agg_lq = per_bucket_agg(bucket_lq)

    # Regression: lift ~ motion_hq, motion_lq, LQ
    def correlate(xs, ys):
        if len(xs) < 3:
            return None
        x = np.array(xs); y = np.array(ys)
        if x.std() < 1e-10 or y.std() < 1e-10:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    correl = {}
    valid_seqs = [s for s in seq_data if all(seq_data[s].get(k) is not None
                                              for k in ("motion_hq", "motion_lq", "LQ"))]
    for c in ckpts:
        valid_c = [s for s in valid_seqs if seq_data[s].get(f"lift_{c}") is not None]
        if not valid_c:
            continue
        lifts = [seq_data[s][f"lift_{c}"] for s in valid_c]
        m_hq = [seq_data[s]["motion_hq"] for s in valid_c]
        m_lq = [seq_data[s]["motion_lq"] for s in valid_c]
        lq_psnr = [seq_data[s]["LQ"] for s in valid_c]
        correl[c] = {
            "n": len(valid_c),
            "r_lift_vs_motion_hq": correlate(m_hq, lifts),
            "r_lift_vs_motion_lq": correlate(m_lq, lifts),
            "r_lift_vs_LQ_psnr": correlate(lq_psnr, lifts),
        }

    # === Write per-seq combined CSV ===
    csv_out = os.path.join(args.out_dir, "ar2b_per_seq.csv")
    keys = ["seq", "motion_hq", "motion_lq", "LQ", "bucket_by_hq", "bucket_by_lq",
            "v9.0_best", "v9.0b_mix_best", "v9.1a_best", "v9.0-noJEPA_best",
            "lift_v9.0_best", "lift_v9.0b_mix_best", "lift_v9.1a_best", "lift_v9.0-noJEPA_best"]
    with open(csv_out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for seq in sorted(seq_data, key=lambda s: seq_data[s].get("motion_hq", 0)):
            row = {"seq": seq, "bucket_by_hq": bucket_hq.get(seq), "bucket_by_lq": bucket_lq.get(seq)}
            for k in ("motion_hq", "motion_lq", "LQ", "v9.0_best", "v9.0b_mix_best",
                      "v9.1a_best", "v9.0-noJEPA_best",
                      "lift_v9.0_best", "lift_v9.0b_mix_best", "lift_v9.1a_best",
                      "lift_v9.0-noJEPA_best"):
                row[k] = seq_data[seq].get(k)
            w.writerow(row)
    print(f"wrote {csv_out}")

    # === Write summary MD ===
    md_out = os.path.join(args.out_dir, "ar2b_summary.md")
    def fmt(v):
        return f"{v:.3f}" if v is not None else "—"

    with open(md_out, "w") as f:
        f.write("# AR2.B Re-Aggregation — AS3 Spec\n\n")
        f.write("Per-bucket means, full-frame tiled inference, RGB-PSNR uint8.\n\n")

        # Bucket assignments
        f.write("## Bucket assignments\n\n")
        f.write("| seq | motion_hq | motion_lq | bucket_by_hq | bucket_by_lq | LQ |\n")
        f.write("|---|---:|---:|:---:|:---:|---:|\n")
        for seq in sorted(seq_data, key=lambda s: seq_data[s].get("motion_hq", 0)):
            d = seq_data[seq]
            f.write(f"| {seq} | {fmt(d['motion_hq'])} | {fmt(d['motion_lq'])} | "
                    f"{bucket_hq.get(seq, '—')} | {bucket_lq.get(seq, '—')} | {fmt(d['LQ'])} |\n")
        f.write("\n")

        # Primary table: bucket by motion_hq
        f.write("## Primary: bucket by motion_hq (GT frame diff — true scene motion)\n\n")
        f.write("| Bucket | n | LQ | v9.0 | v9.0b_mix | v9.1a | noJEPA |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for b in ("static", "mild", "dynamic"):
            a = agg_hq[b]
            f.write(f"| {b} | {a['n']} | {fmt(a.get('LQ'))} | "
                    f"{fmt(a.get('v9.0_best'))} | {fmt(a.get('v9.0b_mix_best'))} | "
                    f"{fmt(a.get('v9.1a_best'))} | {fmt(a.get('v9.0-noJEPA_best'))} |\n")
        f.write("\n### Lift over LQ (bucket by motion_hq)\n\n")
        f.write("| Bucket | n | lift_v9.0 | lift_v9.0b_mix | lift_v9.1a | lift_noJEPA |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for b in ("static", "mild", "dynamic"):
            a = agg_hq[b]
            f.write(f"| {b} | {a['n']} | {fmt(a.get('lift_v9.0_best'))} | "
                    f"{fmt(a.get('lift_v9.0b_mix_best'))} | {fmt(a.get('lift_v9.1a_best'))} | "
                    f"{fmt(a.get('lift_v9.0-noJEPA_best'))} |\n")

        # Sensitivity: bucket by motion_lq
        f.write("\n## Sensitivity: bucket by motion_lq (LQ frame diff — confounds motion + noise)\n\n")
        f.write("| Bucket | n | lift_v9.0 | lift_v9.0b_mix | lift_v9.1a | lift_noJEPA |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for b in ("static", "mild", "dynamic"):
            a = agg_lq[b]
            f.write(f"| {b} | {a['n']} | {fmt(a.get('lift_v9.0_best'))} | "
                    f"{fmt(a.get('lift_v9.0b_mix_best'))} | {fmt(a.get('lift_v9.1a_best'))} | "
                    f"{fmt(a.get('lift_v9.0-noJEPA_best'))} |\n")

        # Correlations
        f.write("\n## Per-ckpt correlation: lift ~ {motion_hq, motion_lq, LQ baseline PSNR}\n\n")
        f.write("Pearson r over n seqs. Negative r = lift decreases with that variable.\n\n")
        f.write("| Model | n | r(lift, motion_hq) | r(lift, motion_lq) | r(lift, LQ) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for c in ckpts:
            cc = correl.get(c, {})
            f.write(f"| {c} | {cc.get('n', '—')} | "
                    f"{fmt(cc.get('r_lift_vs_motion_hq'))} | "
                    f"{fmt(cc.get('r_lift_vs_motion_lq'))} | "
                    f"{fmt(cc.get('r_lift_vs_LQ_psnr'))} |\n")

        # Verdict
        f.write("\n## Verdict (AS3 interpretation table)\n\n")
        # Compare lift trajectory across buckets for each model
        v90_lifts = [agg_hq[b].get('lift_v9.0_best', 0) for b in ("static", "mild", "dynamic")]
        nojepa_lifts = [agg_hq[b].get('lift_v9.0-noJEPA_best', 0) for b in ("static", "mild", "dynamic")]
        all_fall = v90_lifts[2] < v90_lifts[0] - 1.0 and nojepa_lifts[2] < nojepa_lifts[0] - 1.0
        v90_falls_more = (v90_lifts[0] - v90_lifts[2]) > (nojepa_lifts[0] - nojepa_lifts[2]) + 1.0
        v90_falls_less = (nojepa_lifts[0] - nojepa_lifts[2]) > (v90_lifts[0] - v90_lifts[2]) + 1.0
        bucket_disagrees = bucket_by_score(seq_motion_lq) != bucket_by_score(seq_motion_hq)

        f.write(f"- v9.0 lift trajectory across (static, mild, dynamic by motion_hq): "
                f"{v90_lifts[0]:.2f} → {v90_lifts[1]:.2f} → {v90_lifts[2]:.2f}\n")
        f.write(f"- noJEPA lift trajectory:                                            "
                f"{nojepa_lifts[0]:.2f} → {nojepa_lifts[1]:.2f} → {nojepa_lifts[2]:.2f}\n")
        f.write(f"- Bucket assignments disagree (motion_lq vs motion_hq): {bucket_disagrees}\n\n")

        if v90_falls_more:
            verdict = "V-JEPA conditioning may CONFLICT with motion (v9.0 lift falls MORE sharply)"
        elif v90_falls_less:
            verdict = "V-JEPA conditioning may HELP motion robustness (v9.0 lift falls LESS sharply)"
        elif all_fall:
            verdict = "ALL models lift LESS on motion → motion is intrinsically hard, V-JEPA-specific motion claim UNRESOLVED"
        else:
            verdict = "INCONCLUSIVE / NOISE"
        f.write(f"**Verdict:** {verdict}\n")

    print(f"wrote {md_out}")


if __name__ == "__main__":
    main()
