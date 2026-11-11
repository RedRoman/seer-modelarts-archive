"""Build per-seq correlation table (BF Check 3) merging:
  v9.4 (full-frame tiled) per_seq.csv
  v9.4b (center-crop) per_seq.csv
  motion_audit_v3 per_seq_motion_audit.csv (LQ, motion_hq, bucket, noJEPA)
Output: /cache/zihaohe/checkpoint/prior_ablation_v9_4b/per_seq_correlation.{csv,md}
"""
import csv
import os

V94 = "/cache/zihaohe/checkpoint/prior_ablation_v9_4/per_seq.csv"
V94B = "/cache/zihaohe/checkpoint/prior_ablation_v9_4b/per_seq.csv"
MOTION = "/cache/zihaohe/checkpoint/motion_audit_v3/per_seq_motion_audit.csv"
OUT_DIR = "/cache/zihaohe/checkpoint/prior_ablation_v9_4b"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    v94 = {r["seq"]: r for r in load_csv(V94)}
    v94b = {r["seq"]: r for r in load_csv(V94B)}
    motion = {r["seq"]: r for r in load_csv(MOTION) if r.get("dataset") == "indoor"}

    seqs = sorted(set(v94) & set(v94b) & set(motion))

    rows = []
    for s in seqs:
        a = v94[s]; b = v94b[s]; m = motion[s]
        rows.append({
            "seq": s,
            "bucket": m.get("bucket", "?"),
            "motion_hq": float(m["motion_hq"]),
            "LQ": float(m["LQ_rgb_psnr"]),
            "noJEPA": float(m["v9.0-noJEPA_best_rgb_psnr"]),
            # tiled (v9.4) values
            "v9.0_tiled_full": float(a["full_vjepa"]),
            "v9.0_tiled_off": float(a["cross_attn_off"]),
            "delta_off_tiled": float(a["delta_cross_attn_off"]),
            # center-crop (v9.4b) values
            "v9.0_center_full": float(b["full_vjepa"]),
            "v9.0_center_off": float(b["cross_attn_off"]),
            "delta_off_center": float(b["delta_cross_attn_off"]),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "per_seq_correlation.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    md_path = os.path.join(OUT_DIR, "per_seq_correlation.md")
    with open(md_path, "w") as f:
        f.write("# BF Check 3 — Per-seq correlation table\n\n")
        f.write("Columns: bucket (motion_hq-based) / motion_hq / LQ / noJEPA-trained /\n")
        f.write("v9.0 tiled full vs cross_attn_off (Δ) / v9.0 center-crop full vs cross_attn_off (Δ)\n\n")
        f.write("| seq | bucket | motion_hq | LQ | noJEPA | tiled_full | tiled_off | Δ_tiled | center_full | center_off | Δ_center |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['seq']} | {r['bucket']} | {r['motion_hq']:.4f} | {r['LQ']:.2f} | "
                    f"{r['noJEPA']:.2f} | "
                    f"{r['v9.0_tiled_full']:.2f} | {r['v9.0_tiled_off']:.2f} | "
                    f"{r['delta_off_tiled']:+.2f} | "
                    f"{r['v9.0_center_full']:.2f} | {r['v9.0_center_off']:.2f} | "
                    f"{r['delta_off_center']:+.2f} |\n")

        # Aggregates
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        f.write(f"\n**Aggregates (n={n}):**\n\n")
        f.write(f"```\n")
        f.write(f"LQ avg                 : {avg('LQ'):.3f}\n")
        f.write(f"noJEPA avg             : {avg('noJEPA'):.3f}\n")
        f.write(f"v9.0 tiled full        : {avg('v9.0_tiled_full'):.3f}\n")
        f.write(f"v9.0 tiled off         : {avg('v9.0_tiled_off'):.3f}  (Δ {avg('delta_off_tiled'):+.3f})\n")
        f.write(f"v9.0 center-crop full  : {avg('v9.0_center_full'):.3f}\n")
        f.write(f"v9.0 center-crop off   : {avg('v9.0_center_off'):.3f}  (Δ {avg('delta_off_center'):+.3f})\n")
        f.write(f"\n")
        # Pearson-like correlation Δ_tiled vs Δ_center
        import statistics as st
        x = [r["delta_off_tiled"] for r in rows]
        y = [r["delta_off_center"] for r in rows]
        mx, my = st.mean(x), st.mean(y)
        num = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
        den = (sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))**0.5
        corr = num / den if den > 0 else 0
        f.write(f"Pearson corr(Δ_tiled, Δ_center): {corr:+.3f}\n")
        # Δ vs motion_hq
        m = [r["motion_hq"] for r in rows]
        for label, dvals in [("Δ_tiled", x), ("Δ_center", y)]:
            mm = st.mean(m); md = st.mean(dvals)
            num2 = sum((mi - mm)*(di - md) for mi, di in zip(m, dvals))
            den2 = (sum((mi - mm)**2 for mi in m) * sum((di - md)**2 for di in dvals))**0.5
            c2 = num2 / den2 if den2 > 0 else 0
            f.write(f"Pearson corr(motion_hq, {label}): {c2:+.3f}\n")
        # Δ vs LQ
        lq = [r["LQ"] for r in rows]
        for label, dvals in [("Δ_tiled", x), ("Δ_center", y)]:
            ml = st.mean(lq); md = st.mean(dvals)
            num2 = sum((li - ml)*(di - md) for li, di in zip(lq, dvals))
            den2 = (sum((li - ml)**2 for li in lq) * sum((di - md)**2 for di in dvals))**0.5
            c2 = num2 / den2 if den2 > 0 else 0
            f.write(f"Pearson corr(LQ, {label}): {c2:+.3f}\n")
        # Δ vs (full - noJEPA) i.e. how much V-JEPA helps in noJEPA gap
        gap = [r["v9.0_tiled_full"] - r["noJEPA"] for r in rows]
        for label, dvals in [("Δ_tiled", x), ("Δ_center", y)]:
            mg = st.mean(gap); md = st.mean(dvals)
            num2 = sum((gi - mg)*(di - md) for gi, di in zip(gap, dvals))
            den2 = (sum((gi - mg)**2 for gi in gap) * sum((di - md)**2 for di in dvals))**0.5
            c2 = num2 / den2 if den2 > 0 else 0
            f.write(f"Pearson corr(noJEPA-gap (full_tiled - noJEPA), {label}): {c2:+.3f}\n")
        f.write(f"```\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
