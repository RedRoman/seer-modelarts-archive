"""Figure: single-frame illumination invariance of six frozen encoders.

Data: results/aaai_data/invariance_v2.json (15 SDSD-indoor pairs; rho_patch =
||phi(LQ) - phi(GT)||_2 / ||phi(GT)||_2 per patch, averaged per sequence).
Stars: paired one-sided Wilcoxon of V-JEPA < encoder (or encoder < V-JEPA when
the encoder is lower), computed here from the per-sequence values.
Output: figures/fig_invariance.pdf
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from _style import PALETTE, apply_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "aaai_data" / "invariance_v2.json"
OUT = ROOT / "figures" / "fig_invariance.pdf"

ENC = [("siglip", "SigLIP 2"), ("clip", "CLIP"), ("dinov2", "DINOv2"),
       ("dinov3", "DINOv3"), ("vjepa", "V-JEPA 2.1"), ("videomae", "VideoMAE")]
COLOR = {"siglip": PALETTE["baseline"], "clip": PALETTE["clip"], "dinov2": PALETTE["dino"],
         "dinov3": PALETTE["seer_cp"], "vjepa": PALETTE["vjepa"], "videomae": PALETTE["highlight"]}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def main():
    apply_style()
    d = json.load(open(DATA))
    seqs = sorted(d["per_seq"])
    val = {k: np.array([d["per_seq"][s][k] for s in seqs]) for k, _ in ENC}
    order = sorted(ENC, key=lambda e: -val[e[0]].mean())
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    x = np.arange(len(order))
    for i in range(len(seqs)):
        ax.plot(x, [val[k][i] for k, _ in order], color=PALETTE["reference"], lw=0.4, alpha=0.35, zorder=1)
    for j, (k, name) in enumerate(order):
        v = val[k]
        lo, hi = np.percentile(np.random.default_rng(0).choice(v, (10000, len(v))).mean(1), [2.5, 97.5])
        ax.errorbar(j, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", ms=4.5,
                    color=COLOR[k], mec=PALETTE["ink"], mew=0.5, capsize=2, zorder=3)
        if k != "vjepa":
            alt = "less" if val["vjepa"].mean() < v.mean() else "greater"
            p = stats.wilcoxon(val["vjepa"], v, alternative=alt).pvalue
            ax.text(j, hi + 0.012, stars(p), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in order], rotation=18, ha="right")
    ax.set_ylabel(r"per-patch shift $\bar\rho_{\mathrm{patch}}$ ($\downarrow$)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT)
    print("wrote", OUT)
    for k, name in order:
        print(f"{name:10s} {val[k].mean():.3f}")


if __name__ == "__main__":
    main()
