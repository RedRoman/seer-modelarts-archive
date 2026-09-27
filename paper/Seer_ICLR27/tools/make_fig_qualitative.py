"""Qualitative comparison figure (rows = clips, columns = methods) from dump_qualitative.py output.

Input : a local copy of the dump folder (qualitative.json + <clip>/<method>.png), e.g.
        results/qualitative/<run>/  (scp'd from seer_iclr_evals/<run>/ on the server).
Output: figures/fig_qualitative.pdf (5.5 in = ICLR text width, fonts >= 6.5 pt, pdf.fonttype 42).

Columns: LQ (gamma 0.4 for visibility), HVI-CIDNet, DarkIR, URCSA, STCD, Retinexformer,
         [Seer w/o V-JEPA], Seer, [Seer+CP-src], GT   (bracketed ones only if present in the dump).
Each tile: frame 8 of the 512x512 clip downsized to --tile_px; the 96x96 region ZOOM[clip]
(same location in every column) is outlined in red and shown 2x in a red-framed inset at the
bottom-left corner (the inset is the native-resolution crop, so it is not resampled).
Below each tile: PSNR (dB) / LPIPS of that frame vs GT, from qualitative.json.

  python3 tools/make_fig_qualitative.py --run results/qualitative/<run> [--out figures/fig_qualitative.pdf]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZSIZE = 96          # zoom region side, in 512-px clip coordinates
# top-left (x, y) of the 96x96 zoom region per clip, 512-px coordinates; chosen once per clip,
# kept outside the bottom-left 192x192 corner that the 2x inset covers.
ZOOM = {
    "pair60": (140, 120),             # chair headrest + window blinds (fine stripes, dark)
    "pair70": (208, 200),             # desk clutter: router LEDs, red box, cans
    "MVI_1030": (360, 160),           # building edge + hedge leaves (texture)
    "S35_Bunny__10fps": (152, 108),   # "HARRY POTTER" book title (text)
    "S37_Mario3__10fps": (300, 316),  # colour-checker patches + clip (chroma fidelity)
}
ROW_LABEL = {
    "L0_SDSD-indoor": "SDSD-in",
    "L1_SDSD-outdoor": "SDSD-out",
    "L2_BVI-RLV": "BVI-RLV",
}
CLIP_LABEL = {"S35_Bunny__10fps": "S35 Bunny", "S37_Mario3__10fps": "S37 Mario3"}
TITLE = {
    "LQ": "LQ (gamma-\nboosted)",
    "HVI-CIDNet": "HVI-\nCIDNet",
    "DarkIR": "DarkIR",
    "URCSA": "URCSA",
    "STCD": "STCD",
    "Retinexformer": "Retinex-\nformer",
    "Seer w/o V-JEPA": "Seer w/o\nV-JEPA",
    "Seer": "Seer",
    "Seer+CP-src": "Seer\n+CP-src",
    "GT": "GT",
}
BASE_COLS = ["LQ", "HVI-CIDNet", "DarkIR", "URCSA", "STCD", "Retinexformer"]
RED = "#E0201B"

FIG_W = 5.5          # in, ICLR \textwidth
LEFT = 0.25          # in, row-label strip
GAP = 0.025          # in, between tiles
HEAD = 0.27          # in, column-title band
LAB_H = 0.115        # in, metric label band below each tile
ROW_GAP = 0.035      # in, between rows
FS_TITLE = 7.0
FS_LABEL = 6.5
FS_ROW = 6.5
LW = 0.6             # pt, red box / inset frame
MIN_SEP = 0.008      # in, minimum horizontal gap between neighbouring labels


def load_tile(path, tile_px, zoom_xy, gamma=None):
    """512x512 PNG -> (tile_px, tile_px, 3) float image with the 2x inset pasted bottom-left.
    Returns image, inset side (px in tile coords), source box (x, y, s) in tile coords."""
    im = np.asarray(Image.open(path).convert("RGB"))
    H, W = im.shape[:2]
    f = im.astype(np.float64) / 255.0
    if gamma is not None:
        f = f ** gamma
    im = (f * 255 + 0.5).clip(0, 255).astype(np.uint8)
    x0, y0 = zoom_xy
    assert 0 <= x0 and x0 + ZSIZE <= W and 0 <= y0 and y0 + ZSIZE <= H, (path, zoom_xy)
    assert x0 >= 2 * ZSIZE or y0 + ZSIZE <= H - 2 * ZSIZE, \
        f"zoom box {zoom_xy} overlaps the bottom-left inset area of {path}"
    scale = tile_px / W
    tile = np.asarray(Image.fromarray(im).resize((tile_px, tile_px), Image.LANCZOS))
    ins = int(round(2 * ZSIZE * scale))                   # 2x zoom relative to the tile
    crop = Image.fromarray(im[y0:y0 + ZSIZE, x0:x0 + ZSIZE])
    crop = np.asarray(crop.resize((ins, ins), Image.LANCZOS if ins < ZSIZE else Image.NEAREST))
    tile = tile.copy()
    tile[tile_px - ins:, :ins] = crop
    return tile, ins, (x0 * scale, y0 * scale, ZSIZE * scale)


def fmt(r):
    if r is None or not np.isfinite(r.get("psnr_rgb", np.inf)):
        return ""
    return f"{r['psnr_rgb']:.2f}/{r['lpips']:.3f}".replace("/0.", "/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="local dump folder with qualitative.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "fig_qualitative.pdf"))
    ap.add_argument("--seer", default="Seer", help="label of the main Seer checkpoint in the dump")
    ap.add_argument("--ablation", default="Seer w/o V-JEPA", help="label of the w/o V-JEPA ckpt (if dumped)")
    ap.add_argument("--no_cp", action="store_true", help="drop the Seer+CP-src column")
    ap.add_argument("--clips", nargs="+", default=None, help="subset / order of clips (default: dump order)")
    ap.add_argument("--zoom", nargs="+", default=[], metavar="CLIP=X,Y", help="override zoom boxes")
    ap.add_argument("--tile_px", type=int, default=256)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    Q = json.load(open(os.path.join(args.run, "qualitative.json")))
    zoom = dict(ZOOM)
    for z in args.zoom:
        k, v = z.split("=")
        zoom[k] = tuple(int(a) for a in v.split(","))
    clips = args.clips or Q["clip_order"]
    have = set(Q["clips"][clips[0]]["methods"])
    cols = list(BASE_COLS)
    if args.ablation in have:
        cols.append(args.ablation)
    cols.append(args.seer)
    if not args.no_cp and f"{args.seer}+CP-src" in have:
        cols.append(f"{args.seer}+CP-src")
    cols.append("GT")
    for c in clips:
        miss = [m for m in cols if m not in Q["clips"][c]["methods"]]
        assert not miss, f"{c}: missing {miss} in dump"
    alias = {args.seer: "Seer", f"{args.seer}+CP-src": "Seer+CP-src", args.ablation: "Seer w/o V-JEPA"}
    titles = [TITLE.get(alias.get(m, m), m) for m in cols]

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "savefig.bbox": None,
    })
    n, R = len(cols), len(clips)
    t = (FIG_W - LEFT - (n - 1) * GAP) / n
    fig_h = HEAD + R * (t + LAB_H) + (R - 1) * ROW_GAP + 0.01
    fig = plt.figure(figsize=(FIG_W, fig_h))
    P = args.tile_px
    d = 0.5 * LW * P / (t * 72.0)       # half line width in tile pixels: keep inset frame inside the axes
    bands = {}                          # band -> [text artists left to right]  (overlap check)
    row_labels = []
    for i, c in enumerate(clips):
        rec = Q["clips"][c]
        y_top = HEAD + i * (t + LAB_H + ROW_GAP)
        for j, m in enumerate(cols):
            x = LEFT + j * (t + GAP)
            ax = fig.add_axes([x / FIG_W, 1 - (y_top + t) / fig_h, t / FIG_W, t / fig_h])
            r = rec["methods"][m]
            tile, ins, (bx, by, bs) = load_tile(os.path.join(args.run, r["png"]), P,
                                                zoom[c], gamma=0.4 if m == "LQ" else None)
            ax.imshow(tile, interpolation="none", extent=(0, P, P, 0))
            ax.add_patch(Rectangle((bx, by), bs, bs, fill=False, ec=RED, lw=LW))
            ax.add_patch(Rectangle((d, P - ins + d), ins - 2 * d, ins - 2 * d, fill=False, ec=RED, lw=LW))
            ax.set_xlim(0, P); ax.set_ylim(P, 0)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            lab = fmt(r) if m != "GT" else ""
            if lab:
                bands.setdefault(f"row{i}", []).append(
                    fig.text((x + t / 2) / FIG_W, 1 - (y_top + t + 0.012) / fig_h, lab,
                             ha="center", va="top", fontsize=FS_LABEL))
            if i == 0:
                bands.setdefault("titles", []).append(
                    fig.text((x + t / 2) / FIG_W, 1 - (HEAD - 0.025) / fig_h, titles[j],
                             ha="center", va="bottom", fontsize=FS_TITLE, linespacing=1.0,
                             fontweight="bold" if m == args.seer else "normal"))
        sp = rec["split"]
        rl = f"{ROW_LABEL.get(sp, sp)}\n{CLIP_LABEL.get(c, c)}"
        row_labels.append(fig.text(0.02 / FIG_W, 1 - (y_top + t / 2) / fig_h, rl, rotation=90,
                                   ha="left", va="center", fontsize=FS_ROW, linespacing=1.05,
                                   multialignment="center"))

    # ---- legibility / overlap check at the final physical size (inches) ----
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ext = lambda a: a.get_window_extent(rend).transformed(fig.dpi_scale_trans.inverted())  # noqa: E731
    bad = []
    for band, arts in bands.items():
        for a, b in zip(arts, arts[1:]):
            ea, eb = ext(a), ext(b)
            if eb.x0 - ea.x1 < MIN_SEP:
                bad.append(f"{band}: {a.get_text()!r} and {b.get_text()!r} {eb.x0 - ea.x1:+.3f} in apart")
    for a in row_labels:
        e = ext(a)
        if e.height > t + 0.02 or e.x1 > LEFT - 0.01:
            bad.append(f"row label {a.get_text()!r} does not fit ({e.width:.3f} x {e.height:.3f} in)")
    for a in [x for arts in bands.values() for x in arts] + row_labels:
        e = ext(a)
        if e.x0 < 0 or e.x1 > FIG_W or e.y0 < 0 or e.y1 > fig_h:
            bad.append(f"outside figure: {a.get_text()!r}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"[fig] {args.out}: {FIG_W:.2f} x {fig_h:.2f} in, {R} rows x {n} cols, tile {t:.3f} in; cols={cols}")
    print("[fig] zoom boxes:", {c: zoom[c] for c in clips})
    for b in bad:
        print("[fig][WARN]", b)


if __name__ == "__main__":
    main()
