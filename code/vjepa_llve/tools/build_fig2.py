"""Build the paper figure (single-column width, ~3.3in) for AAAI submission:
real V-JEPA cross-attention overlay on a real low-light frame from the
actual v35 (Seer) checkpoint.

Uses the 16-head-averaged attention map at encoder cross-attn scale 4
(coarsest scale, 16x16 query grid, 1:1 with the 16x16 V-JEPA token grid) --
using the head average rather than a single cherry-picked head, since it is
the natural / unbiased summary of "where does this scale borrow information
from" without needing to justify a specific head choice.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from scipy.ndimage import zoom

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 6.3,
    "axes.titlesize": 6.3,
    "axes.labelsize": 6.0,
    "xtick.labelsize": 5.3,
    "ytick.labelsize": 5.3,
    "axes.linewidth": 0.5,
})

save_dir = "/home/ma-user/workspace/zihaohe/visualizations/fig_cross_attention"
meta = json.load(open(f"{save_dir}/meta.json"))

lq = np.load(f"{save_dir}/lq_clip.npy")
frame_idx = 8
lq_img = lq[0, :, frame_idx].transpose(1, 2, 0)
lq_img_disp = np.clip(lq_img * 3.2, 0, 1)

attn = torch.load(f"{save_dir}/attn_enc_s4.pt")  # [BT=16, h=16, HW_q=256, M=256]
a = attn[frame_idx].mean(dim=0)  # head-averaged: [HW_q=256, M=256]
Hq = Wq = 16

Himg, Wimg = lq_img.shape[:2]
cellH, cellW = Himg / Hq, Wimg / Wq
lq_gray = lq_img.mean(axis=-1)
means = np.zeros((Hq, Wq))
var_map = np.zeros((Hq, Wq))
for r in range(Hq):
    for c in range(Wq):
        cell = lq_gray[int(r*cellH):int((r+1)*cellH), int(c*cellW):int((c+1)*cellW)]
        means[r, c] = cell.mean()
        var_map[r, c] = cell.std()
dark_rc = np.unravel_index(np.argmin(means), means.shape)
bright_rc = np.unravel_index(np.argmax(means), means.shape)
struct_rc = np.unravel_index(np.argmax(var_map), var_map.shape)

queries = [("Dark", dark_rc, "#ff5b4d"), ("Bright", bright_rc, "#38d9ff"),
           ("High-detail", struct_rc, "#ffe14d")]

fig = plt.figure(figsize=(3.35, 2.65))
gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.42, left=0.02, right=0.98, top=0.86, bottom=0.05,
                       height_ratios=[1.2, 1.0])

ax0 = fig.add_subplot(gs[0, :])
ax0.imshow(lq_img_disp)
ax0.set_title("SDSD-indoor pair4 -- LQ frame w/ query patches", fontsize=6.5)
ax0.set_xticks([]); ax0.set_yticks([])
txt_offsets = {"Dark": (0, 1), "Bright": (0, -1), "High-detail": (0, 1)}
for name, (r, c), color in queries:
    rect = plt.Rectangle((c*cellW, r*cellH), cellW, cellH, fill=False, edgecolor=color, linewidth=1.5)
    ax0.add_patch(rect)
    dx, dy = txt_offsets[name]
    ypos = r*cellH - 5 if dy > 0 else r*cellH + cellH + 9
    ax0.text(c*cellW + cellW/2, ypos, name, color=color, fontsize=5.4,
             ha="center", va="bottom" if dy > 0 else "top", fontweight="bold",
             path_effects=[pe.withStroke(linewidth=1.4, foreground="black")])

im = None
for idx, (name, (r, c), color) in enumerate(queries):
    ax = fig.add_subplot(gs[1, idx])
    qidx = r * Wq + c
    dist = a[qidx].reshape(16, 16).numpy()
    heat_up = zoom(dist, (Himg/16, Wimg/16), order=1)
    heat_up = heat_up / heat_up.max()
    ax.imshow(lq_img_disp)
    im = ax.imshow(heat_up, cmap="inferno", alpha=0.65, vmin=0, vmax=1)
    rect = plt.Rectangle((c*cellW, r*cellH), cellW, cellH, fill=False, edgecolor=color, linewidth=1.1)
    ax.add_patch(rect)
    ax.set_title(name, fontsize=6.2, color=color, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

cbar_ax = fig.add_axes([0.30, 0.012, 0.42, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("cross-attn. weight (norm. per query)", fontsize=5.3, labelpad=1)
cbar.ax.tick_params(labelsize=4.6, pad=1)

out_path = f"{save_dir}/fig_cross_attention.pdf"
plt.savefig(out_path, dpi=400)
plt.close()
print("saved:", out_path)

# Print the actual max/uniform ratio for each query to report honestly
uniform = 1.0 / 256
for name, (r, c), color in queries:
    qidx = r * Wq + c
    dist = a[qidx].numpy()
    print(f"{name}: max={dist.max():.4f} ratio_to_uniform={dist.max()/uniform:.1f}x  argmax_vjepa_rc=({np.argmax(dist)//16},{np.argmax(dist)%16})")
