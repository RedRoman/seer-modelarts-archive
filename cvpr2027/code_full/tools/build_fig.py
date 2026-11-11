"""Build the paper figure (single-column width, ~3.3in) for AAAI submission:
real V-JEPA cross-attention overlay on a real low-light frame from the
actual v35 (Seer) checkpoint.

Data provenance (see meta.json alongside):
  - Checkpoint: /cache/zihaohe/checkpoint/runs/v35_stage4/final.pt (step 6000)
  - Sequence: SDSD_indoor pair4 (input/GT .npy frames), frames 5-20 (16-frame
    clip), center-cropped to 256x256, middle frame (index 8, i.e. global
    frame #13) used for the still shown here.
  - Attention: encoder cross-attn at scale 4 (enc_xattn[4], the coarsest
    pixel-branch scale, feature map downsampled to 16x16, which is a 1:1
    match with the 16x16 V-JEPA spatial token grid). Captured by
    monkeypatching VJepaCrossAttnBF.forward to expose the post-softmax
    attention matrix (before it's contracted with V), which is otherwise
    discarded inside F.scaled_dot_product_attention.
  - This is head index 0 of 16 (selected because per-head inspection showed
    it has one of the sharper/most concentrated distributions at this scale;
    the 16-head average is much more diffuse -- documented honestly).
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
HEAD = 0
a = attn[frame_idx, HEAD]
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

queries = [("Dark", dark_rc, "#e74c3c"), ("Bright", bright_rc, "#17a2b8"),
           ("High-detail", struct_rc, "#f1c40f")]

# Figure sized for a ~3.3in-wide single column after LaTeX \includegraphics scaling.
fig = plt.figure(figsize=(3.35, 2.55))
gs = fig.add_gridspec(2, 3, wspace=0.10, hspace=0.30, left=0.03, right=0.97, top=0.90, bottom=0.03,
                       height_ratios=[1.15, 1.0])

ax0 = fig.add_subplot(gs[0, :])
ax0.imshow(lq_img_disp)
ax0.set_title("SDSD-indoor pair4, LQ frame (V-JEPA cross-attn queries marked)", fontsize=6.3)
ax0.set_xticks([]); ax0.set_yticks([])
for name, (r, c), color in queries:
    rect = plt.Rectangle((c*cellW, r*cellH), cellW, cellH, fill=False, edgecolor=color, linewidth=1.3)
    ax0.add_patch(rect)
    ax0.text(c*cellW + cellW/2, r*cellH - 6, name, color=color, fontsize=5.0,
             ha="center", va="bottom", fontweight="bold",
             path_effects=[])

im = None
for idx, (name, (r, c), color) in enumerate(queries):
    ax = fig.add_subplot(gs[1, idx])
    qidx = r * Wq + c
    dist = a[qidx].reshape(16, 16).numpy()
    heat_up = zoom(dist, (Himg/16, Wimg/16), order=1)
    heat_up = heat_up / heat_up.max()
    ax.imshow(lq_img_disp)
    im = ax.imshow(heat_up, cmap="inferno", alpha=0.62, vmin=0, vmax=1)
    rect = plt.Rectangle((c*cellW, r*cellH), cellW, cellH, fill=False, edgecolor=color, linewidth=1.1)
    ax.add_patch(rect)
    ax.set_title(name, fontsize=6.0, color=color, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

cbar_ax = fig.add_axes([0.30, 0.008, 0.42, 0.018])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("normalized cross-attn. weight", fontsize=5.3, labelpad=1)
cbar.ax.tick_params(labelsize=4.6, pad=1)

out_path = f"{save_dir}/fig_cross_attention.pdf"
plt.savefig(out_path, dpi=400)
plt.close()
print("saved:", out_path)
