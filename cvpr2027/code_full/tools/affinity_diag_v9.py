"""tools/affinity_diag_v9.py — V-JEPA token affinity diagnostic per AE2 spec.

Goal: determine whether V-JEPA 2.1 tokens carry useful non-warp temporal/structural
affinity signal on SDSD low-light clips. AE4: diagnostic only, no training.

For each (clip × adjacent_pair × modality):
  - Encode 16-frame clip through frozen V-JEPA backbone, take deepest layer.
  - Token grid (T'=8, H'=16, W'=16, D=1024); each token plane covers 2 input frames.
  - For adjacent token planes (p, p+1), compute (256,256) cosine affinity.
  - For each token in plane p, find argmax NN in plane p+1 → displacement.
  - Save heatmap + NN overlay PNG.

Modalities: LQ, pred(v9.0_best), HQ.

Aggregate metrics:
  mean top-1 confidence       (higher = more localized neighborhoods)
  median displacement (px)    (translated back to image-space pixels)
  displacement std            (lower = more stable)
  closer-to-HQ score          (cos(A_pred, A_HQ) vs cos(A_LQ, A_HQ))

Verdict logic in summary.md:
  USEFUL     if pred top-1 conf > 0.5 AND closer-to-HQ ≥ +0.05 vs LQ AND median displacement consistent
  AMBIGUOUS  if some criteria pass and others don't
  NOT USEFUL if affinity is diffuse (top-1 < 0.3) or pred no closer to HQ than LQ
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.common import read_image_01
from data.sdsd_dataset import SDSDDataset
from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import load_config


def list_seqs(root: str, split: str):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def load_clip_center(lq_paths, gt_paths, n_frames: int = 16, crop_size: int = 256):
    """Load center 16 frames at center 256x256 crop."""
    n = min(n_frames, len(lq_paths))
    start = (len(lq_paths) - n) // 2
    lq = np.stack([read_image_01(lq_paths[start + i]) for i in range(n)], axis=0)
    gt = np.stack([read_image_01(gt_paths[start + i]) for i in range(n)], axis=0)
    lq_t = torch.from_numpy(lq).permute(3, 0, 1, 2).unsqueeze(0).float()
    gt_t = torch.from_numpy(gt).permute(3, 0, 1, 2).unsqueeze(0).float()
    H, W = lq_t.shape[-2:]
    if H >= crop_size and W >= crop_size:
        h0 = (H - crop_size) // 2
        w0 = (W - crop_size) // 2
        lq_t = lq_t[..., h0:h0 + crop_size, w0:w0 + crop_size]
        gt_t = gt_t[..., h0:h0 + crop_size, w0:w0 + crop_size]
    return lq_t, gt_t


@torch.no_grad()
def encode_tokens(backbone, video_01: torch.Tensor, layer_idx: int = -1) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Returns (tokens (B, N, D), token_grid (T', H', W'))."""
    out = backbone(video_01)
    feats = out["feats"]
    tg = out["token_grid"]
    return feats[layer_idx], tg


def cosine_affinity(toks_a: torch.Tensor, toks_b: torch.Tensor) -> torch.Tensor:
    """toks_a: (Na, D), toks_b: (Nb, D). Returns (Na, Nb) cosine sim."""
    a = F.normalize(toks_a.float(), dim=-1)
    b = F.normalize(toks_b.float(), dim=-1)
    return a @ b.T


def nn_stats(A: torch.Tensor, grid_h: int, grid_w: int) -> Dict:
    """A: (N, N) cosine affinity. N = grid_h * grid_w. Per-token NN stats."""
    N = A.shape[0]
    top1 = A.max(dim=-1).values.cpu().numpy()      # (N,) top-1 cosine
    nn_idx = A.argmax(dim=-1).cpu().numpy()        # (N,) flat index in grid_b

    yy, xx = np.divmod(np.arange(N), grid_w)       # i in plane a → (yi, xi)
    ny, nx = np.divmod(nn_idx, grid_w)             # NN in plane b → (yj, xj)
    dx = (nx - xx).astype(np.float32)
    dy = (ny - yy).astype(np.float32)
    disp = np.sqrt(dx**2 + dy**2)

    # entropy of top-5 (higher entropy = less confident)
    top5 = torch.topk(A, k=5, dim=-1).values.cpu().numpy()
    top5 = np.maximum(top5, 1e-6)
    top5 = top5 / top5.sum(axis=1, keepdims=True)
    ent = -(top5 * np.log(top5)).sum(axis=1)

    return {
        "mean_top1": float(top1.mean()),
        "median_disp_tokens": float(np.median(disp)),
        "std_disp_tokens": float(disp.std()),
        "mean_top5_entropy": float(ent.mean()),
        "dx_per_token": dx, "dy_per_token": dy,
        "top1_per_token": top1,
        "nn_idx_per_token": nn_idx,
    }


def closer_to_hq_score(A_pred: torch.Tensor, A_LQ: torch.Tensor, A_HQ: torch.Tensor) -> float:
    """Flatten and compare cosine similarity to HQ affinity. Higher = pred closer to HQ."""
    p, l, h = (A_pred.flatten().float(), A_LQ.flatten().float(), A_HQ.flatten().float())
    p, l, h = (F.normalize(x, dim=0) for x in (p, l, h))
    return float((p @ h).item() - (l @ h).item())


def save_affinity_heatmap(A: torch.Tensor, out_path: str, label: str = ""):
    """A: (N, N) → png heatmap."""
    a = A.cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    img = (a * 255).astype(np.uint8)
    img_rgb = np.stack([img, img, img], axis=-1)
    Image.fromarray(img_rgb).save(out_path)


def save_nn_overlay(frame_uint8: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                    grid_h: int, grid_w: int, out_path: str,
                    arrow_step: int = 2, color: Tuple[int, int, int] = (255, 60, 60)):
    """Draw NN displacement arrows on frame.

    frame_uint8: (H, W, 3) uint8 image (e.g., the LQ or pred frame at 256x256).
    dx, dy: (grid_h*grid_w,) displacements in token coords.
    """
    img = Image.fromarray(frame_uint8.copy())
    draw = ImageDraw.Draw(img)
    H, W = frame_uint8.shape[:2]
    cell_h = H // grid_h
    cell_w = W // grid_w
    for gy in range(0, grid_h, arrow_step):
        for gx in range(0, grid_w, arrow_step):
            i = gy * grid_w + gx
            x0 = int(gx * cell_w + cell_w / 2)
            y0 = int(gy * cell_h + cell_h / 2)
            x1 = int(x0 + dx[i] * cell_w)
            y1 = int(y0 + dy[i] * cell_h)
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            draw.line([(x0, y0), (x1, y1)], fill=color, width=1)
            draw.ellipse([(x0 - 1, y0 - 1), (x0 + 1, y0 + 1)], fill=color)
    img.save(out_path)


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """t: (3, H, W) in [0,1] → (H, W, 3) uint8."""
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


@torch.no_grad()
def diag_one_pair(modality_name: str, video_01: torch.Tensor,
                  backbone, plane_a: int, plane_b: int,
                  out_dir: str, frame_t_idx: int):
    """Encode video, extract token planes (plane_a, plane_b), compute affinity + NN.

    video_01: (1, 3, T=16, H, W) in [0,1].
    Returns dict with stats and the affinity matrix.
    """
    tokens, tg = encode_tokens(backbone, video_01, layer_idx=-1)  # (1, N, D)
    Tp, Hp, Wp = tg
    D = tokens.shape[-1]
    N_per_plane = Hp * Wp
    tokens = tokens[0].view(Tp, N_per_plane, D)  # (Tp, N_per_plane, D)
    toks_a = tokens[plane_a]  # (256, 1024)
    toks_b = tokens[plane_b]
    A = cosine_affinity(toks_a, toks_b)  # (256, 256)

    stats = nn_stats(A, Hp, Wp)

    # Save heatmap
    save_affinity_heatmap(A, os.path.join(out_dir, f"affinity_{modality_name}.png"),
                          label=modality_name)

    # NN overlay on the actual frame at index frame_t_idx
    frame_uint8 = to_uint8_image(video_01[0, :, frame_t_idx])
    save_nn_overlay(frame_uint8, stats["dx_per_token"], stats["dy_per_token"],
                    Hp, Wp, os.path.join(out_dir, f"nn_overlay_{modality_name}.png"))

    return {"affinity": A, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/cache/zihaohe/visualizations/affinity_diag_v9")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--n_train", type=int, default=3)
    ap.add_argument("--n_val", type=int, default=6)
    ap.add_argument("--pair_planes", type=str, default="0,1;3,4;6,7",
                    help="V-JEPA token plane pairs to compare (semicolon-separated)")
    ap.add_argument("--v9_ckpt", default="/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[affinity_diag] device: {device}")

    # === Load v9.0_best (gives us both backbone and pixel_branch for pred) ===
    cfg = load_config("configs/train_v9.yaml", overrides=[])
    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    model.eval()
    state = torch.load(args.v9_ckpt, map_location="cpu", weights_only=False)
    model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    print(f"[affinity_diag] loaded v9.0_best from {args.v9_ckpt}")
    backbone = model.backbone

    # === Parse plane pairs ===
    pair_planes = [tuple(map(int, p.split(","))) for p in args.pair_planes.split(";")]
    print(f"[affinity_diag] V-JEPA token plane pairs to compare: {pair_planes}")

    # === Pick clips ===
    train_seqs = list_seqs(args.val_root, "train")[:args.n_train]
    val_seqs = list_seqs(args.val_root, "test")[:args.n_val]

    rows_csv = []
    summary_per_mod = {"LQ": [], "pred": [], "HQ": []}

    for split, seq_list in (("train", train_seqs), ("val", val_seqs)):
        for seq_name, lq_paths, gt_paths in seq_list:
            print(f"\n[affinity_diag] {split}/{seq_name} ({len(lq_paths)} frames)")
            lq, gt = load_clip_center(lq_paths, gt_paths, n_frames=16, crop_size=256)
            lq, gt = lq.to(device), gt.to(device)

            # Pred from v9.0_best
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model.sample(lq)
            pred = out["pred_pixels"].clamp(0, 1).float()

            for ip, (pa, pb) in enumerate(pair_planes):
                # original frame indices: V-JEPA tubelet_size=2 → plane p covers frames 2p, 2p+1
                frame_t_idx = pa * 2  # representative frame for visualization

                pair_dir = os.path.join(args.out_root, split, seq_name, f"pair{ip}_planes{pa}-{pb}")
                os.makedirs(pair_dir, exist_ok=True)

                results = {}
                for mod_name, video in (("LQ", lq), ("pred", pred), ("HQ", gt)):
                    t0 = time.time()
                    results[mod_name] = diag_one_pair(
                        mod_name, video, backbone, pa, pb, pair_dir, frame_t_idx,
                    )
                    summary_per_mod[mod_name].append(results[mod_name]["stats"])
                    print(f"  [{mod_name}] mean_top1={results[mod_name]['stats']['mean_top1']:.3f} "
                          f"median_disp_tok={results[mod_name]['stats']['median_disp_tokens']:.2f} "
                          f"std_disp_tok={results[mod_name]['stats']['std_disp_tokens']:.2f} "
                          f"({time.time()-t0:.2f}s)")

                # closer-to-HQ score
                close_score = closer_to_hq_score(
                    results["pred"]["affinity"], results["LQ"]["affinity"], results["HQ"]["affinity"],
                )
                print(f"  closer-to-HQ score (pred vs LQ): {close_score:+.4f}")

                csv_row = {
                    "split": split, "seq": seq_name, "pair_idx": ip,
                    "plane_a": pa, "plane_b": pb,
                    "frame_t_idx": frame_t_idx,
                    "closer_to_hq": close_score,
                }
                for mod_name, r in results.items():
                    s = r["stats"]
                    csv_row[f"{mod_name}_mean_top1"] = s["mean_top1"]
                    csv_row[f"{mod_name}_median_disp_tok"] = s["median_disp_tokens"]
                    csv_row[f"{mod_name}_std_disp_tok"] = s["std_disp_tokens"]
                    csv_row[f"{mod_name}_top5_ent"] = s["mean_top5_entropy"]
                rows_csv.append(csv_row)

    # === per-pair CSV ===
    csv_path = os.path.join(args.out_root, "per_pair_affinity.csv")
    keys = list(rows_csv[0].keys()) if rows_csv else []
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)
    print(f"\n[affinity_diag] wrote {csv_path}")

    # === aggregate summary + verdict ===
    agg = {}
    for mod_name, stats_list in summary_per_mod.items():
        agg[mod_name] = {
            "mean_top1": float(np.mean([s["mean_top1"] for s in stats_list])),
            "median_disp_tok": float(np.mean([s["median_disp_tokens"] for s in stats_list])),
            "std_disp_tok": float(np.mean([s["std_disp_tokens"] for s in stats_list])),
            "mean_top5_ent": float(np.mean([s["mean_top5_entropy"] for s in stats_list])),
        }
    mean_close = float(np.mean([r["closer_to_hq"] for r in rows_csv]))

    # Verdict per AE3
    pred_top1 = agg["pred"]["mean_top1"]
    hq_top1 = agg["HQ"]["mean_top1"]
    if pred_top1 > 0.5 and mean_close >= 0.05 and pred_top1 > 0.7 * hq_top1:
        verdict = "USEFUL"
    elif pred_top1 < 0.3 or mean_close <= -0.02:
        verdict = "NOT USEFUL"
    else:
        verdict = "AMBIGUOUS"

    md_path = os.path.join(args.out_root, "affinity_summary.md")
    with open(md_path, "w") as f:
        f.write("# V-JEPA Token Affinity Diagnostic — Summary\n\n")
        f.write(f"Eval: {len(train_seqs)} train + {len(val_seqs)} val clips, "
                f"{len(pair_planes)} V-JEPA token-plane pairs each.\n")
        f.write(f"Token plane size: 16×16 = 256 tokens, dim 1024 (V-JEPA 2.1 ViT-L deepest layer).\n\n")
        f.write("## Aggregate stats per modality\n\n")
        f.write("| Modality | mean top-1 cosine | median disp (tok) | std disp (tok) | mean top-5 entropy |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for mod in ("LQ", "pred", "HQ"):
            a = agg[mod]
            f.write(f"| {mod} | {a['mean_top1']:.3f} | {a['median_disp_tok']:.2f} | "
                    f"{a['std_disp_tok']:.2f} | {a['mean_top5_ent']:.3f} |\n")
        f.write(f"\n**closer-to-HQ score (pred vs LQ): {mean_close:+.4f}** "
                f"(positive = pred affinity closer to HQ affinity than LQ is)\n\n")
        f.write(f"## VERDICT: **{verdict}**\n\n")
        f.write("Verdict criteria (per AE3):\n")
        f.write(f"- pred mean top-1 > 0.5: {'YES' if pred_top1 > 0.5 else 'NO'} ({pred_top1:.3f})\n")
        f.write(f"- closer-to-HQ ≥ +0.05: {'YES' if mean_close >= 0.05 else 'NO'} ({mean_close:+.4f})\n")
        f.write(f"- pred top-1 ≥ 70% of HQ top-1: {'YES' if pred_top1 > 0.7 * hq_top1 else 'NO'} "
                f"({pred_top1:.3f} vs {0.7 * hq_top1:.3f})\n")
        f.write(f"\nVisualizations under `{args.out_root}/{{train,val}}/<seq>/pair*/`:\n")
        f.write("- `affinity_{LQ,pred,HQ}.png` — 256×256 cosine affinity heatmaps\n")
        f.write("- `nn_overlay_{LQ,pred,HQ}.png` — NN displacement arrows on representative frame\n")
    print(f"[affinity_diag] wrote {md_path}")
    print(f"\n[affinity_diag] === VERDICT: {verdict} ===")
    print(f"[affinity_diag] DONE")


if __name__ == "__main__":
    main()
