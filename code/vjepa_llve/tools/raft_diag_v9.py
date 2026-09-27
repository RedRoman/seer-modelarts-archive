"""tools/raft_diag_v9.py — real-flow diagnostic per reviewer O6.

Goal: answer "is RAFT flow useful enough on real low-light SDSD frames to
supervise enhancement, or is the LQ-flow itself the weak link?"

For each selected clip, save PNG grids and numerical reductions for 3 variants:

  Variant A: flow estimated from LQ_t, LQ_{t+1}; eval warp on LQ and HQ.
  Variant B: flow estimated from v9.0_pred_t, v9.0_pred_{t+1}; eval warp on pred and HQ.
  Variant C (oracle): flow from HQ_t, HQ_{t+1}; eval on HQ.

Numeric per-pair: unwarped L1, warped masked L1, reduction %, mask mean.

Outputs:
  /cache/zihaohe/visualizations/raft_diag_v9_1a/{train,val}/{seq}/grid_{t}.png
  /cache/zihaohe/visualizations/raft_diag_v9_1a/diag_summary.csv
  /cache/zihaohe/visualizations/raft_diag_v9_1a/diag_summary.md
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.common import read_image_01
from data.sdsd_dataset import SDSDDataset
from losses.v9_1 import warp_with_flow, occlusion_mask_fb, masked_l1
from models.raft_flow import build_raft_small, _raft_pair
from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import load_config


def list_sdsd_sequences(root: str, split: str):
    ds = SDSDDataset(root=root, subset="direct", split=split,
                     clip_len=16, crop_size=256, is_train=False)
    return list(ds.seqs)


def load_pair(lq_paths, gt_paths, idx_t: int, crop_size: int = 256):
    """Load (LQ_t, LQ_{t+1}, HQ_t, HQ_{t+1}) center-cropped."""
    def _load(p):
        a = read_image_01(p)  # (H, W, 3) float32
        return torch.from_numpy(a).permute(2, 0, 1)  # (3, H, W)
    lq_t = _load(lq_paths[idx_t])
    lq_tp1 = _load(lq_paths[idx_t + 1])
    gt_t = _load(gt_paths[idx_t])
    gt_tp1 = _load(gt_paths[idx_t + 1])
    H, W = lq_t.shape[-2:]
    if H >= crop_size and W >= crop_size:
        h0 = (H - crop_size) // 2
        w0 = (W - crop_size) // 2
        lq_t = lq_t[..., h0:h0 + crop_size, w0:w0 + crop_size]
        lq_tp1 = lq_tp1[..., h0:h0 + crop_size, w0:w0 + crop_size]
        gt_t = gt_t[..., h0:h0 + crop_size, w0:w0 + crop_size]
        gt_tp1 = gt_tp1[..., h0:h0 + crop_size, w0:w0 + crop_size]
    return lq_t.unsqueeze(0), lq_tp1.unsqueeze(0), gt_t.unsqueeze(0), gt_tp1.unsqueeze(0)


@torch.no_grad()
def flow_pair(raft, img1: torch.Tensor, img2: torch.Tensor, threshold_px: float = 1.0):
    """Compute (flow_back, mask, flow_fwd) for the pair (img1=t, img2=t+1).

    flow_back is flow_{t+1 → t}, used to warp img2 back to img1's grid.
    """
    flow_fwd = _raft_pair(raft, img1, img2)  # t → t+1
    flow_bwd = _raft_pair(raft, img2, img1)  # t+1 → t  (= flow_back)
    mask = occlusion_mask_fb(flow_fwd, flow_bwd, threshold_px=threshold_px)
    return flow_bwd, mask, flow_fwd


def to_png(t: torch.Tensor) -> np.ndarray:
    """t (3,H,W) or (1,H,W) in [0,1] → uint8 (H,W,3)."""
    t = t.clamp(0, 1).cpu().numpy()
    if t.shape[0] == 1:
        t = np.repeat(t, 3, axis=0)
    arr = (np.transpose(t, (1, 2, 0)) * 255).astype(np.uint8)
    return arr


def err_to_png(diff: torch.Tensor, scale: float = 5.0) -> np.ndarray:
    """diff (3,H,W) or (1,H,W) → uint8 heatmap."""
    err = diff.abs().mean(dim=0, keepdim=True)  # (1,H,W)
    err = (err * scale).clamp(0, 1)
    return to_png(err)


def mask_to_png(mask: torch.Tensor) -> np.ndarray:
    """mask can be (1,H,W) or (H,W). Returns (H,W,3) uint8."""
    while mask.dim() > 2:
        mask = mask.squeeze(0)
    return to_png(mask.unsqueeze(0))


def run_variant(variant: str, raft, model, lq_t, lq_tp1, gt_t, gt_tp1, device,
                threshold_px: float = 1.0):
    """Returns dict with flow, mask, warped, error metrics, and viz tensors."""
    lq_t, lq_tp1 = lq_t.to(device), lq_tp1.to(device)
    gt_t, gt_tp1 = gt_t.to(device), gt_tp1.to(device)

    if variant == "A":  # LQ-flow
        flow_back, mask, _ = flow_pair(raft, lq_t, lq_tp1, threshold_px=threshold_px)
        # warp evaluations
        warped_lq = warp_with_flow(lq_tp1, flow_back)
        warped_hq = warp_with_flow(gt_tp1, flow_back)
        return {
            "variant": variant,
            "flow_back": flow_back,
            "mask": mask,
            "img_t_top": lq_t[0], "img_tp1_top": lq_tp1[0], "warped_top": warped_lq[0],
            "img_t_mid": gt_t[0], "img_tp1_mid": gt_tp1[0], "warped_mid": warped_hq[0],
            "err_lq_unwarped": float(masked_l1(lq_t - lq_tp1, mask)),
            "err_lq_warped":   float(masked_l1(lq_t - warped_lq, mask)),
            "err_hq_unwarped": float(masked_l1(gt_t - gt_tp1, mask)),
            "err_hq_warped":   float(masked_l1(gt_t - warped_hq, mask)),
            "mask_mean": float(mask.mean()),
        }
    elif variant == "B":  # pred-flow
        if model is None:
            return None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # v9 needs T>=8 (V-JEPA grid). Build a 16-frame clip by repeating lq_t/lq_tp1.
            # We'll keep frames 7 and 8 as our pair after inference.
            T_NEED = 16
            half = T_NEED // 2
            lq_clip = torch.cat(
                [lq_t.unsqueeze(2)] * half + [lq_tp1.unsqueeze(2)] * half, dim=2,
            )  # (1,3,16,H,W)
            out = model.sample(lq_clip)
        pred = out["pred_pixels"].clamp(0, 1).float()
        pred_t = pred[:, :, half - 1]
        pred_tp1 = pred[:, :, half]
        flow_back, mask, _ = flow_pair(raft, pred_t, pred_tp1, threshold_px=threshold_px)
        warped_pred = warp_with_flow(pred_tp1, flow_back)
        warped_hq = warp_with_flow(gt_tp1, flow_back)
        return {
            "variant": variant,
            "flow_back": flow_back,
            "mask": mask,
            "img_t_top": pred_t[0], "img_tp1_top": pred_tp1[0], "warped_top": warped_pred[0],
            "img_t_mid": gt_t[0], "img_tp1_mid": gt_tp1[0], "warped_mid": warped_hq[0],
            "err_pred_unwarped": float(masked_l1(pred_t - pred_tp1, mask)),
            "err_pred_warped":   float(masked_l1(pred_t - warped_pred, mask)),
            "err_hq_unwarped":   float(masked_l1(gt_t - gt_tp1, mask)),
            "err_hq_warped":     float(masked_l1(gt_t - warped_hq, mask)),
            "mask_mean": float(mask.mean()),
        }
    elif variant == "C":  # HQ-flow (oracle)
        flow_back, mask, _ = flow_pair(raft, gt_t, gt_tp1, threshold_px=threshold_px)
        warped_hq = warp_with_flow(gt_tp1, flow_back)
        warped_lq = warp_with_flow(lq_tp1, flow_back)
        return {
            "variant": variant,
            "flow_back": flow_back,
            "mask": mask,
            "img_t_top": gt_t[0], "img_tp1_top": gt_tp1[0], "warped_top": warped_hq[0],
            "img_t_mid": lq_t[0], "img_tp1_mid": lq_tp1[0], "warped_mid": warped_lq[0],
            "err_hq_unwarped": float(masked_l1(gt_t - gt_tp1, mask)),
            "err_hq_warped":   float(masked_l1(gt_t - warped_hq, mask)),
            "err_lq_unwarped": float(masked_l1(lq_t - lq_tp1, mask)),
            "err_lq_warped":   float(masked_l1(lq_t - warped_lq, mask)),
            "mask_mean": float(mask.mean()),
        }
    raise ValueError(variant)


def save_grid(out_path: str, result: dict, label_top: str, label_mid: str):
    """5-column grid: t | t+1 | warped | err | mask."""
    rows = []
    for which in ("top", "mid"):
        img_t = to_png(result[f"img_t_{which}"])
        img_tp1 = to_png(result[f"img_tp1_{which}"])
        warped = to_png(result[f"warped_{which}"])
        diff = result[f"img_t_{which}"] - result[f"warped_{which}"]
        err = err_to_png(diff)
        mask = mask_to_png(result["mask"][0])
        row = np.concatenate([img_t, img_tp1, warped, err, mask], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    # caption strip
    H_strip = 20
    strip = np.full((H_strip, grid.shape[1], 3), 255, dtype=np.uint8)
    grid = np.concatenate([strip, grid], axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(grid).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/cache/zihaohe/visualizations/raft_diag_v9_1a")
    ap.add_argument("--val_root", default="/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor")
    ap.add_argument("--n_train", type=int, default=3)
    ap.add_argument("--n_val", type=int, default=3)
    ap.add_argument("--frame_pairs_per_clip", type=int, default=3)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--threshold_px", type=float, default=1.0)
    ap.add_argument("--v9_ckpt", default="/cache/zihaohe/checkpoint/runs/v9/ckpts/best.pt")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device: {device}")

    raft = build_raft_small(device, weights="C_T_V2")

    # v9.0 model for variant B
    cfg = load_config("configs/train_v9.yaml", overrides=[])
    v9 = VJepaLLVE_v9(cfg["model"]).to(device)
    v9.freeze_backbone()
    v9.eval()
    state = torch.load(args.v9_ckpt, map_location="cpu", weights_only=False)
    v9.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    print(f"[diag] loaded v9.0 best.pt")

    train_seqs = list_sdsd_sequences(args.val_root, "train")[:args.n_train]
    val_seqs = list_sdsd_sequences(args.val_root, "test")[:args.n_val]

    rows_csv = []

    for split, seq_list in (("train", train_seqs), ("val", val_seqs)):
        for seq_name, lq_paths, gt_paths in seq_list:
            n = len(lq_paths)
            if n < 2:
                continue
            # pick frame_pairs_per_clip evenly spaced pairs
            n_pairs = min(args.frame_pairs_per_clip, n - 1)
            indices = np.linspace(0, n - 2, n_pairs, dtype=int)
            for ip, idx_t in enumerate(indices):
                lq_t, lq_tp1, gt_t, gt_tp1 = load_pair(
                    lq_paths, gt_paths, int(idx_t), crop_size=args.crop_size,
                )
                for variant in ("A", "B", "C"):
                    t0 = time.time()
                    r = run_variant(variant, raft, v9 if variant == "B" else None,
                                    lq_t, lq_tp1, gt_t, gt_tp1, device,
                                    threshold_px=args.threshold_px)
                    if r is None:
                        continue
                    out_path = os.path.join(args.out_root, split, seq_name,
                                             f"pair{ip}_t{idx_t}_var{variant}.png")
                    save_grid(out_path, r, f"top-{variant}", f"mid-{variant}")
                    csv_row = {
                        "split": split, "seq": seq_name, "pair_idx": ip,
                        "frame_t": int(idx_t), "variant": variant,
                        "mask_mean": r["mask_mean"],
                        "time_s": round(time.time() - t0, 2),
                    }
                    for k in ["err_lq_unwarped", "err_lq_warped",
                              "err_hq_unwarped", "err_hq_warped",
                              "err_pred_unwarped", "err_pred_warped"]:
                        csv_row[k] = r.get(k)
                    rows_csv.append(csv_row)
                    print(f"  [{split}/{seq_name} pair{ip} var{variant}] mask={r['mask_mean']:.3f} "
                          f"keys={list(k for k in r if k.startswith('err_'))}")

    # csv
    csv_path = os.path.join(args.out_root, "diag_summary.csv")
    keys = ["split", "seq", "pair_idx", "frame_t", "variant", "mask_mean",
            "err_lq_unwarped", "err_lq_warped",
            "err_hq_unwarped", "err_hq_warped",
            "err_pred_unwarped", "err_pred_warped", "time_s"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows_csv:
            w.writerow({k: row.get(k) for k in keys})
    print(f"\n[diag] wrote {csv_path}")

    # md summary
    md_path = os.path.join(args.out_root, "diag_summary.md")
    with open(md_path, "w") as f:
        f.write("# RAFT Real-Flow Diagnostic Summary\n\n")
        f.write(f"Variants:  A=LQ-flow,  B=pred-flow (v9.0_best),  C=HQ-flow (oracle).\n")
        f.write(f"Crop: center-{args.crop_size}.  FB threshold: {args.threshold_px} px.\n\n")
        # aggregate per variant
        for variant in ("A", "B", "C"):
            sub = [r for r in rows_csv if r["variant"] == variant]
            if not sub:
                continue
            f.write(f"## Variant {variant}\n\n")
            f.write(f"Pairs: {len(sub)}.  Mean mask coverage: "
                    f"{np.mean([r['mask_mean'] for r in sub]):.3f}\n\n")
            # for each error type, print mean reduction
            for unw_k, w_k, label in [
                ("err_lq_unwarped", "err_lq_warped", "LQ-pair"),
                ("err_hq_unwarped", "err_hq_warped", "HQ-pair (using this variant's flow)"),
                ("err_pred_unwarped", "err_pred_warped", "pred-pair"),
            ]:
                vals_u = [r[unw_k] for r in sub if r.get(unw_k) is not None]
                vals_w = [r[w_k] for r in sub if r.get(w_k) is not None]
                if not vals_u:
                    continue
                mu_u = float(np.mean(vals_u))
                mu_w = float(np.mean(vals_w))
                red = (1 - mu_w / max(mu_u, 1e-6)) * 100
                f.write(f"- {label}: unwarped L1 = {mu_u:.4f},  warped masked L1 = {mu_w:.4f},  "
                        f"reduction = {red:.1f}%\n")
            f.write("\n")
    print(f"[diag] wrote {md_path}")
    print(f"\n[diag] DONE — visualizations saved under {args.out_root}/{{train,val}}/")


if __name__ == "__main__":
    main()
