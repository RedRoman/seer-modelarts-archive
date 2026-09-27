"""CP-transfer experiment, DarkIR variant (second baseline; L2 BVI-RLV
per-frame perceptual leader in Table 1). Same protocol/logic as
run_cp_transfer_retinexformer.py: fit alpha* on 15 held-out SDSD-indoor
validation pairs using DarkIR's own predictions, then apply CP to DarkIR's
full L2-L5 test-set predictions.

DarkIR checkpoint: DarkIR_384.pt (LOL-Blur trained; the only officially
reachable public weight; used cross-domain on ALL splits including SDSD,
matching run_darkir_v2.py / REPORT_v2.md's Table-1 methodology exactly).
Requires reflect-pad to multiple of 8 before the forward pass, unpadded on
output (same as run_darkir_v2.py's infer wrapper).
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/DarkIR")
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/quick_ablations/cp_transfer")

from common.eval_lib import (
    DATA_ROOTS, STCD_INDOOR_12, _load_img_any, _list_sorted,
    rgb_psnr_uint8, get_lpips_model, lpips_batch,
    iter_bvi, iter_lol_v1, iter_lol_v2,
)
from archs.DarkIR import DarkIR
from cp_module import cp_surrogate, fuse_per_channel, fit_alpha_star

DEVICE = torch.device("cuda")
OUT_DIR = "/home/ma-user/workspace/zihaohe/quick_ablations/cp_transfer"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_PATH = "/cache/zihaohe/baseline_ckpts/DarkIR/DarkIR_384.pt"


def load_model(device):
    m = DarkIR(img_channel=3, width=32,
               middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
               dilations=[1, 4, 9], extra_depth_wise=True)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["params"], strict=True)
    return m.to(device).eval()


@torch.no_grad()
def darkir_forward(model, lq: torch.Tensor) -> torch.Tensor:
    _, _, H, W = lq.shape
    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph or pw:
        lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
    y = model(lq)[:, :, :H, :W]
    return y.clamp(0, 1)


@torch.no_grad()
def infer_frame(model, hwc_u8: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(hwc_u8).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    y = darkir_forward(model, x).float()
    return y[0].permute(1, 2, 0).cpu().numpy()


VAL15 = ['pair1', 'pair16_2', 'pair1_2', 'pair2', 'pair24', 'pair26', 'pair27_2',
         'pair31_2', 'pair34', 'pair34_2', 'pair4_2', 'pair5', 'pair8', 'pair9', 'pair9_2']


def fit_alpha(model):
    assert not (set(VAL15) & set(STCD_INDOOR_12)), "validation pool overlaps test split!"
    root = DATA_ROOTS["SDSD_indoor"]
    lq_root = os.path.join(root, "indoor_static_np", "input")
    gt_root = os.path.join(root, "indoor_static_np", "GT")

    pred_list, lq_list, gt_list = [], [], []
    per_pair_info = []
    for seq in VAL15:
        lq_dir = os.path.join(lq_root, seq); gt_dir = os.path.join(gt_root, seq)
        lq_files = _list_sorted(lq_dir); gt_files = _list_sorted(gt_dir)
        # index-position pairing (see run_cp_transfer_retinexformer.py note:
        # some SDSD "_2" sequences have offset LQ/GT filenames)
        n = min(len(lq_files), len(gt_files))
        lq_files = lq_files[:n]; gt_files = gt_files[:n]
        preds, lqs, gts = [], [], []
        for lqp, gtp in zip(lq_files, gt_files):
            lq_img = _load_img_any(lqp)
            gt_img = _load_img_any(gtp)
            pred_img = infer_frame(model, lq_img)
            preds.append(pred_img.transpose(2, 0, 1))
            lqs.append(lq_img.astype(np.float32).transpose(2, 0, 1) / 255.0)
            gts.append(gt_img.astype(np.float32).transpose(2, 0, 1) / 255.0)
        pred_stack = np.stack(preds, axis=1)
        lq_stack = np.stack(lqs, axis=1)
        gt_stack = np.stack(gts, axis=1)
        pred_list.append(pred_stack); lq_list.append(lq_stack); gt_list.append(gt_stack)
        per_pair_info.append({"seq": seq, "n_frames": n})
        print(f"[fit] {seq}: {n} frames processed", flush=True)

    alpha, moments = fit_alpha_star(pred_list, lq_list, gt_list)
    return alpha, moments, per_pair_info


@torch.no_grad()
def eval_split_video(model, alpha_t, ds_name, n_frames=16, crop=512):
    raw_psnr, raw_lpips, cp_psnr, cp_lpips = [], [], [], []
    per_seq = {}
    for seq_name, lq_arr, gt_arr in iter_bvi(n_frames=n_frames, crop=crop):
        T = lq_arr.shape[0]
        ps_r, lp_r, ps_c, lp_c = [], [], [], []
        for t in range(T):
            lq_t = torch.from_numpy(lq_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            gt_t = torch.from_numpy(gt_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            pred = darkir_forward(model, lq_t).float()
            cp_surr = cp_surrogate(pred, lq_t)
            pred_cp = fuse_per_channel(pred, cp_surr, alpha_t)

            pred_u8 = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cp_u8 = (pred_cp[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_u8 = (gt_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            ps_r.append(rgb_psnr_uint8(pred_u8, gt_u8))
            ps_c.append(rgb_psnr_uint8(cp_u8, gt_u8))
            lp_r.append(lpips_batch(pred, gt_t))
            lp_c.append(lpips_batch(pred_cp, gt_t))
        mp_r, ml_r = float(np.mean(ps_r)), float(np.mean(lp_r))
        mp_c, ml_c = float(np.mean(ps_c)), float(np.mean(lp_c))
        per_seq[seq_name] = {"raw_psnr": mp_r, "raw_lpips": ml_r, "cp_psnr": mp_c, "cp_lpips": ml_c, "n_frames": T}
        raw_psnr.append(mp_r); raw_lpips.append(ml_r); cp_psnr.append(mp_c); cp_lpips.append(ml_c)
        print(f"  [{ds_name}] {seq_name}: raw PSNR={mp_r:.3f} LPIPS={ml_r:.4f} | "
              f"+CP PSNR={mp_c:.3f} LPIPS={ml_c:.4f}", flush=True)
    return {
        "raw_psnr": float(np.mean(raw_psnr)), "raw_lpips": float(np.mean(raw_lpips)),
        "cp_psnr": float(np.mean(cp_psnr)), "cp_lpips": float(np.mean(cp_lpips)),
        "n_seqs": len(raw_psnr), "per_seq": per_seq,
    }


@torch.no_grad()
def eval_split_image(model, alpha_t, ds_name, iterator):
    raw_psnr, raw_lpips, cp_psnr, cp_lpips = [], [], [], []
    per_img = {}
    for name, lq_stk, gt_stk in iterator:
        lq_img = lq_stk[0]; gt_img = gt_stk[0]
        lq_t = torch.from_numpy(lq_img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        pred = darkir_forward(model, lq_t).float()
        cp_surr = cp_surrogate(pred, lq_t)
        pred_cp = fuse_per_channel(pred, cp_surr, alpha_t)

        pred_u8 = (pred[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cp_u8 = (pred_cp[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gt_u8 = (gt_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        pr = rgb_psnr_uint8(pred_u8, gt_u8); pc = rgb_psnr_uint8(cp_u8, gt_u8)
        lr = lpips_batch(pred, gt_t); lc = lpips_batch(pred_cp, gt_t)
        per_img[name] = {"raw_psnr": pr, "raw_lpips": lr, "cp_psnr": pc, "cp_lpips": lc}
        raw_psnr.append(pr); raw_lpips.append(lr); cp_psnr.append(pc); cp_lpips.append(lc)
    print(f"  [{ds_name}] MEAN: raw PSNR={np.mean(raw_psnr):.3f} LPIPS={np.mean(raw_lpips):.4f} | "
          f"+CP PSNR={np.mean(cp_psnr):.3f} LPIPS={np.mean(cp_lpips):.4f}  (n={len(raw_psnr)})", flush=True)
    return {
        "raw_psnr": float(np.mean(raw_psnr)), "raw_lpips": float(np.mean(raw_lpips)),
        "cp_psnr": float(np.mean(cp_psnr)), "cp_lpips": float(np.mean(cp_lpips)),
        "n_imgs": len(raw_psnr), "per_img": per_img,
    }


def main():
    t_start = time.time()
    model = load_model(DEVICE)
    print(f"[main] DarkIR loaded from {CKPT_PATH}", flush=True)

    print("\n===== Fitting alpha* on 15 SDSD-indoor validation pairs (DarkIR) =====", flush=True)
    alpha, moments, pair_info = fit_alpha(model)
    print(f"[fit] alpha* = {alpha.tolist()}", flush=True)
    print(f"[fit] moments = {moments}", flush=True)
    alpha_t = torch.from_numpy(alpha).float()

    results = {
        "method": "DarkIR",
        "checkpoint": CKPT_PATH,
        "alpha_star_darkir": alpha.tolist(),
        "alpha_star_seer_reference": [0.50, 0.50, 0.68],
        "fit_moments": moments,
        "fit_val_pairs": pair_info,
        "fit_n_pairs": len(pair_info),
        "splits": {},
    }

    print("\n===== L2 BVI-RLV =====", flush=True)
    results["splits"]["L2_BVI_RLV"] = eval_split_video(model, alpha_t, "L2_BVI_RLV")

    print("\n===== L3 LOL-v1 =====", flush=True)
    results["splits"]["L3_LOL_v1"] = eval_split_image(model, alpha_t, "L3_LOL_v1", iter_lol_v1())

    print("\n===== L4 LOLv2-Syn =====", flush=True)
    results["splits"]["L4_LOLv2_Syn"] = eval_split_image(model, alpha_t, "L4_LOLv2_Syn", iter_lol_v2("Syn"))

    print("\n===== L5 LOLv2-Real =====", flush=True)
    results["splits"]["L5_LOLv2_Real"] = eval_split_image(model, alpha_t, "L5_LOLv2_Real", iter_lol_v2("Real"))

    out_path = os.path.join(OUT_DIR, "darkir_cp_transfer.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {out_path}", flush=True)

    print("\n===== SUMMARY (DarkIR, own-fit alpha) =====")
    for k, v in results["splits"].items():
        dP = v["cp_psnr"] - v["raw_psnr"]
        dL = v["raw_lpips"] - v["cp_lpips"]
        n = v.get("n_seqs", v.get("n_imgs"))
        print(f"  {k:14s} raw={v['raw_psnr']:.3f}/{v['raw_lpips']:.4f}  "
              f"+CP={v['cp_psnr']:.3f}/{v['cp_lpips']:.4f}  "
              f"dPSNR={dP:+.3f} dLPIPS={dL:+.4f}  (n={n})")
    print(f"\nTotal wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
