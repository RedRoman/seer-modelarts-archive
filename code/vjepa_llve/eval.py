"""eval.py — V-JEPA-LLVE v5 evaluation.

Loads a trained ckpt, runs ODE sampling on the SDSD val set, computes
PSNR/SSIM/LPIPS/MABD, optionally dumps qualitative samples.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import build_dataset
from models.vjepa_diffusion import VJepaDiffusion
from utils import (
    AverageMeter, dump_config, ensure_dir, load_config, psnr,
    setup_logger, ssim, lpips_score, mabd, ewarp_proxy,
)


def save_triptych(lq, pred, hq, out_path: str) -> None:
    from PIL import Image
    T = lq.shape[2]
    t = T // 2
    arr_lq   = (lq  [0, :, t].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr_pred = (pred[0, :, t].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr_hq   = (hq  [0, :, t].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr_lq, arr_pred, arr_hq = (np.transpose(a, (1, 2, 0))
                                 for a in (arr_lq, arr_pred, arr_hq))
    Image.fromarray(np.concatenate([arr_lq, arr_pred, arr_hq], axis=1)).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_dir=out_dir, name="vjepa_eval")
    dump_config(cfg, os.path.join(out_dir, "eval_config.yaml"))
    device = torch.device(cfg.get("device", "cuda"))
    ecfg = cfg["eval"]

    val_set = build_dataset(
        cfg["data"]["val"],
        clip_len=ecfg.get("clip_len", 16),
        crop_size=ecfg.get("crop_size", 256),
        is_train=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=ecfg.get("batch_size", 1), shuffle=False,
        num_workers=cfg.get("num_workers", 4), pin_memory=True,
    )
    logger.info(f"Eval set: {len(val_set)} samples")

    model = VJepaDiffusion(cfg["model"]).to(device)
    model.freeze_backbone()

    ckpt_path = cfg.get("ckpt")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt={ckpt_path!r} missing. Pass `ckpt=/path/to/best.pt`.")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.dit.load_state_dict(state["dit"], strict=False)
    if model.predictor_enabled and "refiner" in state:
        model.refiner.load_state_dict(state["refiner"], strict=False)
    logger.info(f"Loaded ckpt from {ckpt_path} (step={state.get('step', 0)})")
    model.eval()

    inference_steps = int(ecfg.get("inference_steps", 8))

    psnr_m, ssim_m, lpips_m, mabd_m, ewp_m = (AverageMeter() for _ in range(5))
    per_seq = []
    samples_dir = ensure_dir(os.path.join(out_dir, "samples"))
    save_n = int(ecfg.get("save_samples", 0))
    saved = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            lq = batch["lq"].to(device, non_blocking=True)
            hq = batch["hq"].to(device, non_blocking=True)
            out = model.sample(lq, num_steps=inference_steps)
            pred = out["pred"]

            p = psnr(pred, hq).item()
            s = ssim(pred, hq).item()
            try:
                l = lpips_score(pred, hq).item()
            except Exception as e:
                logger.warning(f"LPIPS skipped at sample {i}: {e}")
                l = float("nan")
            mb = mabd(pred, hq).item()
            ew = ewarp_proxy(pred, hq).item()
            psnr_m.update(p); ssim_m.update(s); ewp_m.update(ew); mabd_m.update(mb)
            if not np.isnan(l):
                lpips_m.update(l)

            meta = batch.get("meta", {})
            seq_name = meta.get("seq", ["?"])
            seq_name = seq_name[0] if isinstance(seq_name, list) else str(seq_name)
            per_seq.append({
                "idx": i, "seq": seq_name,
                "psnr": p, "ssim": s, "lpips": l, "mabd": mb, "ewarp_proxy": ew,
            })

            if saved < save_n:
                save_triptych(lq, pred, hq,
                              os.path.join(samples_dir, f"sample_{i:04d}.png"))
                saved += 1

            if (i + 1) % 5 == 0:
                logger.info(
                    f"  {i+1}/{len(val_loader)} | "
                    f"PSNR {psnr_m.avg:.3f}  SSIM {ssim_m.avg:.4f}  "
                    f"LPIPS {lpips_m.avg:.4f}  MABD {mabd_m.avg:.4f}"
                )

    metrics = {
        "psnr": psnr_m.avg, "ssim": ssim_m.avg,
        "lpips": lpips_m.avg if lpips_m.count else None,
        "mabd": mabd_m.avg, "ewarp_proxy": ewp_m.avg,
        "n": psnr_m.count, "ckpt": ckpt_path,
        "inference_steps": inference_steps,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Aggregate: {metrics}")

    keys = list(per_seq[0].keys())
    with open(os.path.join(out_dir, "per_sequence_metrics.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_seq:
            w.writerow(row)


if __name__ == "__main__":
    main()
