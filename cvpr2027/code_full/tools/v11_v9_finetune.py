"""K1: Finetune v9.0b ckpt on REPA protocol (LLVE-STCD-12 train/test, crop=256).

Loads the strong v9 architecture + ckpt, retrains for 1500 iter at our exact
protocol with Charbonnier-only loss. Tests if v9.0b's 20.578 (zero-shot direct
eval) can be pushed by a brief targeted finetune.
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import load_config
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr,
                                           charbonnier, build_split_seqs)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.cfg, overrides=[])
    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    if args.ckpt and args.ckpt.lower() not in ("none", "scratch") and os.path.isfile(args.ckpt):
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        miss, unexp = model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
        print(f"[K] loaded {args.ckpt} | missing={len(miss)} unexpected={len(unexp)}")
    else:
        print(f"[K] FROM SCRATCH (no ckpt loaded; arg was {args.ckpt!r})")
    n_pb = sum(p.numel() for p in model.pixel_branch.parameters() if p.requires_grad)
    print(f"[K1] pixel_branch {n_pb/1e6:.1f}M trainable + 304M frozen V-JEPA")

    optim = torch.optim.AdamW(
        [p for p in model.pixel_branch.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        _, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(lq)
            pred = out["pred_pixels"].clamp(0, 1)
        L = charbonnier(pred.float(), gt)

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.pixel_branch.parameters() if p.requires_grad], 1.0,
        )
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item()})
            t0 = time.time()
        # Early periodic eval
        if args.eval_every > 0 and (it + 1) % args.eval_every == 0:
            _quick_eval(model, test_seqs, device, args.crop, prefix=f"@{it+1}")

    torch.save({"pixel_branch": model.pixel_branch.state_dict(),
                "args": vars(args), "log": log, "stage": "v9_repa_finetune"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Final eval
    print("\n=== FINAL EVAL: SDSD held-out (LLVE-STCD-12, crop=256) ===")
    psnrs = _quick_eval(model, test_seqs, device, args.crop, prefix="final", verbose=True)
    mean_p = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "K1_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# K1: v9 finetune on REPA protocol\n\n")
        f.write(f"ckpt source: {args.ckpt}\n")
        f.write(f"finetune iters: {args.iters}, lr: {args.lr}, crop: {args.crop}\n")
        f.write(f"pixel_branch: {n_pb/1e6:.1f}M trainable + 304M frozen V-JEPA\n\n")
        f.write(f"## SDSD held-out PSNR = {mean_p:.3f}  (vs v9.0b zero-shot 20.578, vs E2b 19.556, vs pure_rgb 18.533)\n\n")
        f.write(f"per-seq PSNRs: {psnrs}\n")
    print(f"\nK1 PSNR = {mean_p:.3f}  (vs v9.0b zero-shot 20.578)")
    print(f"wrote {md}")


@torch.no_grad()
def _quick_eval(model, test_seqs, device, crop, prefix="", verbose=False):
    model.eval()
    psnrs = []
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.sample(lq)
            pred = out["pred_pixels"].clamp(0, 1).float()
        p = mean_psnr(to_u8(pred), to_u8(gt))
        psnrs.append(p)
        if verbose:
            print(f"    [{name}] LQ→pred = {p:.3f}")
    print(f"  [{prefix}] mean PSNR = {float(np.mean(psnrs)):.3f}")
    model.train()
    if model.backbone is not None:
        model.backbone.eval()
    return psnrs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_v9_0b.yaml")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-5, help="v9.0b finetune used 2e-5")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
