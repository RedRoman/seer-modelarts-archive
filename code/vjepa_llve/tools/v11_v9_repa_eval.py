"""I1: Evaluate v9.0b best.pt (V-DUN family, 361M pixel branch) on REPA protocol.

Uses the SAME LLVE-STCD-12 held-out + center-crop 256 + 16 frames + mean_psnr
that v11 REPA experiments use, so the resulting PSNR is directly comparable to
E2b 19.556, pure_rgb 18.533, etc.
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_llve_v9 import VJepaLLVE_v9
from utils import load_config
from tools.v11_oracle_delta_probe import load_clip, to_u8, mean_psnr, build_split_seqs


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    cfg = load_config(args.cfg, overrides=[])
    model = VJepaLLVE_v9(cfg["model"]).to(device)
    model.freeze_backbone()
    model.eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    print(f"[ckpt] {args.ckpt} loaded; missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]: print(f"  missing first: {missing[:3]}")
    if unexpected[:3]: print(f"  unexpected first: {unexpected[:3]}")

    n_pb = sum(p.numel() for p in model.pixel_branch.parameters())
    print(f"[v9.0b] pixel_branch {n_pb/1e6:.1f}M trainable, V-JEPA frozen 304M")

    _, test_seqs = build_split_seqs()
    print(f"[split] held-out = {len(test_seqs)} sequences (LLVE-STCD-12)")

    psnrs = []
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.sample(lq)
        if "pred_pixels" in out:
            pred = out["pred_pixels"].clamp(0, 1).float()
        elif "pred" in out:
            pred = out["pred"].clamp(0, 1).float()
        else:
            raise RuntimeError(f"unknown output keys: {list(out.keys())}")
        p = mean_psnr(to_u8(pred), to_u8(gt))
        p_lq = mean_psnr(to_u8(lq), to_u8(gt))
        psnrs.append(p)
        print(f"    [{name}] LQ={p_lq:.2f}  pred={p:.3f}")

    mean_p = float(np.mean(psnrs))
    os.makedirs(args.out_dir, exist_ok=True)
    md = os.path.join(args.out_dir, "V9_REPA_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# v9.0b on REPA protocol (LLVE-STCD-12 held-out, crop=256, 16 frames)\n\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"pixel_branch: {n_pb/1e6:.1f}M trainable + 304M frozen V-JEPA\n\n")
        f.write(f"## SDSD held-out PSNR = {mean_p:.3f}  (vs REPA E2b 19.556, vs pure_rgb 18.533)\n\n")
        f.write(f"per-seq PSNRs: {psnrs}\n")
    print(f"\nv9.0b PSNR (REPA protocol) = {mean_p:.3f}  (vs REPA E2b 19.556)")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_v9_0b.yaml")
    ap.add_argument("--ckpt", default="/cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/v9_repa_eval")
    ap.add_argument("--crop", type=int, default=256)
    main(ap.parse_args())
