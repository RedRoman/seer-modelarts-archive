"""Eval E2b (or any REPA) ckpt on BVI-RLV held-out test."""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from data.bvi_rlv_dataset import BVIRLVDataset
from tools.v11_oracle_delta_probe import to_u8, mean_psnr


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    decoder.load_state_dict(sd["decoder"])
    print(f"[ckpt] loaded {args.ckpt}")

    ds = BVIRLVDataset(root=args.bvi_root, split="test", clip_len=16,
                        crop_size=args.crop, is_train=False)
    print(f"[BVI-RLV test] {len(ds)} sequences")

    psnrs = []
    for i in range(len(ds)):
        item = ds[i]
        lq = item["lq"].unsqueeze(0).to(device)
        gt = item["hq"].unsqueeze(0).to(device)
        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        psnr = mean_psnr(to_u8(pred.float()), to_u8(gt))
        psnr_lq = mean_psnr(to_u8(lq), to_u8(gt))
        psnrs.append(psnr)
        print(f"  pair {i}: LQ={psnr_lq:.2f} pred={psnr:.3f}")

    mean = float(np.mean(psnrs))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "BVI_RLV_EVAL.md"), "w") as f:
        f.write(f"# BVI-RLV eval of {args.ckpt}\n\n## mean PSNR = {mean:.3f}\n")
        f.write(f"per-pair PSNRs: {psnrs}\n")
    print(f"\n[BVI-RLV] mean PSNR = {mean:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bvi_root", default="/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos")
    ap.add_argument("--crop", type=int, default=256)
    main(ap.parse_args())
