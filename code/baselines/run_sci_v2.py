"""SCI eval v2 -- adds DISTS + CLIPIQA. Uses shared eval_lib_v2 framework."""
from __future__ import annotations
import os, sys, argparse
import torch

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/SCI/CVPR")

from common.eval_lib_v2 import evaluate_method_v2
from model import Finetunemodel

CKPT_PATH = "/cache/zihaohe/ext_methods/SCI/CVPR/weights/medium.pt"

def load_model(device):
    m = Finetunemodel(CKPT_PATH)
    return m.to(device).eval()

def make_infer_fn(model):
    @torch.no_grad()
    def infer(lq):
        i, r = model(lq)
        return r.clamp(0, 1)
    return infer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["SDSD_indoor", "SDSD_outdoor",
                             "BVI_RLV", "LOL_v1", "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/cache/zihaohe/checkpoint/runs/SCI_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(device)
    infer_fn = make_infer_fn(model)

    note = "SCI (CVPR 2022, Self-Calibrated Illumination), unsupervised Retinex-style single-image enhancement, 'medium' weights. Same checkpoint applied to all splits (no domain-specific finetuning)."

    r = evaluate_method_v2(
        method_name="SCI",
        infer_fn=infer_fn,
        datasets=args.datasets,
        out_path=args.out,
        notes=note,
        limit_seqs=args.limit,
    )
    print(f"\n[saved] {args.out}")
    for ds, m in r["summary"].items():
        print(f"  {ds}: psnr={m.get('psnr_rgb', float('nan')):.3f} lpips={m.get('lpips', float('nan')):.4f}")

if __name__ == "__main__":
    main()
