"""Zero-DCE eval v2 -- adds DISTS + CLIPIQA. Uses shared eval_lib_v2 framework."""
from __future__ import annotations
import os, sys, argparse
import torch

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/Zero-DCE/Zero-DCE_code")

from common.eval_lib_v2 import evaluate_method_v2
import model as zdce_model

CKPT_PATH = "/cache/zihaohe/ext_methods/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth"

def load_model(device):
    net = zdce_model.enhance_net_nopool().to(device)
    net.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    return net.eval()

def make_infer_fn(model):
    @torch.no_grad()
    def infer(lq):
        _, pred, _ = model(lq)
        return pred.clamp(0, 1)
    return infer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["SDSD_indoor", "SDSD_outdoor",
                             "BVI_RLV", "LOL_v1", "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/cache/zihaohe/checkpoint/runs/Zero-DCE_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(device)
    infer_fn = make_infer_fn(model)

    note = "Zero-DCE (CVPR 2020), zero-reference unsupervised curve estimation, no GT training. Same checkpoint applied to all splits (no domain-specific finetuning)."

    r = evaluate_method_v2(
        method_name="Zero-DCE",
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
