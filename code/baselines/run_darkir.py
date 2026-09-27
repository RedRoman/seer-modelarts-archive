"""DarkIR eval on all 6 datasets.

Weight available on HuggingFace: DarkIR_384.pt (LOL-Blur trained, width=32).
The LOL / LOLv2 / LSRW multi-task checkpoint (DarkIR_1k_cr_mt.pt) is on
OneDrive only and requires auth; unreachable from this host. So we run
DarkIR_384.pt cross-domain on every dataset and note the mismatch.

Arch config comes from options/test/LOLBlur.yml which matches the ckpt exactly:
  width=32, enc=[1,2,3], dec=[3,1,1], mid_enc=mid_dec=2, dilations=[1,4,9],
  extra_depth_wise=True.
"""
from __future__ import annotations
import os, sys, argparse, json
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/DarkIR")

from common.eval_lib import evaluate_method
from archs.DarkIR import DarkIR


CKPT_PATH = "/cache/zihaohe/baseline_ckpts/DarkIR/DarkIR_384.pt"


def load_model(device):
    m = DarkIR(img_channel=3, width=32,
               middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1],
               dilations=[1, 4, 9], extra_depth_wise=True)
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["params"], strict=True)
    return m.to(device).eval()


def make_infer_fn(model):
    # DarkIR has 3 downsample stages -> padder_size = 8. It also has an
    # internal check_image_size that reflects-pads to that size, so we can
    # just feed the raw tensor and rely on the model's padding. But calling
    # check_image_size and cropping afterward is cleaner.
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph or pw:
            lq = F.pad(lq, (0, pw, 0, ph), mode="reflect")
        y = model(lq)[:, :, :H, :W]
        return y.clamp(0, 1)
    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["SDSD_indoor", "SDSD_outdoor",
                             "BVI_RLV", "LOL_v1", "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/cache/zihaohe/baseline_ckpts/DarkIR_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(device)
    infer_fn = make_infer_fn(model)

    note = "DarkIR_384.pt (LOL-Blur trained; cross-domain -- LOL/LOLv2/LSRW multi-task ckpt is on OneDrive only)"

    combined_per_seq = {}
    combined_summary = {}
    for ds in args.datasets:
        print(f"\n########## DarkIR @ {ds} ##########", flush=True)
        r = evaluate_method(
            method_name=f"DarkIR[{ds}]",
            infer_fn=infer_fn,
            datasets=[ds],
            out_path=f"/tmp/darkir_{ds}.json",
            notes=note,
            limit_seqs=args.limit,
        )
        combined_per_seq.update(r["per_seq"])
        combined_summary[ds] = r["summary"][ds]

    out = {
        "method": "DarkIR",
        "notes": ("DarkIR_384.pt (LOL-Blur trained; from HF Cidaut/DarkIR). "
                  "Multi-task LOL/LOLv2/LSRW weight DarkIR_1k_cr_mt.pt is on "
                  "OneDrive and requires auth (unreachable). Same single weight "
                  "used cross-domain on all 6 datasets."),
        "protocol": "video: center T=16, 512x512 center-crop; images: full-image. PSNR uint8 RGB, LPIPS pyiqa alex.",
        "per_seq": combined_per_seq,
        "summary": combined_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")

    print("\n===== DarkIR summary =====")
    for k, v in combined_summary.items():
        print(f"  {k:14s} PSNR={v['psnr_rgb']:.3f}  LPIPS={v['lpips']:.4f}  (n={v['n_seqs']})")


if __name__ == "__main__":
    main()
