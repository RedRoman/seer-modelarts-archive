"""HVI-CIDNet eval v2 -- adds DISTS + CLIPIQA."""
from __future__ import annotations
import os, sys, argparse, json
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines/HVI-CIDNet")

from common.eval_lib_v2 import evaluate_method_v2
from net.CIDNet import CIDNet


CKPT = {
    "LOLv1":          "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/LOLv1-wperc/pytorch_model.bin",
    "LOLv2_real":     "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/LOLv2_real-bestPSNR/pytorch_model.bin",
    "LOLv2_syn":      "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/LOLv2_syn-wperc/pytorch_model.bin",
    "Generalization": "/cache/zihaohe/baseline_ckpts/HVI-CIDNet/Generalization/pytorch_model.bin",
}


def load_model(weights_path, device):
    m = CIDNet()
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def make_infer_fn(model):
    @torch.no_grad()
    def infer(lq):
        _, _, H, W = lq.shape
        ph = (16 - H % 16) % 16
        pw = (16 - W % 16) % 16
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
                    default="/cache/zihaohe/baseline_ckpts/HVI-CIDNet_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    lol1  = load_model(CKPT["LOLv1"], device)
    lol2r = load_model(CKPT["LOLv2_real"], device)
    lol2s = load_model(CKPT["LOLv2_syn"], device)
    gen   = load_model(CKPT["Generalization"], device)

    infer_lol1  = make_infer_fn(lol1)
    infer_lol2r = make_infer_fn(lol2r)
    infer_lol2s = make_infer_fn(lol2s)
    infer_gen   = make_infer_fn(gen)

    weight_map = {
        "LOL_v1":       (infer_lol1,  "LOLv1-wperc.bin (matched)"),
        "LOLv2_Syn":    (infer_lol2s, "LOLv2-syn-wperc.bin (matched)"),
        "LOLv2_Real":   (infer_lol2r, "LOLv2-real-bestPSNR.bin (matched)"),
        "SDSD_indoor":  (infer_gen,   "Generalization.bin (cross-domain; LOLv2-syn+random-gamma-trained)"),
        "SDSD_outdoor": (infer_gen,   "Generalization.bin (cross-domain)"),
        "BVI_RLV":      (infer_gen,   "Generalization.bin (cross-domain)"),
    }

    combined_per_seq = {}
    combined_summary = {}
    for ds in args.datasets:
        infer_fn, note = weight_map[ds]
        print(f"\n########## HVI-CIDNet @ {ds}  ({note}) ##########", flush=True)
        r = evaluate_method_v2(
            method_name=f"HVI-CIDNet[{ds}]",
            infer_fn=infer_fn,
            datasets=[ds],
            out_path=f"/tmp/hvi_v2_{ds}.json",
            notes=note,
            limit_seqs=args.limit,
        )
        combined_per_seq.update(r["per_seq"])
        combined_summary[ds] = r["summary"][ds]

    out = {
        "method": "HVI-CIDNet",
        "notes": ("LOL splits use matched official weights (wperc/bestPSNR/wperc). "
                  "SDSD_indoor, SDSD_outdoor, BVI_RLV use Generalization.bin. "
                  "gated=False, gated2=False (no GT-mean match). v2 adds DISTS+CLIPIQA."),
        "protocol": "video: center T=16, 512x512 center-crop; images: full-image. PSNR uint8 RGB, LPIPS+DISTS+CLIPIQA pyiqa.",
        "per_seq": combined_per_seq,
        "summary": combined_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")

    print("\n===== HVI-CIDNet v2 summary =====")
    for k, v in combined_summary.items():
        print(f"  {k:14s} PSNR={v['psnr_rgb']:.3f}  LPIPS={v['lpips']:.4f}  DISTS={v['dists']:.4f}  CLIPIQA={v['clipiqa']:.4f}  (n={v['n_seqs']})")


if __name__ == "__main__":
    main()
