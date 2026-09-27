"""Retinexformer eval v2 -- adds DISTS + CLIPIQA on top of PSNR + LPIPS."""
from __future__ import annotations
import os, sys, argparse, json
import torch

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")

from common.eval_lib_v2 import evaluate_method_v2
from RetinexFormer_arch import RetinexFormer


CKPT_DIR = "/cache/zihaohe/baseline_ckpts/Retinexformer"
NET_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])


def load_model(weights_path, device):
    m = RetinexFormer(**NET_KWARGS)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "params" in ck:
        sd = ck["params"]
    elif isinstance(ck, dict) and "params_ema" in ck:
        sd = ck["params_ema"]
    else:
        sd = ck
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def make_infer_fn(model):
    @torch.no_grad()
    def infer(lq):
        return model(lq)
    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["SDSD_indoor", "SDSD_outdoor",
                                                       "BVI_RLV", "LOL_v1",
                                                       "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/cache/zihaohe/baseline_ckpts/Retinexformer_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    indoor = load_model(os.path.join(CKPT_DIR, "SDSD_indoor.pth"), device)
    outdoor = load_model(os.path.join(CKPT_DIR, "SDSD_outdoor.pth"), device)
    infer_indoor = make_infer_fn(indoor)
    infer_outdoor = make_infer_fn(outdoor)

    weight_map = {
        "SDSD_indoor":  (infer_indoor, "SDSD_indoor.pth (matched)"),
        "SDSD_outdoor": (infer_outdoor, "SDSD_outdoor.pth (matched)"),
        "LOL_v1":       (infer_indoor, "SDSD_indoor.pth (cross-domain; LOL weights unreachable)"),
        "LOLv2_Syn":    (infer_indoor, "SDSD_indoor.pth (cross-domain; LOL weights unreachable)"),
        "LOLv2_Real":   (infer_indoor, "SDSD_indoor.pth (cross-domain; LOL weights unreachable)"),
        "BVI_RLV":      (infer_indoor, "SDSD_indoor.pth (cross-domain; no official BVI weight)"),
    }

    combined_per_seq = {}
    combined_summary = {}
    for ds in args.datasets:
        infer_fn, note = weight_map[ds]
        print(f"\n########## Retinexformer @ {ds}  ({note}) ##########", flush=True)
        r = evaluate_method_v2(
            method_name=f"Retinexformer[{ds}]",
            infer_fn=infer_fn,
            datasets=[ds],
            out_path=f"/tmp/rf_v2_{ds}.json",
            notes=note,
            limit_seqs=args.limit,
        )
        combined_per_seq.update(r["per_seq"])
        combined_summary[ds] = r["summary"][ds]

    out = {
        "method": "Retinexformer",
        "notes": "SDSD_indoor.pth & SDSD_outdoor.pth from official repo; LOL/BVI use SDSD_indoor.pth cross-domain (LOL weights on GDrive, unreachable). v2 adds DISTS+CLIPIQA.",
        "protocol": "video: center T=16, 512x512 center-crop; images: full-image. PSNR uint8 RGB, LPIPS+DISTS+CLIPIQA pyiqa.",
        "per_seq": combined_per_seq,
        "summary": combined_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")

    print("\n===== Retinexformer v2 summary =====")
    for k, v in combined_summary.items():
        print(f"  {k:14s} PSNR={v['psnr_rgb']:.3f}  LPIPS={v['lpips']:.4f}  DISTS={v['dists']:.4f}  CLIPIQA={v['clipiqa']:.4f}  (n={v['n_seqs']})")


if __name__ == "__main__":
    main()
