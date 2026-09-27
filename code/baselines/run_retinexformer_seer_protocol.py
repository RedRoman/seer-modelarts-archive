"""Eval the retrained Retinexformer-under-Seer-protocol checkpoint on all 6 OOD splits.
Same protocol as run_retinexformer_v2.py: video T=16 512x512 center-crop, images full-res.
PSNR uint8 RGB + LPIPS-alex + DISTS + CLIPIQA."""
from __future__ import annotations
import os, sys, argparse, json
import torch

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")

from common.eval_lib_v2 import evaluate_method_v2
from RetinexFormer_arch import RetinexFormer


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
    # basicsr sometimes saves with 'module.' prefix under distributed
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3]: print(f"  missing[:3]={missing[:3]}")
    if unexpected[:3]: print(f"  unexpected[:3]={unexpected[:3]}")
    return m.to(device).eval()


def make_infer_fn(model):
    @torch.no_grad()
    def infer(lq):
        return model(lq)
    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to the retrained checkpoint (.pth from basicsr)")
    ap.add_argument("--datasets", nargs="+", default=["SDSD_indoor", "SDSD_outdoor",
                                                       "BVI_RLV", "LOL_v1",
                                                       "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/home/ma-user/workspace/zihaohe/checkpoint/runs/Retinexformer_SeerProto_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(args.ckpt, device)
    infer_fn = make_infer_fn(model)

    note = f"Retinexformer retrained on SDSD-indoor+outdoor union under Seer protocol; ckpt={args.ckpt}"

    combined_per_seq = {}
    combined_summary = {}
    for ds in args.datasets:
        print(f"\n########## Retinexformer[SeerProto] @ {ds} ##########", flush=True)
        r = evaluate_method_v2(
            method_name=f"Retinexformer_SeerProto[{ds}]",
            infer_fn=infer_fn,
            datasets=[ds],
            out_path=f"/tmp/rf_seer_v2_{ds}.json",
            notes=note,
            limit_seqs=args.limit,
        )
        combined_per_seq.update(r["per_seq"])
        combined_summary[ds] = r["summary"][ds]

    out = {
        "method": "Retinexformer_SeerProtocol",
        "notes": note,
        "protocol": "video: center T=16, 512x512 center-crop; images: full-image. PSNR uint8 RGB, LPIPS+DISTS+CLIPIQA pyiqa.",
        "per_seq": combined_per_seq,
        "summary": combined_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")

    print("\n===== Retinexformer_SeerProtocol summary =====")
    for k, v in combined_summary.items():
        print(f"  {k:14s} PSNR={v['psnr_rgb']:.3f}  LPIPS={v['lpips']:.4f}  DISTS={v['dists']:.4f}  CLIPIQA={v['clipiqa']:.4f}  (n={v['n_seqs']})")


if __name__ == "__main__":
    main()
