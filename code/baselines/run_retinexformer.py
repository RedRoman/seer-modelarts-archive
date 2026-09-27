"""Retinexformer eval on all 6 datasets.

Weights available locally:
  - SDSD_indoor.pth  (use for SDSD_indoor)
  - SDSD_outdoor.pth (use for SDSD_outdoor)

LOL-v1 / LOLv2-Real / LOLv2-Syn weights are on Google Drive only (blocked from
this host). We fall back to using SDSD_indoor.pth cross-domain and note the mismatch.
For BVI-RLV (no official Retinexformer weight), we also use SDSD_indoor.pth
cross-domain as a proxy.
"""
from __future__ import annotations
import os, sys, argparse
import torch

sys.path.insert(0, "/home/ma-user/workspace/zihaohe/baselines")
sys.path.insert(0, "/cache/zihaohe/ext_methods/Retinexformer/basicsr/models/archs")

from common.eval_lib import evaluate_method
from RetinexFormer_arch import RetinexFormer


CKPT_DIR = "/cache/zihaohe/baseline_ckpts/Retinexformer"

# Retinexformer variants per training set (see repo Options/*.yml):
#   LOL_v1 / LOL_v2_real / LOL_v2_synthetic: n_feat=40, stage=1, [1,2,2]
#   SDSD_indoor / SDSD_outdoor:              n_feat=40, stage=1, [1,2,2]
NET_KWARGS = dict(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])


def load_model(weights_path: str, device):
    m = RetinexFormer(**NET_KWARGS)
    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "params" in ck:
        sd = ck["params"]
    elif isinstance(ck, dict) and "params_ema" in ck:
        sd = ck["params_ema"]
    else:
        sd = ck
    m.load_state_dict(sd, strict=True)
    m = m.to(device).eval()
    return m


def make_infer_fn(model):
    @torch.no_grad()
    def infer(lq: torch.Tensor) -> torch.Tensor:
        # RetinexFormer expects even-divisible spatial dims; SDSD is 960x540 → 512 center crop is fine
        return model(lq)
    return infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["SDSD_indoor", "SDSD_outdoor",
                                                       "BVI_RLV", "LOL_v1",
                                                       "LOLv2_Syn", "LOLv2_Real"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/cache/zihaohe/baseline_ckpts/Retinexformer_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda")

    # We reuse the SDSD_indoor.pth for all "no-weight" datasets and note it.
    indoor = load_model(os.path.join(CKPT_DIR, "SDSD_indoor.pth"), device)
    outdoor = load_model(os.path.join(CKPT_DIR, "SDSD_outdoor.pth"), device)

    infer_indoor = make_infer_fn(indoor)
    infer_outdoor = make_infer_fn(outdoor)

    # Split by dataset → weight
    weight_map = {
        "SDSD_indoor":  (infer_indoor, "SDSD_indoor.pth (matched)"),
        "SDSD_outdoor": (infer_outdoor, "SDSD_outdoor.pth (matched)"),
        # LOL / BVI have no official weight reachable — use SDSD_indoor.pth cross-domain
        "LOL_v1":       (infer_indoor, "SDSD_indoor.pth (cross-domain; LOL weights unreachable)"),
        "LOLv2_Syn":    (infer_indoor, "SDSD_indoor.pth (cross-domain; LOL weights unreachable)"),
        "LOLv2_Real":   (infer_indoor, "SDSD_indoor.pth (cross-domain; LOL weights unreachable)"),
        "BVI_RLV":      (infer_indoor, "SDSD_indoor.pth (cross-domain; no official BVI weight)"),
    }

    # We call evaluate_method once per dataset so we can vary the infer_fn.
    from common.eval_lib import evaluate_method
    import json
    combined_per_seq = {}
    combined_summary = {}
    for ds in args.datasets:
        infer_fn, note = weight_map[ds]
        print(f"\n########## Retinexformer @ {ds}  ({note}) ##########", flush=True)
        r = evaluate_method(
            method_name=f"Retinexformer[{ds}]",
            infer_fn=infer_fn,
            datasets=[ds],
            out_path=f"/tmp/rf_{ds}.json",
            notes=note,
            limit_seqs=args.limit,
        )
        combined_per_seq.update(r["per_seq"])
        combined_summary[ds] = r["summary"][ds]

    out = {
        "method": "Retinexformer",
        "notes": "SDSD_indoor.pth & SDSD_outdoor.pth from official repo; LOL/BVI use SDSD_indoor.pth cross-domain (LOL weights on GDrive, unreachable).",
        "protocol": "video: center T=16, 512x512 center-crop; images: full-image. PSNR uint8 RGB, LPIPS pyiqa alex.",
        "per_seq": combined_per_seq,
        "summary": combined_summary,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")

    # summary print
    print("\n===== Retinexformer summary =====")
    for k, v in combined_summary.items():
        print(f"  {k:14s} PSNR={v['psnr_rgb']:.3f}  LPIPS={v['lpips']:.4f}  (n={v['n_seqs']})")


if __name__ == "__main__":
    main()
