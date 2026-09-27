"""Eval a Retinexformer checkpoint on Seer's own SDSD-indoor test split (15 seqs).

run_retinexformer_seer_protocol.py scores "SDSD_indoor" on STCD-12, 6 of whose
sequences (pair13/15/21/23/31/33) are in Seer's TRAIN split. This wrapper registers
an extra dataset "SDSD_indoor_seer" (iter_sdsd_indoor_seer from
vjepa_llve/tools/seer_eval_unified.py: SDSDDataset 80/20 test split, center T=16,
512x512 center crop) and scores it with the same evaluate_method_v2 harness.
The JSON additionally reports the mean over the 6 seqs that are held out for both
Seer and STCD-12 (pair50/52/58/60/68/70).
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
import torch

BASE = "/temp/zihaohe/workspace/baselines"
VJ_TOOLS = "/temp/zihaohe/workspace/vjepa_llve/tools"
for p in (BASE, VJ_TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

import common.eval_lib as EL  # noqa: E402
from seer_eval_unified import iter_sdsd_indoor_seer  # noqa: E402

EL.DATASET_ITERS["SDSD_indoor_seer"] = ("video", iter_sdsd_indoor_seer)

from common.eval_lib_v2 import evaluate_method_v2, DATASET_ITERS  # noqa: E402
assert DATASET_ITERS is EL.DATASET_ITERS and "SDSD_indoor_seer" in DATASET_ITERS
from run_retinexformer_seer_protocol import load_model, make_infer_fn  # noqa: E402

DS = "SDSD_indoor_seer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str,
                    default="/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals/"
                            "Retinexformer_union_seersplit_L0seer_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(args.ckpt, device)
    note = (f"Retinexformer (SDSD union, Seer split) on Seer's SDSD-indoor test split "
            f"(15 seqs); ckpt={args.ckpt}")
    r = evaluate_method_v2(
        method_name=f"Retinexformer_SeerSplit[{DS}]",
        infer_fn=make_infer_fn(model),
        datasets=[DS],
        out_path=f"/tmp/rf_seersplit_v2_{DS}.json",
        notes=note,
        limit_seqs=args.limit,
    )
    stcd = set(EL.STCD_INDOOR_12)
    both = {k: v for k, v in r["per_seq"].items() if k.split("__", 1)[1] in stcd}
    sub = {m: float(np.mean([v[m] for v in both.values()])) if both else float("nan")
           for m in ("psnr_rgb", "lpips", "dists", "clipiqa")}
    sub["n_seqs"] = len(both)
    sub["seqs"] = sorted(k.split("__", 1)[1] for k in both)
    out = {
        "method": "Retinexformer_SeerSplit",
        "notes": note,
        "protocol": r["protocol"],
        "per_seq": r["per_seq"],
        "summary": r["summary"],
        "subset_summary": {f"{DS}&STCD12": sub},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {args.out}")
    s = r["summary"][DS]
    print(f"  {DS:18s} PSNR={s['psnr_rgb']:.3f} LPIPS={s['lpips']:.4f} DISTS={s['dists']:.4f} "
          f"CLIPIQA={s['clipiqa']:.4f} (n={s['n_seqs']})")
    print(f"  {DS+'&STCD12':18s} PSNR={sub['psnr_rgb']:.3f} LPIPS={sub['lpips']:.4f} "
          f"DISTS={sub['dists']:.4f} CLIPIQA={sub['clipiqa']:.4f} (n={sub['n_seqs']})")


if __name__ == "__main__":
    main()
