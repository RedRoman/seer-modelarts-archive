"""Temporal metrics for the two Retinexformer checkpoints retrained on Seer's split.

Checkpoints (same files the *_seersplit_eval_v2.json notes point to on host 30555; the /temp mirrors
below are byte-identical, md5 1ec353cf... union / 3ea7d5f7... indoor):
  Retinexformer-union-seersplit  : runs/retinexformer_seersplit_mirror/models/net_g_24000.pth
  Retinexformer-indoor-seersplit : runs/retinexformer_indoor_seersplit_mirror/models/net_g_24000.pth
Model construction / inference: run_retinexformer_seer_protocol.load_model + make_infer_fn (one
checkpoint for all splits), imported, not reimplemented.
Machinery: run_temporal_all (per-frame inference exactly as eval_lib_v2, GT RAFT flows via
safe_gt_flows once per clip, temporal_metrics(pred, lpips_batch, flows)); STCD-12idx iterator from
run_fixedpairing_all. Per-frame PSNR/LPIPS/DISTS/CLIPIQA via score_frames.
Layout: per_seq[split_key][model_label][clip]{metrics}; GT/LQ rows under reference_rows.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/temp/zihaohe/workspace/baselines")
import run_temporal_all as RT  # noqa: E402
import run_fixedpairing_all as RFP  # noqa: E402
from common.eval_lib import iter_sdsd_outdoor, iter_bvi, lpips_batch  # noqa: E402
from seer_eval_unified import iter_sdsd_indoor_seer, score_frames  # noqa: E402
from temporal_metrics import safe_gt_flows, temporal_metrics  # noqa: E402

RUNS = "/temp/zihaohe/cache/checkpoint/runs"
EV = os.path.join(RUNS, "seer_iclr_evals")
OUT = os.path.join(EV, "Retinexformer_seersplit_temporal.json")
MODELS = {
    "Retinexformer-union-seersplit": {
        "ckpt": os.path.join(RUNS, "retinexformer_seersplit_mirror/models/net_g_24000.pth"),
        "ckpt_in_eval_notes": "/cache/zihaohe/runs/seer_iclr/rf_seersplit/Retinexformer/experiments/"
                              "RetinexFormer_SDSD_union_seersplit/models/net_g_24000.pth (host 30555)",
        "md5": "1ec353cfc702f1f1254a45c89d4805f1",
        "old": {"L0": "Retinexformer_union_seersplit_L0seer_eval_v2.json",
                "other": "Retinexformer_union_seersplit_eval_v2.json"}},
    "Retinexformer-indoor-seersplit": {
        "ckpt": os.path.join(RUNS, "retinexformer_indoor_seersplit_mirror/models/net_g_24000.pth"),
        "ckpt_in_eval_notes": "/cache/zihaohe/runs/seer_iclr/rf_indoor_seersplit/Retinexformer/experiments/"
                              "RetinexFormer_SDSD_indoor_seersplit/models/net_g_24000.pth (host 30555)",
        "md5": "3ea7d5f7350d19378a2b8a62f8bf4758",
        "old": {"L0": "Retinexformer_indoor_seersplit_L0seer_eval_v2.json",
                "other": "Retinexformer_indoor_seersplit_eval_v2.json"}},
}
SPLITS = {
    "L0_SDSD-indoor": (lambda: iter_sdsd_indoor_seer(n_frames=16, crop=512), "L0", "SDSD_indoor_seer__"),
    "STCD-12idx_SDSD-indoor": (lambda: RFP.iter_sdsd_indoor_stcd_idx(n_frames=16, crop=512), "other", "SDSD_indoor__"),
    "L1_SDSD-outdoor": (lambda: iter_sdsd_outdoor(n_frames=16, crop=512), "other", "SDSD_outdoor__"),
    "L2_BVI-RLV": (lambda: iter_bvi(n_frames=16, crop=512), "other", "BVI_RLV__"),
}
CLEAN6 = RT.CLEAN6
FK = ("psnr_rgb", "lpips", "dists", "clipiqa")
TK = ("tlp", "mabd", "flicker", "warp_err")


def summ(d, names=None):
    names = [n for n in (names if names is not None else d) if n in d]
    ks = [k for k in FK + TK if names and all(k in d[n] for n in names)]
    s = {k: float(np.mean([d[n][k] for n in names])) for k in ks}
    s["n"] = len(names)
    return s


def main():
    dev = torch.device("cuda")
    infer = {}
    for label, spec in MODELS.items():
        infer[label] = RT.load_isolated(
            "run_retinexformer_seer_protocol.py",
            lambda mod, p=spec["ckpt"]: mod.make_infer_fn(mod.load_model(p, dev)))
        spec["old_json"] = {k: json.load(open(os.path.join(EV, v))) for k, v in spec["old"].items()}

    per_seq = {s: {m: {} for m in MODELS} for s in SPLITS}
    refs = {s: {"GT": {}, "LQ": {}} for s in SPLITS}
    cons = {m: {} for m in MODELS}
    t_all = time.time()
    for split, (it_fn, oldkey, prefix) in SPLITS.items():
        t0 = time.time()
        for name, lq_u8, gt_u8 in it_fn():
            lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            flows = safe_gt_flows(gt)
            refs[split]["GT"][name] = temporal_metrics(gt, lpips_batch, flows)
            r = score_frames(lq, gt); r.update(temporal_metrics(lq, lpips_batch, flows))
            refs[split]["LQ"][name] = r
            msg = []
            for label, spec in MODELS.items():
                pred, _, _ = RT.run_frames(infer[label], lq_u8, gt_u8, dev)
                r = score_frames(pred, gt)
                r.update(temporal_metrics(pred, lpips_batch, flows))
                o = spec["old_json"][oldkey]["per_seq"].get(prefix + name)
                if o is not None:
                    r["old_psnr_rgb"], r["old_lpips"] = o["psnr_rgb"], o["lpips"]
                    cons[label].setdefault(split, []).append(
                        (abs(r["psnr_rgb"] - o["psnr_rgb"]), abs(r["lpips"] - o["lpips"])))
                per_seq[split][label][name] = r
                msg.append(f"{label.split('-')[1]}:{r['psnr_rgb']:.2f}/{r['warp_err']:.4f}")
                del pred
            print(f"  {split:22s} {name:22s} " + " ".join(msg), flush=True)
        print(f"== {split} done ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for label, d in list(per_seq[split].items()) + list(refs[split].items()):
            summary[split][label] = summ(d)
            if split in ("L0_SDSD-indoor", "STCD-12idx_SDSD-indoor"):
                summary[split][label]["clean6"] = summ(d, CLEAN6)
    consistency = {m: {sp: {"n": len(v), "max_abs_dpsnr": float(max(a for a, _ in v)),
                            "max_abs_dlpips": float(max(b for _, b in v))}
                       for sp, v in dd.items() if v} for m, dd in cons.items()}
    out = {
        "protocol": ("unified clips (center T=16, 512x512 center crop); L0 = iter_sdsd_indoor_seer, "
                     "STCD-12idx = run_fixedpairing_all.iter_sdsd_indoor_stcd_idx (sorted-index pairing), "
                     "L1 = iter_sdsd_outdoor, L2 = iter_bvi; per-frame inference as eval_lib_v2; temporal "
                     "metrics from vjepa_llve/tools/temporal_metrics.py with GT RAFT-small flows (once per clip)"),
        "models": {m: {k: v for k, v in s.items() if k != "old_json"} for m, s in MODELS.items()},
        "consistency_note": ("PSNR/LPIPS checked against *_seersplit_L0seer_eval_v2.json (L0) and "
                             "*_seersplit_eval_v2.json (L1, L2, and pair13..33 of STCD-12idx whose basename "
                             "and index pairing coincide; the old SDSD_indoor entries for pair50..70 are "
                             "basename-paired and are not compared)"),
        "clean6_definition": CLEAN6,
        "consistency": consistency,
        "per_seq": per_seq,
        "reference_rows": refs,
        "summary": summary,
        "wall_s": round(time.time() - t_all, 1),
    }
    # restrict STCD-12idx consistency to the 6 seqs whose pairing is unchanged
    for m in MODELS:
        d = per_seq["STCD-12idx_SDSD-indoor"][m]
        for s in CLEAN6:
            d[s].pop("old_psnr_rgb", None); d[s].pop("old_lpips", None)
        v = [(abs(d[s]["psnr_rgb"] - d[s]["old_psnr_rgb"]), abs(d[s]["lpips"] - d[s]["old_lpips"]))
             for s in d if "old_psnr_rgb" in d[s]]
        out["consistency"][m]["STCD-12idx_SDSD-indoor"] = {
            "n": len(v), "max_abs_dpsnr": float(max(a for a, _ in v)), "max_abs_dlpips": float(max(b for _, b in v))}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[SAVE] {OUT}", flush=True)
    print(json.dumps(out["consistency"], indent=1))
    for split in SPLITS:
        for label, s in summary[split].items():
            line = f"{split:22s} {label:32s} " + " ".join(f"{k}={s[k]:.4f}" for k in FK + TK if k in s) + f" n={s['n']}"
            if "clean6" in s:
                c = s["clean6"]
                line += " | clean6 " + " ".join(f"{k}={c[k]:.4f}" for k in ("psnr_rgb", "lpips", "tlp", "warp_err") if k in c)
            print(line)


if __name__ == "__main__":
    main()
