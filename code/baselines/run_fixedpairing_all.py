"""Re-evaluate all baselines on SDSD-indoor with CORRECT (sorted-index) LQ/GT pairing.

eval_lib.iter_sdsd_indoor_stcd pairs LQ/GT by file basename, which misaligns pair50/52/58/60/68/70
(their input/ and GT/ filenames are offset; content is aligned by sorted index). This script does
not touch eval_lib; it adds an index-paired STCD-12 iterator that uses exactly the pairing and
slicing of SDSDDataset / seer_eval_unified.iter_sdsd_indoor_seer (sorted frame list, first
min(len) frames paired by position, center 16 frames, 512x512 center crop).

Splits:
  L0_SDSD-indoor          iter_sdsd_indoor_seer (Seer's 15-seq test split)
  STCD-12idx_SDSD-indoor  iter_sdsd_indoor_stcd_idx (STCD-12, index-paired)

Methods / weights / inference: identical to run_temporal_all.build_methods (each run_*_v2.py's
own loader + infer_fn; STCD via run_stcd_unified). Per clip: PSNR/LPIPS/DISTS/CLIPIQA via
seer_eval_unified.score_frames (per frame -> per seq mean -> mean over seqs) and temporal metrics
(tools/temporal_metrics.py, GT RAFT flows computed once per clip, shared by all rows).

Outputs:
  seer_iclr_evals/baselines_L0_fixedpairing_eval.json   per-frame metrics, per_seq + summary + clean6
  seer_iclr_evals/baselines_temporal_unified_v2.json    old baselines_temporal_unified.json content
                                                        + the new STCD-12idx split (old file kept)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

BASE = "/temp/zihaohe/workspace/baselines"
sys.path.insert(0, BASE)
import run_temporal_all as RT  # noqa: E402  (also sets up STCD / vjepa_llve / tools paths)
from common.eval_lib import DATA_ROOTS, STCD_INDOOR_12, _load_img_any, center_crop, lpips_batch  # noqa: E402
from seer_eval_unified import iter_sdsd_indoor_seer, score_frames  # noqa: E402
from data.common import list_images  # noqa: E402  (same frame lister as SDSDDataset)
from temporal_metrics import safe_gt_flows, temporal_metrics  # noqa: E402

EVDIR = "/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals"
OUT_FRAME = os.path.join(EVDIR, "baselines_L0_fixedpairing_eval.json")
OLD_TEMPORAL = os.path.join(EVDIR, "baselines_temporal_unified.json")
OUT_TEMPORAL = os.path.join(EVDIR, "baselines_temporal_unified_v2.json")
CLEAN6 = RT.CLEAN6
FKEYS = ("psnr_rgb", "lpips", "dists", "clipiqa")
TKEYS = ("tlp", "mabd", "flicker", "warp_err")


def iter_sdsd_indoor_stcd_idx(n_frames: int = 16, crop: int = 512):
    """STCD-12 with SDSDDataset-style sorted-index pairing (same slicing as iter_sdsd_indoor_seer)."""
    root = os.path.join(DATA_ROOTS["SDSD_indoor"], "indoor_static_np")
    for seq in STCD_INDOOR_12:
        lq_dir = os.path.join(root, "input", seq)
        gt_dir = os.path.join(root, "GT", seq)
        if not (os.path.isdir(lq_dir) and os.path.isdir(gt_dir)):
            print(f"[SKIP] {seq}: dir missing")
            continue
        lq_paths, gt_paths = list_images(lq_dir), list_images(gt_dir)
        n = min(len(lq_paths), len(gt_paths))
        lq_paths, gt_paths = lq_paths[:n], gt_paths[:n]
        start = max(0, (n - n_frames) // 2)
        end = min(n, start + n_frames)
        lqs = np.stack([center_crop(_load_img_any(p), crop) for p in lq_paths[start:end]])
        gts = np.stack([center_crop(_load_img_any(p), crop) for p in gt_paths[start:end]])
        yield seq, lqs, gts


SPLITS = {
    "L0_SDSD-indoor": lambda: iter_sdsd_indoor_seer(n_frames=16, crop=512),
    "STCD-12idx_SDSD-indoor": lambda: iter_sdsd_indoor_stcd_idx(n_frames=16, crop=512),
}


@torch.no_grad()
def predict(spec, lq_u8, lq, dev):
    if spec["kind"] == "clip":
        return spec["indoor"](lq)
    outs = []
    for t in range(lq_u8.shape[0]):  # identical tensor prep to eval_lib_v2.evaluate_method_v2
        lq_t = torch.from_numpy(lq_u8[t]).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
        outs.append(spec["indoor"](lq_t).clamp(0, 1).float())
    return torch.cat(outs, 0)


def summ(d, names=None, keys=FKEYS + TKEYS):
    names = [n for n in (names if names is not None else d.keys()) if n in d]
    ks = [k for k in keys if names and all(k in d[n] for n in names)]
    s = {k: float(np.mean([d[n][k] for n in names])) for k in ks}
    s["n"] = len(names)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_frame", default=OUT_FRAME)
    ap.add_argument("--out_temporal", default=OUT_TEMPORAL)
    args = ap.parse_args()

    dev = torch.device("cuda")
    methods = RT.build_methods(dev)
    old_t = json.load(open(OLD_TEMPORAL))
    old_v2 = {m: json.load(open(p)) for m, p in RT.OLD_JSON.items()}
    rows = ["GT", "LQ"] + list(methods)
    per_seq = {s: {r: {} for r in rows} for s in SPLITS}
    t_all = time.time()
    for split, it_fn in SPLITS.items():
        t0 = time.time()
        for i, (name, lq_u8, gt_u8) in enumerate(it_fn()):
            if args.limit is not None and i >= args.limit:
                break
            lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            flows = safe_gt_flows(gt)
            per_seq[split]["GT"][name] = temporal_metrics(gt, lpips_batch, flows)
            r = score_frames(lq, gt)
            r.update(temporal_metrics(lq, lpips_batch, flows))
            per_seq[split]["LQ"][name] = r
            msg = []
            for m, spec in methods.items():
                pred = predict(spec, lq_u8, lq, dev)
                r = score_frames(pred, gt)
                r.update(temporal_metrics(pred, lpips_batch, flows))
                per_seq[split][m][name] = r
                msg.append(f"{m}:{r['psnr_rgb']:.2f}")
                del pred
            print(f"  {split:22s} {name:10s} " + " ".join(msg), flush=True)
        print(f"== {split} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- consistency checks ----
    cons = {}
    for m in methods:
        c = {}
        # L0 psnr/lpips vs baselines_temporal_unified.json (same clips, same inference)
        o = old_t["per_seq"].get("L0_SDSD-indoor", {}).get(m, {})
        d = per_seq["L0_SDSD-indoor"][m]
        com = [s for s in d if s in o]
        if com:
            c["L0_vs_temporal_v1"] = {
                "n": len(com),
                "max_abs_dpsnr": max(abs(d[s]["psnr_rgb"] - o[s]["psnr_rgb"]) for s in com),
                "max_abs_dlpips": max(abs(d[s]["lpips"] - o[s]["lpips"]) for s in com),
                "max_abs_dwarp": max([abs(d[s]["warp_err"] - o[s]["warp_err"]) for s in com
                                      if "warp_err" in d[s] and "warp_err" in o[s]], default=None)}
        # STCD-12idx: pair13..33 have identical basenames -> must equal the old basename-paired eval
        d = per_seq["STCD-12idx_SDSD-indoor"][m]
        same = [s for s in STCD_INDOOR_12 if s not in CLEAN6 and s in d]
        ov = old_v2[m]
        get = (lambda s: ov["per_seq"]["STCD-12_SDSD-indoor"].get(s)) if m == "STCD" else \
              (lambda s: ov["per_seq"].get(f"SDSD_indoor__{s}"))
        diffs = {k: max([abs(d[s][k] - get(s)[k]) for s in same if get(s)], default=None) for k in FKEYS}
        c["STCD12idx_same6_vs_old_eval_v2"] = {"n": len(same), **{f"max_abs_d{k}": v for k, v in diffs.items()}}
        # clean-6: STCD-12idx and L0 must be the same clips
        a, b = per_seq["STCD-12idx_SDSD-indoor"][m], per_seq["L0_SDSD-indoor"][m]
        c["clean6_STCD12idx_vs_L0_max_abs_dpsnr"] = max([abs(a[s]["psnr_rgb"] - b[s]["psnr_rgb"])
                                                         for s in CLEAN6 if s in a and s in b], default=None)
        cons[m] = c

    # ---- frame-metric JSON ----
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for rname in ["LQ"] + list(methods):
            d = per_seq[split][rname]
            summary[split][rname] = summ(d, keys=FKEYS)
            summary[split][rname]["clean6"] = summ(d, CLEAN6, keys=FKEYS)
    frame_out = {
        "protocol": ("SDSD-indoor, LQ/GT paired by sorted index (SDSDDataset convention), center T=16, "
                     "512x512 center crop; PSNR uint8 RGB, LPIPS/DISTS/CLIPIQA pyiqa via score_frames; "
                     "per-frame -> per-seq mean -> mean over seqs"),
        "splits": {"L0_SDSD-indoor": "seer_eval_unified.iter_sdsd_indoor_seer (Seer 15-seq test split)",
                   "STCD-12idx_SDSD-indoor": "run_fixedpairing_all.iter_sdsd_indoor_stcd_idx (STCD-12, index-paired)"},
        "methods": {m: v["note"] for m, v in methods.items()},
        "clean6_definition": CLEAN6,
        "consistency": cons,
        "per_seq": {s: {r: {n: {k: v[k] for k in FKEYS + ("n_frames",) if k in v} for n, v in d.items()}
                        for r, d in per_seq[s].items() if r != "GT"} for s in SPLITS},
        "summary": summary,
        "wall_s": round(time.time() - t_all, 1),
    }
    with open(args.out_frame, "w") as f:
        json.dump(frame_out, f, indent=1)
    print(f"[SAVE] {args.out_frame}", flush=True)

    # ---- temporal v2 = v1 + STCD-12idx split ----
    t2 = copy.deepcopy(old_t)
    sp = "STCD-12idx_SDSD-indoor"
    t2["per_seq"][sp] = per_seq[sp]
    t2["summary"][sp] = {}
    for rname, d in per_seq[sp].items():
        t2["summary"][sp][rname] = summ(d)
        t2["summary"][sp][rname]["clean6"] = summ(d, CLEAN6)
    t2["notes"] = list(t2.get("notes", [])) + [
        "v2: added STCD-12idx_SDSD-indoor = STCD-12 with sorted-index LQ/GT pairing "
        "(run_fixedpairing_all.iter_sdsd_indoor_stcd_idx). Use it instead of STCD-12_SDSD-indoor "
        "(basename pairing, GT misaligned on pair50/52/58/60/68/70). Rows in the new split also carry "
        "DISTS/CLIPIQA.",
    ]
    t2["consistency_fixedpairing"] = cons
    with open(args.out_temporal, "w") as f:
        json.dump(t2, f, indent=1)
    print(f"[SAVE] {args.out_temporal}", flush=True)
    print(json.dumps(cons, indent=1))
    for split in SPLITS:
        for rname, s in summary[split].items():
            c = s["clean6"]
            print(f"{split:22s} {rname:14s} " + " ".join(f"{k}={s[k]:.4f}" for k in FKEYS if k in s)
                  + f" n={s['n']} | clean6 " + " ".join(f"{k}={c[k]:.4f}" for k in FKEYS if k in c))
    for rname, s in t2["summary"][sp].items():
        print(f"T {sp} {rname:14s} " + " ".join(f"{k}={s[k]:.4f}" for k in TKEYS if k in s))


if __name__ == "__main__":
    main()
