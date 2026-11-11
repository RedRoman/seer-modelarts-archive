"""tools/v10_audit.py — v10 fixed-split + sentinel evaluation.

Spec: V10_DESIGN_SPEC.md §1, §5 (CH6), §7 (M7).
Scope per CJ4: 4 fixed splits at locked tile=512 / overlap=128 RGB-PSNR uint8.
              + 5-seq sentinel panel for fast iteration.
              + v9.0b_mix sentinel baseline (CJ3) recorded in same protocol.

Output:
  /cache/zihaohe/checkpoint/v10_audit/<run_tag>/
    splits/{llve_stcd_12,outdoor_25,original_indoor_14,bz_indoor_15}.csv
    sentinel.csv
    summary.md

Usage:
  # v9.0b_mix baseline (run once before v10 training launches)
  python tools/v10_audit.py --model v9 \
      --ckpt /cache/zihaohe/checkpoint/runs/v9_0b/ckpts/best.pt \
      --cfg  /cache/zihaohe/checkpoint/runs/v9_0b/config.yaml \
      --tag  v9_0b_mix_baseline

  # v10 evaluation
  python tools/v10_audit.py --model v10 \
      --ckpt /cache/zihaohe/checkpoint/runs/v10_first/ckpts/best.pt \
      --cfg  /cache/zihaohe/checkpoint/runs/v10_first/config.yaml \
      --tag  v10_first

  # sentinel-only (fast, every 250 iter during training):
  python tools/v10_audit.py --sentinel-only --model v10 ...
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import SDSDDataset, _detect_pair_dirs
from data.common import list_sequences, list_images
from tools.audit_llvestcd import build_v9_model, load_seq_full
from tools.theta_llvestcd_eval import tiled_inference, to_u8, mean_psnr


# ---- Locked split definitions (V10_DESIGN_SPEC.md §1) ---------------------
LLVE_STCD_12 = [
    "pair13", "pair15", "pair21", "pair23", "pair31", "pair33",
    "pair50", "pair52", "pair58", "pair60", "pair68", "pair70",
]

# v9.4d / v9.0b_mix audit_v9 reference 14-seq indoor split.
ORIGINAL_INDOOR_14 = [
    "pair34_2", "pair3_2", "pair4",   "pair4_2", "pair5",   "pair5_2",
    "pair6",    "pair6_2", "pair7",   "pair7_2", "pair8",   "pair8_2",
    "pair9",    "pair9_2",
]

# Sentinel panel (CH6): hard indoor + outdoor + legacy guard.
SENTINEL_5 = [
    ("indoor", "pair68"),
    ("indoor", "pair31"),
    ("outdoor", "MVI_1030"),
    ("outdoor", "MVI_1019"),
    ("indoor", "pair6"),     # legacy easy guard. Originally pair3_2 but only 14 basename-matched frames < 16.
]

INDOOR_ROOT  = "/home/ma-user/workspace/zihaohe/datasets/SDSD_indoor"
OUTDOOR_ROOT = "/home/ma-user/workspace/zihaohe/datasets/SDSD_outdoor"


# ---- Model builders -------------------------------------------------------
def build_model(model_kind: str, cfg_path: str, ckpt_path: str, device: torch.device,
                allow_v9_ckpt: bool = False):
    """Build either v9 (reference) or v10 model and load checkpoint."""
    if model_kind == "v9":
        return build_v9_model(cfg_path, ckpt_path, device)
    if model_kind == "v10":
        from models.v10 import VJepaLLVE_v10
        with open(cfg_path) as f:
            cfg_full = yaml.safe_load(f)
        cfg = cfg_full.get("model", cfg_full)
        model = VJepaLLVE_v10(cfg).to(device).eval()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # CL2 hardening: prefer "model" key (v10 ckpts), then "state_dict",
        # then v9-style "pixel_branch" (only with --allow-v9-ckpt).
        loaded_via = None
        if isinstance(state, dict) and "model" in state:
            payload = state["model"]; loaded_via = "model"
        elif isinstance(state, dict) and "state_dict" in state:
            payload = state["state_dict"]; loaded_via = "state_dict"
        elif isinstance(state, dict) and "pixel_branch" in state:
            if not allow_v9_ckpt:
                raise RuntimeError(
                    f"v9-style ckpt at {ckpt_path}: only pixel_branch is present, "
                    f"vjepa_pos_emb would be at random init. Pass --allow-v9-ckpt "
                    f"if you really want a v9-shaped baseline."
                )
            payload = {f"pixel_branch.{k}": v for k, v in state["pixel_branch"].items()}
            loaded_via = "pixel_branch (v9-style)"
        else:
            payload = state; loaded_via = "raw"
        # CO5/CP5: require vjepa_pos_emb.* keys (unless --allow-v9-ckpt).
        n_pe = sum(1 for k in payload if k.startswith("vjepa_pos_emb."))
        if not allow_v9_ckpt and n_pe == 0:
            raise RuntimeError(
                f"v10 ckpt at {ckpt_path} has 0 vjepa_pos_emb.* keys "
                f"(loaded via {loaded_via!r}). M1 state is missing -> evaluating "
                f"a model with random pos-emb is invalid. Pass --allow-v9-ckpt "
                f"only if you really want a v9-equivalent baseline."
            )
        miss, unexp = model.load_state_dict(payload, strict=False)
        own_n = len(model.state_dict()); loaded_n = own_n - len(miss)
        frac = loaded_n / max(own_n, 1)
        print(f"[v10 model] loaded via '{loaded_via}': "
              f"loaded={loaded_n}/{own_n} ({frac:.1%}) missing={len(miss)} unexpected={len(unexp)} "
              f"vjepa_pos_emb keys in payload={n_pe}")
        if miss: print(f"  missing first 10: {list(miss)[:10]}")
        if unexp: print(f"  unexpected first 10: {list(unexp)[:10]}")
        if frac < 0.80:
            raise RuntimeError(f"too few keys loaded ({frac:.1%}); refusing to eval random model")
        return model
    raise ValueError(f"unknown model kind: {model_kind!r}")


# ---- Sequence index resolver ---------------------------------------------
def index_seqs(root: str, split: str = "test") -> dict:
    """Return {seq_name: (lq_paths, gt_paths)} scanning the entire SDSD subset
    on disk, ignoring train/test split membership. v10 audit must be able to
    eval LLVE-STCD-12 / sentinel seqs that may live in either split.

    CN-A FIX: pair-up LQ and GT by basename intersection. Some SDSD seqs
    (e.g. pair68) have offset frame ranges between input/ and GT/ — taking
    list[:T] would pair LQ[0043] with GT[0062]. theta_llvestcd_eval handled
    this correctly via basename intersection; we now match it.
    """
    # subset='direct' style: root IS the subset root; _detect_pair_dirs scans it.
    lq_dir, gt_dir, _ = _detect_pair_dirs(root, split, "LQ", "GT")
    out = {}
    for seq_dir in list_sequences(lq_dir):
        seq = os.path.basename(seq_dir)
        lq_frames = list_images(os.path.join(lq_dir, seq))
        gt_frames = list_images(os.path.join(gt_dir, seq))
        if not lq_frames or not gt_frames:
            continue
        # CN-A: basename intersection (matches theta_llvestcd_eval.load_seq)
        lq_by_name = {os.path.basename(p): p for p in lq_frames}
        gt_by_name = {os.path.basename(p): p for p in gt_frames}
        common = sorted(set(lq_by_name) & set(gt_by_name))
        if not common:
            print(f"[index_seqs] WARNING: {seq} has 0 basename-matched frames "
                  f"(LQ {len(lq_frames)} vs GT {len(gt_frames)}); skipping.")
            continue
        lq_aligned = [lq_by_name[n] for n in common]
        gt_aligned = [gt_by_name[n] for n in common]
        out[seq] = (lq_aligned, gt_aligned)
    return out


# ---- Single-seq tile=512 PSNR --------------------------------------------
def eval_one_seq(model, lq_paths: List[str], gt_paths: List[str],
                 device: torch.device, n_frames: int = 16) -> Tuple[float, float, int, int, int]:
    lq, gt = load_seq_full(lq_paths, gt_paths, n_frames=n_frames)
    H, W = lq.shape[-2:]
    lq_dev, gt_dev = lq.to(device), gt.to(device)
    pred = tiled_inference(model, lq_dev, tile=512, overlap=128)
    pred_u8, gt_u8, lq_u8 = to_u8(pred), to_u8(gt_dev), to_u8(lq_dev)
    return mean_psnr(pred_u8, gt_u8), mean_psnr(lq_u8, gt_u8), lq.shape[2], H, W


def run_split(model, device, label: str, seq_index: dict,
              seq_names: List[str], out_dir: str) -> Tuple[float, float, list]:
    print(f"\n=== {label} ({len(seq_names)} seqs) ===")
    rows = []
    missing = []
    for s in seq_names:
        if s not in seq_index:
            missing.append(s); continue
        t0 = time.time()
        lq_p, gt_p = seq_index[s]
        try:
            psnr, lq_psnr, T, H, W = eval_one_seq(model, lq_p, gt_p, device)
        except Exception as e:
            print(f"  [{s}] FAILED: {type(e).__name__}: {e}")
            continue
        rows.append({"seq": s, "T": T, "HxW": f"{H}x{W}",
                     "lq_psnr": lq_psnr, "model_psnr": psnr,
                     "delta_vs_lq": psnr - lq_psnr,
                     "time_s": round(time.time() - t0, 1)})
        print(f"  [{s}] LQ={lq_psnr:.2f} model={psnr:.3f} ({rows[-1]['time_s']}s)")
    if missing:
        print(f"  MISSING from dataset: {missing}")
    csv_path = os.path.join(out_dir, f"{label}.csv")
    if rows:
        with open(csv_path, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    mean = float(np.mean([r["model_psnr"] for r in rows])) if rows else float("nan")
    mean_lq = float(np.mean([r["lq_psnr"] for r in rows])) if rows else float("nan")
    print(f"  → MEAN model={mean:.3f}  LQ={mean_lq:.3f}  n={len(rows)}/{len(seq_names)}")
    return mean, mean_lq, rows


def run_sentinel(model, device, indoor_idx: dict, outdoor_idx: dict, out_dir: str
                 ) -> Tuple[float, list]:
    print(f"\n=== sentinel panel ({len(SENTINEL_5)} seqs) ===")
    rows = []
    for kind, s in SENTINEL_5:
        idx = indoor_idx if kind == "indoor" else outdoor_idx
        if s not in idx:
            print(f"  [{kind}/{s}] MISSING"); continue
        lq_p, gt_p = idx[s]
        t0 = time.time()
        try:
            psnr, lq_psnr, T, H, W = eval_one_seq(model, lq_p, gt_p, device)
        except Exception as e:
            print(f"  [{kind}/{s}] FAILED: {type(e).__name__}: {e}")
            continue
        rows.append({"kind": kind, "seq": s, "T": T, "HxW": f"{H}x{W}",
                     "lq_psnr": lq_psnr, "model_psnr": psnr,
                     "delta_vs_lq": psnr - lq_psnr,
                     "time_s": round(time.time() - t0, 1)})
        print(f"  [{kind}/{s}] LQ={lq_psnr:.2f} model={psnr:.3f} ({rows[-1]['time_s']}s)")
    csv_path = os.path.join(out_dir, "sentinel.csv")
    if rows:
        with open(csv_path, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    mean = float(np.mean([r["model_psnr"] for r in rows])) if rows else float("nan")
    # CO3/CP4: write a JSON the trainer can load instead of hardcoding numbers
    import json as _json
    json_path = os.path.join(out_dir, "sentinel_baseline.json")
    payload = {
        "protocol": "tile=512,overlap=128,RGB-PSNR-uint8,full-frame-tiled",
        "sentinel_mean": mean,
        "per_seq": {r["seq"]: r["model_psnr"] for r in rows},
        "n": len(rows),
    }
    with open(json_path, "w") as f: _json.dump(payload, f, indent=2)
    print(f"  → SENTINEL MEAN model={mean:.3f}  n={len(rows)}/{len(SENTINEL_5)}; baseline JSON: {json_path}")
    return mean, rows


def write_summary(out_dir: str, tag: str, model_kind: str, ckpt: str,
                  results: dict) -> None:
    md = os.path.join(out_dir, "summary.md")
    with open(md, "w") as f:
        f.write(f"# v10 audit — {tag}\n\n")
        f.write(f"- model: **{model_kind}**\n")
        f.write(f"- ckpt: `{ckpt}`\n")
        f.write(f"- protocol: tile=512, overlap=128, RGB-PSNR uint8 full-frame tiled\n\n")
        f.write("## Locked split means\n\n")
        f.write("| split | n | LQ mean | model mean | v9.0b baseline | Δ vs v9.0b |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        baselines = {"llve_stcd_12": 20.868, "outdoor_25": 21.779,
                     "original_indoor_14": 25.738, "bz_indoor_15": 27.016}
        for k in ["llve_stcd_12", "outdoor_25", "original_indoor_14", "bz_indoor_15"]:
            r = results.get(k)
            if r is None: continue
            mean, lq, n = r["mean"], r["lq"], r["n"]
            base = baselines[k]
            f.write(f"| {k} | {n} | {lq:.3f} | **{mean:.3f}** | {base:.3f} | {mean - base:+.3f} |\n")
        f.write("\n## Sentinel panel\n\n")
        s = results.get("sentinel")
        if s is not None:
            f.write(f"sentinel mean: **{s['mean']:.3f}** (n={len(s['rows'])})\n\n")
            f.write("| kind | seq | LQ | model | Δ vs LQ |\n|---|---|---:|---:|---:|\n")
            for r in s["rows"]:
                f.write(f"| {r['kind']} | {r['seq']} | {r['lq_psnr']:.2f} | "
                        f"{r['model_psnr']:.3f} | {r['delta_vs_lq']:+.3f} |\n")
        f.write("\n## Acceptance (V10_DESIGN_SPEC.md §7)\n\n")
        f.write("- LLVE-STCD-12: ≥ 21.87 (≥ +1.0 vs v9.0b)\n")
        f.write("- outdoor-25:   ≥ 22.78 (≥ +1.0 vs v9.0b)\n")
        f.write("- original-14:  ≥ 25.54 (≥ -0.2, no regression)\n")
    print(f"\nwrote {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v9", "v10"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--tag", required=True, help="subdir under v10_audit/")
    ap.add_argument("--sentinel-only", action="store_true",
                    help="skip the 4 splits; only run the 5-seq sentinel panel")
    ap.add_argument("--out-root", default="/cache/zihaohe/checkpoint/v10_audit")
    ap.add_argument("--allow-v9-ckpt", action="store_true",
                    help="permit v9-style pixel_branch-only ckpts (vjepa_pos_emb at random init)")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")

    print(f"[v10 audit] tag={args.tag}  model={args.model}  ckpt={args.ckpt}")
    print(f"[v10 audit] indexing datasets...")
    indoor_idx = index_seqs(INDOOR_ROOT, split="test")
    outdoor_idx = index_seqs(OUTDOOR_ROOT, split="test")
    print(f"[v10 audit] indoor test={len(indoor_idx)}  outdoor test={len(outdoor_idx)}")

    print(f"[v10 audit] building model...")
    model = build_model(args.model, args.cfg, args.ckpt, device,
                        allow_v9_ckpt=args.allow_v9_ckpt)

    results = {}
    if not args.sentinel_only:
        for label, idx, names in [
            ("llve_stcd_12",       indoor_idx,  LLVE_STCD_12),
            ("outdoor_25",         outdoor_idx, list(outdoor_idx.keys())),
            ("original_indoor_14", indoor_idx,  ORIGINAL_INDOOR_14),
            ("bz_indoor_15",       indoor_idx,  list(indoor_idx.keys())),
        ]:
            mean, lq, rows = run_split(model, device, label, idx, names, out_dir)
            results[label] = {"mean": mean, "lq": lq, "n": len(rows), "rows": rows}

    sm, sr = run_sentinel(model, device, indoor_idx, outdoor_idx, out_dir)
    results["sentinel"] = {"mean": sm, "rows": sr}

    write_summary(out_dir, args.tag, args.model, args.ckpt, results)
    print(f"\n[v10 audit] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
