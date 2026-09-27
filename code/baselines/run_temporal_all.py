"""Temporal metrics (tlp / mabd / flicker / warp_err) for all single-frame baselines + STCD
on the unified 16-frame 512x512 clips, using the shared tools/temporal_metrics.py (same as Seer).

Splits (same iterators as the rest of the ICLR evals):
  L0_SDSD-indoor       seer_eval_unified.iter_sdsd_indoor_seer   (Seer's 15-seq test split, index-paired)
  STCD-12_SDSD-indoor  eval_lib.iter_sdsd_indoor_stcd            (basename-paired, see NOTE)
  L1_SDSD-outdoor      eval_lib.iter_sdsd_outdoor
  L2_BVI-RLV           eval_lib.iter_bvi

Per clip: GT flows (RAFT-small, t+1->t, fw-bw occlusion mask) are computed ONCE via
safe_gt_flows(gt) and reused for every method. Each baseline is run frame by frame exactly like
eval_lib_v2.evaluate_method_v2 (same tensor prep, same infer_fn from the method's run_*_v2.py,
same PSNR/LPIPS calls); STCD uses run_stcd_unified.STCD.enhance_clip (2-frame model,
neighbour t+1 inside the clip, replicate at the end). The float [0,1] outputs are stacked to
(16,3,512,512) and passed to temporal_metrics(pred, lpips_batch, flows). GT and LQ rows are stored
as references. PSNR/LPIPS are recomputed and checked against the existing *_eval_v2.json
(STCD: STCD_unified_eval.json).

Weights per split follow the run_*_v2.py weight maps:
  Retinexformer: SDSD_indoor.pth (indoor splits, BVI), SDSD_outdoor.pth (outdoor)
  URCSA:         RCSAindoor.pth  (indoor splits, BVI), RCSAoutdoor.pth (outdoor)
  HVI-CIDNet:    Generalization.bin (all three video datasets)
  DarkIR / Zero-DCE / SCI: single checkpoint;  STCD: indoor_G.pth everywhere.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

BASE = "/temp/zihaohe/workspace/baselines"
TOOLS = "/temp/zihaohe/workspace/vjepa_llve/tools"
RUNS = "/temp/zihaohe/cache/checkpoint/runs"
OUT = os.path.join(RUNS, "seer_iclr_evals", "baselines_temporal_unified.json")
SITE_PREFIX = "/cache/zihaohe/conda_envs"
CLEAN6 = ["pair50", "pair52", "pair58", "pair60", "pair68", "pair70"]

sys.path.insert(0, BASE)
import run_stcd_unified as RS  # noqa: E402  (isolates STCD's `models` package; adds VJ/tools/BASE)
from common.eval_lib import (  # noqa: E402
    iter_sdsd_indoor_stcd, iter_sdsd_outdoor, iter_bvi, rgb_psnr_uint8, lpips_batch,
)
from seer_eval_unified import iter_sdsd_indoor_seer  # noqa: E402
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from temporal_metrics import safe_gt_flows, temporal_metrics  # noqa: E402

SPLITS = {
    "L0_SDSD-indoor": ("indoor", lambda: iter_sdsd_indoor_seer(n_frames=16, crop=512)),
    "STCD-12_SDSD-indoor": ("indoor", lambda: iter_sdsd_indoor_stcd(n_frames=16, crop=512)),
    "L1_SDSD-outdoor": ("outdoor", lambda: iter_sdsd_outdoor(n_frames=16, crop=512)),
    "L2_BVI-RLV": ("bvi", lambda: iter_bvi(n_frames=16, crop=512)),
}
OLD_KEY = {"STCD-12_SDSD-indoor": "SDSD_indoor", "L1_SDSD-outdoor": "SDSD_outdoor",
           "L2_BVI-RLV": "BVI_RLV"}
OLD_JSON = {
    "Retinexformer": os.path.join(RUNS, "Retinexformer_eval_v2.json"),
    "URCSA": os.path.join(RUNS, "URCSA_eval_v2.json"),
    "DarkIR": os.path.join(RUNS, "DarkIR_eval_v2.json"),
    "HVI-CIDNet": os.path.join(RUNS, "HVI-CIDNet_eval_v2.json"),
    "Zero-DCE": os.path.join(RUNS, "Zero-DCE_eval_v2.json"),
    "SCI": os.path.join(RUNS, "SCI_eval_v2.json"),
    "STCD": os.path.join(RUNS, "seer_iclr_evals", "STCD_unified_eval.json"),
}
TKEYS = ("tlp", "mabd", "flicker", "warp_err")
MKEYS = ("psnr_rgb", "lpips") + TKEYS


def load_isolated(script, build):
    """exec BASE/<script> as a fresh module, call build(mod) -> dict, then drop every newly
    imported non-site-packages module (e.g. `model`, `net`, `archs`, `loss`) so the next
    baseline's same-named modules do not collide. `common.*` is kept (shared harness)."""
    saved_path = list(sys.path)
    before = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location("_bl_" + script[:-3], os.path.join(BASE, script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return build(mod)
    finally:
        sys.path[:] = saved_path
        for k in set(sys.modules) - before:
            if k == "common" or k.startswith("common.") or not _is_local(sys.modules[k]):
                continue
            del sys.modules[k]


def _is_local(mod):
    """True for modules that live in a method repo (not site-packages / stdlib / builtins)."""
    f = getattr(mod, "__file__", None)
    if f:
        return not f.startswith(SITE_PREFIX)
    p = list(getattr(mod, "__path__", None) or [])  # namespace package, e.g. HVI-CIDNet `net`
    return bool(p) and not str(p[0]).startswith(SITE_PREFIX)


def build_methods(dev):
    """Returns {name: {"indoor": fn, "outdoor": fn, "bvi": fn, "kind": "frame"|"clip", "note": str}}."""
    M = {}

    def rf(mod):
        ck = mod.CKPT_DIR
        ind = mod.make_infer_fn(mod.load_model(os.path.join(ck, "SDSD_indoor.pth"), dev))
        out = mod.make_infer_fn(mod.load_model(os.path.join(ck, "SDSD_outdoor.pth"), dev))
        return dict(indoor=ind, outdoor=out, bvi=ind, kind="frame",
                    note="SDSD_indoor.pth (indoor, BVI) / SDSD_outdoor.pth (outdoor)")
    M["Retinexformer"] = load_isolated("run_retinexformer_v2.py", rf)

    def ur(mod):
        ind = mod.make_infer_sdsd(mod.load_sdsd(os.path.join(mod._URCSA, "sdsd", "RCSAindoor.pth"), dev))
        out = mod.make_infer_sdsd(mod.load_sdsd(os.path.join(mod._URCSA, "sdsd", "RCSAoutdoor.pth"), dev))
        return dict(indoor=ind, outdoor=out, bvi=ind, kind="frame",
                    note="RCSAindoor.pth (indoor, BVI) / RCSAoutdoor.pth (outdoor), deline patch")
    M["URCSA"] = load_isolated("run_urcsa_v2.py", ur)

    def dk(mod):
        f = mod.make_infer_fn(mod.load_model(dev))
        return dict(indoor=f, outdoor=f, bvi=f, kind="frame", note="DarkIR_384.pt (all)")
    M["DarkIR"] = load_isolated("run_darkir_v2.py", dk)

    def hv(mod):
        f = mod.make_infer_fn(mod.load_model(mod.CKPT["Generalization"], dev))
        return dict(indoor=f, outdoor=f, bvi=f, kind="frame", note="Generalization.bin (all video)")
    M["HVI-CIDNet"] = load_isolated("run_hvi_cidnet_v2.py", hv)

    def zd(mod):
        f = mod.make_infer_fn(mod.load_model(dev))
        return dict(indoor=f, outdoor=f, bvi=f, kind="frame", note="Epoch99.pth (all)")
    M["Zero-DCE"] = load_isolated("run_zerodce_v2.py", zd)

    def sc(mod):
        f = mod.make_infer_fn(mod.load_model(dev))
        return dict(indoor=f, outdoor=f, bvi=f, kind="frame", note="SCI medium.pt (all)")
    M["SCI"] = load_isolated("run_sci_v2.py", sc)

    st = RS.STCD(dev)
    M["STCD"] = dict(indoor=st.enhance_clip, outdoor=st.enhance_clip, bvi=st.enhance_clip, kind="clip",
                     note="indoor_G.pth (all); 2-frame (t, t+1) with replicate pad at clip end; BGR in/out")
    return M


@torch.no_grad()
def run_frames(infer_fn, lq_arr, gt_arr, dev):
    """Exactly evaluate_method_v2's per-frame path; returns pred (T,3,H,W) float and psnr/lpips lists."""
    preds, ps, lp = [], [], []
    for t in range(lq_arr.shape[0]):
        lq_t = torch.from_numpy(lq_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
        gt_t = torch.from_numpy(gt_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
        out = infer_fn(lq_t).clamp(0, 1).float()
        ps.append(rgb_psnr_uint8(
            (out[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
            (gt_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)))
        lp.append(lpips_batch(out, gt_t))
        preds.append(out)
    return torch.cat(preds, 0), ps, lp


@torch.no_grad()
def frame_scores(pred, gt):
    ps, lp = [], []
    for t in range(pred.shape[0]):
        ps.append(rgb_psnr_uint8(
            (pred[t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
            (gt[t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)))
        lp.append(lpips_batch(pred[t:t + 1], gt[t:t + 1]))
    return ps, lp


def old_lookup(old, method, split, name):
    if old is None:
        return None
    if method == "STCD":
        return old.get("per_seq", {}).get(split, {}).get(name)
    if split not in OLD_KEY:
        return None
    return old.get("per_seq", {}).get(f"{OLD_KEY[split]}__{name}")


def summarize(d, names=None):
    names = [n for n in (names if names is not None else d.keys()) if n in d]
    keys = [k for k in MKEYS if names and all(k in d[n] for n in names)]
    s = {k: float(np.mean([d[n][k] for n in names])) for k in keys}
    s["n"] = len(names)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    methods = build_methods(dev)
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}
    olds = {}
    for m in methods:
        try:
            olds[m] = json.load(open(OLD_JSON[m]))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] no old json for {m}: {e}")
            olds[m] = None

    rows = ["GT", "LQ"] + list(methods)
    per_seq = {}
    summary = {}
    check = {m: {} for m in methods}
    t_all = time.time()
    for split in args.splits:
        dom, it_fn = SPLITS[split]
        per_seq[split] = {r: {} for r in rows}
        t0 = time.time()
        for i, (name, lq_u8, gt_u8) in enumerate(it_fn()):
            if args.limit is not None and i >= args.limit:
                break
            lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            flows = safe_gt_flows(gt)
            per_seq[split]["GT"][name] = temporal_metrics(gt, lpips_batch, flows)
            ps, lp = frame_scores(lq, gt)
            r = dict(psnr_rgb=float(np.mean(ps)), lpips=float(np.mean(lp)), n_frames=len(ps))
            r.update(temporal_metrics(lq, lpips_batch, flows))
            per_seq[split]["LQ"][name] = r
            msg = []
            for m, spec in methods.items():
                fn = spec[dom]
                if spec["kind"] == "frame":
                    pred, ps, lp = run_frames(fn, lq_u8, gt_u8, dev)
                else:
                    pred = fn(lq)
                    ps, lp = frame_scores(pred, gt)
                r = dict(psnr_rgb=float(np.mean(ps)), lpips=float(np.mean(lp)), n_frames=len(ps))
                r.update(temporal_metrics(pred, lpips_batch, flows))
                o = old_lookup(olds[m], m, split, name)
                if o is not None:
                    r["old_psnr_rgb"] = o["psnr_rgb"]; r["old_lpips"] = o["lpips"]
                    check[m].setdefault(split, []).append(
                        (abs(r["psnr_rgb"] - o["psnr_rgb"]), abs(r["lpips"] - o["lpips"])))
                per_seq[split][m][name] = r
                msg.append(f"{m}:{r['psnr_rgb']:.2f}/{r.get('warp_err', float('nan')):.4f}")
                del pred
            g = per_seq[split]["GT"][name]
            print(f"  {split:20s} {name:24s} GT warp={g.get('warp_err', float('nan')):.4f} | "
                  + " ".join(msg), flush=True)
        summary[split] = {}
        for rname, d in per_seq[split].items():
            if not d:
                continue
            summary[split][rname] = summarize(d)
            if split in ("L0_SDSD-indoor", "STCD-12_SDSD-indoor"):
                summary[split][rname]["clean6"] = summarize(d, CLEAN6)
        print(f"== {split} done ({time.time() - t0:.0f}s)", flush=True)
        for rname, s in summary[split].items():
            print(f"   {rname:14s} " + " ".join(f"{k}={s[k]:.4f}" for k in MKEYS if k in s)
                  + f" n={s['n']}", flush=True)
        consistency = {}
        for m, dd in check.items():
            consistency[m] = {sp: {"n": len(v), "max_abs_dpsnr": float(max(a for a, _ in v)),
                                   "max_abs_dlpips": float(max(b for _, b in v))}
                              for sp, v in dd.items() if v}
        out = {
            "protocol": ("unified clips: center T=16, 512x512 center crop (same iterators as the ICLR "
                         "evals); per-frame inference as eval_lib_v2 (STCD: clip-level 2-frame); "
                         "temporal metrics from vjepa_llve/tools/temporal_metrics.py on float [0,1] "
                         "outputs; GT RAFT-small flows computed once per clip and shared by all rows"),
            "methods": {m: v["note"] for m, v in methods.items()},
            "clean6_definition": CLEAN6,
            "notes": [
                "Use L0_SDSD-indoor (index-paired, Seer's split) for Seer comparisons; its clean6 = the 6 "
                "seqs held out for both STCD and Seer.",
                "STCD-12_SDSD-indoor uses eval_lib.iter_sdsd_indoor_stcd, which pairs LQ/GT by basename; "
                "for pair50/52/58/60/68/70 filenames are offset and the GT is temporally misaligned "
                "(content is aligned by sorted index). Its PSNR/LPIPS/warp_err for those seqs are "
                "not meaningful; kept only for consistency with the existing *_eval_v2.json.",
            ],
            "consistency_vs_existing_json": consistency,
            "per_seq": per_seq,
            "summary": summary,
            "wall_s": round(time.time() - t_all, 1),
        }
        tmp = "/cache/zihaohe/stcd_unified/baselines_temporal_unified.partial.json"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[SAVE] {args.out}", flush=True)
    print("consistency:", json.dumps(out["consistency_vs_existing_json"], indent=1))


if __name__ == "__main__":
    main()
