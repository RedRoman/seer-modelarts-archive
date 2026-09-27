"""GT-mean diagnostic for all baselines on the six unified splits.

For every output frame: metrics of (a) the raw output and (b) the GT-mean output
pred_gm = clip(pred * mean(Y(gt)) / mean(Y(pred)), 0, 1), Y = BT.601 luma per frame on [0,1] RGB,
using seer_eval_gtmean.gt_mean itself. Metrics via seer_eval_unified.score_frames (uint8-RGB PSNR,
LPIPS, DISTS, CLIPIQA; per frame -> per clip mean -> mean over clips).

Splits / iterators (same crops as before):
  L0_SDSD-indoor  iter_sdsd_indoor_seer (16f, 512)   L1_SDSD-outdoor iter_sdsd_outdoor (16f, 512)
  L2_BVI-RLV      iter_bvi (16f, 512)                L3_LOL-v1 iter_lol_v1 (full res)
  L4_LOLv2-Syn    iter_lol_v2("Syn")                 L5_LOLv2-Real iter_lol_v2("Real")
Methods / weights (imported loaders + infer fns, identical to the earlier evals):
  Retinexformer  SDSD_indoor.pth (L0, L2, L3-5), SDSD_outdoor.pth (L1)       [run_retinexformer_v2]
  URCSA          RCSAindoor (L0, L2), RCSAoutdoor (L1), LOL checkpoint.pth (L3-5)  [run_urcsa_v2]
  HVI-CIDNet     Generalization (L0-2), LOLv1-wperc / LOLv2_syn-wperc / LOLv2_real-bestPSNR (L3/L4/L5)
  DarkIR, Zero-DCE, SCI  single checkpoint
  STCD           indoor_G.pth; video: 2-frame (t, t+1), replicate at clip end; images netG(x, x)
  Retinexformer-{union,indoor}-seersplit  net_g_24000.pth, one checkpoint for all splits
Raw numbers are checked against the earlier JSONs (baselines_L0_fixedpairing_eval.json for L0,
*_eval_v2.json for L1-L5, STCD_unified_eval.json, Retinexformer_*_seersplit_{L0seer_,}eval_v2.json).

Usage: run per split in parallel with --splits X --part_out P, then --merge P1 P2 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

BASE = "/temp/zihaohe/workspace/baselines"
sys.path.insert(0, BASE)
import run_temporal_all as RT  # noqa: E402  (sets up STCD / vjepa_llve / tools paths)
import run_stcd_lol as SL  # noqa: E402
import run_rf_seersplit_temporal as R3  # noqa: E402
from common.eval_lib import iter_sdsd_outdoor, iter_bvi, iter_lol_v1, iter_lol_v2  # noqa: E402
from seer_eval_unified import iter_sdsd_indoor_seer, score_frames  # noqa: E402
from seer_eval_gtmean import gt_mean  # noqa: E402

RUNS = "/temp/zihaohe/cache/checkpoint/runs"
EV = os.path.join(RUNS, "seer_iclr_evals")
OUT = os.path.join(EV, "baselines_gtmean_eval.json")
FK = ("psnr_rgb", "lpips", "dists", "clipiqa")
CLEAN6 = RT.CLEAN6

SPLITS = {
    "L0_SDSD-indoor": ("video", "indoor", lambda: iter_sdsd_indoor_seer(n_frames=16, crop=512)),
    "L1_SDSD-outdoor": ("video", "outdoor", lambda: iter_sdsd_outdoor(n_frames=16, crop=512)),
    "L2_BVI-RLV": ("video", "bvi", lambda: iter_bvi(n_frames=16, crop=512)),
    "L3_LOL-v1": ("image", "lol1", iter_lol_v1),
    "L4_LOLv2-Syn": ("image", "lolsyn", lambda: iter_lol_v2("Syn")),
    "L5_LOLv2-Real": ("image", "lolreal", lambda: iter_lol_v2("Real")),
}
OLD_DS = {"L1_SDSD-outdoor": "SDSD_outdoor", "L2_BVI-RLV": "BVI_RLV", "L3_LOL-v1": "LOL_v1",
          "L4_LOLv2-Syn": "LOLv2_Syn", "L5_LOLv2-Real": "LOLv2_Real"}


def build_methods(dev):
    M = {}

    def rf(mod):
        ck = mod.CKPT_DIR
        ind = mod.make_infer_fn(mod.load_model(os.path.join(ck, "SDSD_indoor.pth"), dev))
        out = mod.make_infer_fn(mod.load_model(os.path.join(ck, "SDSD_outdoor.pth"), dev))
        return dict(indoor=ind, outdoor=out, bvi=ind, lol1=ind, lolsyn=ind, lolreal=ind,
                    note="SDSD_indoor.pth (L0, L2, L3-5) / SDSD_outdoor.pth (L1)")
    M["Retinexformer"] = RT.load_isolated("run_retinexformer_v2.py", rf)

    def ur(mod):
        ind = mod.make_infer_sdsd(mod.load_sdsd(os.path.join(mod._URCSA, "sdsd", "RCSAindoor.pth"), dev))
        out = mod.make_infer_sdsd(mod.load_sdsd(os.path.join(mod._URCSA, "sdsd", "RCSAoutdoor.pth"), dev))
        lol_m, _ = mod.load_lol(os.path.join(mod._URCSA, "lol", "checkpoint.pth"), dev)
        lol = mod.make_infer_lol(lol_m)
        return dict(indoor=ind, outdoor=out, bvi=ind, lol1=lol, lolsyn=lol, lolreal=lol,
                    note="RCSAindoor (L0, L2) / RCSAoutdoor (L1) / LOL checkpoint.pth (L3-5)")
    M["URCSA"] = RT.load_isolated("run_urcsa_v2.py", ur)

    def hv(mod):
        f = lambda k: mod.make_infer_fn(mod.load_model(mod.CKPT[k], dev))  # noqa: E731
        g = f("Generalization")
        return dict(indoor=g, outdoor=g, bvi=g, lol1=f("LOLv1"), lolsyn=f("LOLv2_syn"),
                    lolreal=f("LOLv2_real"),
                    note="Generalization (L0-2) / LOLv1-wperc, LOLv2_syn-wperc, LOLv2_real-bestPSNR (L3/L4/L5)")
    M["HVI-CIDNet"] = RT.load_isolated("run_hvi_cidnet_v2.py", hv)

    for name, script, note in [("DarkIR", "run_darkir_v2.py", "DarkIR_384.pt (all)"),
                               ("Zero-DCE", "run_zerodce_v2.py", "Epoch99.pth (all)"),
                               ("SCI", "run_sci_v2.py", "SCI medium.pt (all)")]:
        def one(mod, note=note):
            f = mod.make_infer_fn(mod.load_model(dev))
            return dict(indoor=f, outdoor=f, bvi=f, lol1=f, lolsyn=f, lolreal=f, note=note)
        M[name] = RT.load_isolated(script, one)

    st = RT.RS.STCD(dev)
    img = lambda x: SL.enhance_image(st.net, x)  # noqa: E731
    M["STCD"] = dict(indoor=st.enhance_clip, outdoor=st.enhance_clip, bvi=st.enhance_clip,
                     lol1=img, lolsyn=img, lolreal=img, clip=True,
                     note="indoor_G.pth; video 2-frame (t,t+1) replicate at end; images netG(x,x); BGR in/out")

    for label, spec in R3.MODELS.items():
        f = RT.load_isolated("run_retinexformer_seer_protocol.py",
                             lambda mod, p=spec["ckpt"]: mod.make_infer_fn(mod.load_model(p, dev)))
        M[label] = dict(indoor=f, outdoor=f, bvi=f, lol1=f, lolsyn=f, lolreal=f,
                        note=f"{spec['ckpt']} (md5 {spec['md5']}), all splits")
    return M


def load_old():
    """(method, split, clip) -> old raw metrics dict."""
    old = {}
    fp = json.load(open(os.path.join(EV, "baselines_L0_fixedpairing_eval.json")))["per_seq"]["L0_SDSD-indoor"]
    for m, d in fp.items():
        for c, v in d.items():
            old[(m, "L0_SDSD-indoor", c)] = v
    for m in ["Retinexformer", "URCSA", "DarkIR", "HVI-CIDNet", "Zero-DCE", "SCI"]:
        ps = json.load(open(RT.OLD_JSON[m]))["per_seq"]
        for sp, ds in OLD_DS.items():
            for k, v in ps.items():
                if k.startswith(ds + "__"):
                    old[(m, sp, k[len(ds) + 2:])] = v
    st = json.load(open(os.path.join(EV, "STCD_unified_eval.json")))["per_seq"]
    for sp in SPLITS:
        for c, v in st.get(sp, {}).items():
            old[("STCD", sp, c)] = v
    for label, spec in R3.MODELS.items():
        l0 = json.load(open(os.path.join(EV, spec["old"]["L0"])))["per_seq"]
        for k, v in l0.items():
            old[(label, "L0_SDSD-indoor", k[len("SDSD_indoor_seer__"):])] = v
        ps = json.load(open(os.path.join(EV, spec["old"]["other"])))["per_seq"]
        for sp, ds in OLD_DS.items():
            for k, v in ps.items():
                if k.startswith(ds + "__"):
                    old[(label, sp, k[len(ds) + 2:])] = v
    return old


@torch.no_grad()
def predict(spec, dom, lq_u8, lq, dev):
    fn = spec[dom]
    if spec.get("clip") and dom in ("indoor", "outdoor", "bvi"):
        return fn(lq)
    outs = []
    for t in range(lq_u8.shape[0]):  # identical tensor prep to eval_lib_v2.evaluate_method_v2
        lq_t = torch.from_numpy(lq_u8[t]).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
        outs.append(fn(lq_t).clamp(0, 1).float())
    return torch.cat(outs, 0)


def summ(d, names=None):
    names = [n for n in (names if names is not None else d) if n in d]
    s = {k: float(np.mean([d[n][k] for n in names])) for k in FK} if names else {}
    s["n"] = len(names)
    return s


def run(splits, part_out):
    dev = torch.device("cuda")
    methods = build_methods(dev)
    old = load_old()
    per_seq, cons = {}, {}
    for split in splits:
        kind, dom, it_fn = SPLITS[split]
        per_seq[split] = {m: {"raw": {}, "gtmean": {}} for m in methods}
        t0 = time.time()
        for name, lq_u8, gt_u8 in it_fn():
            lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
            for m, spec in methods.items():
                pred = predict(spec, dom, lq_u8, lq, dev)
                r = score_frames(pred, gt)
                g = score_frames(gt_mean(pred, gt), gt)
                per_seq[split][m]["raw"][name] = r
                per_seq[split][m]["gtmean"][name] = g
                o = old.get((m, split, name))
                c = cons.setdefault(split, {}).setdefault(m, {"n": 0, "missing": 0, **{f"max_abs_d{k}": 0.0 for k in FK}})
                if o is None:
                    c["missing"] += 1
                else:
                    c["n"] += 1
                    for k in FK:
                        c[f"max_abs_d{k}"] = max(c[f"max_abs_d{k}"], abs(r[k] - o[k]))
                del pred
        print(f"== {split} done ({time.time() - t0:.0f}s)", flush=True)
        json.dump({"per_seq": per_seq, "consistency": cons,
                   "methods": {m: v["note"] for m, v in methods.items()}}, open(part_out, "w"))
    print(f"[PART] {part_out}", flush=True)


def merge(parts, out_path):
    per_seq, cons, methods = {}, {}, {}
    for p in parts:
        d = json.load(open(p))
        per_seq.update(d["per_seq"]); cons.update(d["consistency"]); methods.update(d["methods"])
    per_seq = {s: per_seq[s] for s in SPLITS if s in per_seq}
    summary = {}
    for s, dm in per_seq.items():
        summary[s] = {}
        for m, dv in dm.items():
            summary[s][m] = {v: summ(dd) for v, dd in dv.items()}
            if s == "L0_SDSD-indoor":
                for v, dd in dv.items():
                    summary[s][m][v]["clean6"] = summ(dd, CLEAN6)
    out = {
        "protocol": ("same clips/crops/metric functions as the unified evals; per frame raw and GT-mean "
                     "(seer_eval_gtmean.gt_mean: pred*mean(Y(gt))/mean(Y(pred)), BT.601 luma per frame, "
                     "clamped to [0,1]); per clip mean -> mean over clips. Diagnostic only (uses GT)."),
        "methods": methods,
        "clean6_definition": CLEAN6,
        "consistency_raw_vs_earlier": cons,
        "per_seq": per_seq,
        "summary": summary,
    }
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[SAVE] {out_path}")
    print("consistency (max |d| over clips, n compared / missing):")
    for s, dm in cons.items():
        for m, c in dm.items():
            print(f"  {s:16s} {m:32s} n={c['n']} miss={c['missing']} "
                  + " ".join(f"{k[10:]}={c[k]:.2e}" for k in c if k.startswith("max_abs_d")))
    for s in summary:
        for m, sv in summary[s].items():
            r, g = sv["raw"], sv["gtmean"]
            print(f"{s:16s} {m:32s} PSNR {r['psnr_rgb']:.3f}->{g['psnr_rgb']:.3f}  LPIPS {r['lpips']:.4f}->{g['lpips']:.4f}"
                  f"  DISTS {r['dists']:.4f}->{g['dists']:.4f}  CLIPIQA {r['clipiqa']:.4f}->{g['clipiqa']:.4f} n={r['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--part_out", default="/cache/zihaohe/stcd_unified/gtmean_parts/all.json")
    ap.add_argument("--merge", nargs="+", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.merge:
        merge(args.merge, args.out)
    else:
        os.makedirs(os.path.dirname(args.part_out), exist_ok=True)
        run(args.splits, args.part_out)


if __name__ == "__main__":
    main()
