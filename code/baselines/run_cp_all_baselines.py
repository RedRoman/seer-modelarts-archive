"""Chroma-preservation (CP) post-processor applied to every baseline, with exactly the two
calibrations used for Seer in tools/seer_eval_unified.py.

CP (quick_ablations/cp_transfer/cp_module.py): Ycp = X * Y(Yhat) / (Y(X) + 1e-3) (BT.601 luma),
Ystar_c = a_c * Ycp_c + (1 - a_c) * Yhat_c, clipped to [0, 1]. Per-channel
a_c = (m22 - m12) / (m11 - 2 m12 + m22), uncentred moments of e1 = Ycp - Y, e2 = Yhat - Y pooled over
all pixels of the fit set (seer_eval_unified.moments / alpha_from), clipped to [0, 1]
(0.5 if the denominator vanishes); raw values are stored too.

  CP-src : alpha fitted on the 15 fixed SDSD *train* clips (seer_eval_unified.iter_sdsd_train_fitset:
           8 indoor clips first, then 7 outdoor clips). Each method is run on the fit clips with the
           checkpoint it uses for that SDSD domain (Retinexformer SDSD_indoor.pth / SDSD_outdoor.pth,
           URCSA RCSAindoor.pth / RCSAoutdoor.pth, HVI-CIDNet Generalization.bin, STCD indoor_G.pth,
           DarkIR / Zero-DCE / SCI their single checkpoint). One alpha per method, applied to all splits.
  CP-tgt : target split cross-fitted in 2 folds by scene (same block as "HMP+CPxfit": scenes =
           sorted scene_key(name), fold = index parity; each clip is fused with the alpha fitted on the
           other fold).

Baselines, per-split checkpoints and inference: identical to run_temporal_all.build_methods (video
splits) and the run_*_v2.py weight maps (LOL splits); STCD via run_stcd_unified.STCD.enhance_clip
(video) and run_stcd_lol.enhance_image (images). Frame-based methods run frame by frame with batch 1
(same tensor prep as eval_lib_v2.evaluate_method_v2); STCD video inference uses chunk=--stcd_chunk.

Splits / crops: L0 seer_eval_unified.iter_sdsd_indoor_seer, L1 iter_sdsd_outdoor, L2 iter_bvi
(center T=16, 512x512 crop), L3 iter_lol_v1, L4 iter_lol_v2("Syn"), L5 iter_lol_v2("Real")
(full resolution). Metrics: seer_eval_unified.score_frames (per frame uint8-RGB PSNR, LPIPS, DISTS,
CLIPIQA -> mean per clip -> mean over clips).

Consistency: raw per-clip PSNR/LPIPS are compared with baselines_L0_fixedpairing_eval.json (L0) and
<method>_eval_v2.json / STCD_unified_eval.json (L1-L5).

Parallel use: run disjoint --methods subsets with different --out, then
  python run_cp_all_baselines.py --merge part1.json part2.json ... --out final.json
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
import run_temporal_all as RT  # noqa: E402  (isolates STCD's `models`; adds vjepa_llve/tools/BASE paths)
import run_stcd_unified as RS  # noqa: E402
import run_stcd_lol as RSL  # noqa: E402
from common.eval_lib import (  # noqa: E402
    STCD_INDOOR_12, iter_sdsd_outdoor, iter_bvi, iter_lol_v1, iter_lol_v2,
)
from seer_eval_unified import (  # noqa: E402  (also puts cp_transfer on sys.path)
    iter_sdsd_indoor_seer, iter_sdsd_train_fitset, moments, alpha_from, scene_key, score_frames,
)
from cp_module import cp_surrogate, fuse_per_channel  # noqa: E402

RUNS = "/temp/zihaohe/cache/checkpoint/runs"
EVDIR = os.path.join(RUNS, "seer_iclr_evals")
OUT = os.path.join(EVDIR, "baselines_cp_eval.json")
OLD_L0 = os.path.join(EVDIR, "baselines_L0_fixedpairing_eval.json")
OLD_V2_KEY = {"L1_SDSD-outdoor": "SDSD_outdoor", "L2_BVI-RLV": "BVI_RLV", "L3_LOL-v1": "LOL_v1",
              "L4_LOLv2-Syn": "LOLv2_Syn", "L5_LOLv2-Real": "LOLv2_Real"}
SPLITS = {
    "L0_SDSD-indoor": ("video", lambda: iter_sdsd_indoor_seer(n_frames=16, crop=512)),
    "L1_SDSD-outdoor": ("video", lambda: iter_sdsd_outdoor(n_frames=16, crop=512)),
    "L2_BVI-RLV": ("video", lambda: iter_bvi(n_frames=16, crop=512)),
    "L3_LOL-v1": ("image", iter_lol_v1),
    "L4_LOLv2-Syn": ("image", lambda: iter_lol_v2("Syn")),
    "L5_LOLv2-Real": ("image", lambda: iter_lol_v2("Real")),
}
ALL_METHODS = ["Retinexformer", "URCSA", "DarkIR", "HVI-CIDNet", "Zero-DCE", "SCI", "STCD"]
VARIANTS = ("raw", "CP-src", "CP-tgt")
FKEYS = ("psnr_rgb", "lpips", "dists", "clipiqa")
N_FIT_INDOOR = 8  # iter_sdsd_train_fitset yields n_in=8 SDSD-indoor clips, then n_out=7 SDSD-outdoor clips
TOL = 1e-3


def frame_pred(infer, dev):
    """Per-frame inference with eval_lib_v2.evaluate_method_v2's tensor prep; -> (T,3,H,W) float."""
    @torch.no_grad()
    def f(lq_u8, lq):
        outs = []
        for t in range(lq_u8.shape[0]):
            x = torch.from_numpy(lq_u8[t]).permute(2, 0, 1).unsqueeze(0).float().to(dev) / 255.0
            outs.append(infer(x).clamp(0, 1).float())
        return torch.cat(outs, 0)
    return f


def build_methods(dev, names, stcd_chunk):
    """{method: {"fn": {key: f(lq_u8, lq) -> pred}, "ckpt": {key: str}}}; key in SPLITS + fit_indoor/fit_outdoor."""
    M = {}
    LOL = ("L3_LOL-v1", "L4_LOLv2-Syn", "L5_LOLv2-Real")

    def table(fit_in, fit_out, l0, l1, l2, l3, l4, l5):
        return dict(zip(("fit_indoor", "fit_outdoor") + tuple(SPLITS), (fit_in, fit_out, l0, l1, l2, l3, l4, l5)))

    if "Retinexformer" in names:
        def rf(mod):
            ck = mod.CKPT_DIR
            ind = frame_pred(mod.make_infer_fn(mod.load_model(os.path.join(ck, "SDSD_indoor.pth"), dev)), dev)
            out = frame_pred(mod.make_infer_fn(mod.load_model(os.path.join(ck, "SDSD_outdoor.pth"), dev)), dev)
            i, o = "SDSD_indoor.pth", "SDSD_outdoor.pth"
            return dict(fn=table(ind, out, ind, out, ind, ind, ind, ind), ckpt=table(i, o, i, o, i, i, i, i))
        M["Retinexformer"] = RT.load_isolated("run_retinexformer_v2.py", rf)

    if "URCSA" in names:
        def ur(mod):
            ind = frame_pred(mod.make_infer_sdsd(mod.load_sdsd(os.path.join(mod._URCSA, "sdsd", "RCSAindoor.pth"), dev)), dev)
            out = frame_pred(mod.make_infer_sdsd(mod.load_sdsd(os.path.join(mod._URCSA, "sdsd", "RCSAoutdoor.pth"), dev)), dev)
            lol = frame_pred(mod.make_infer_lol(mod.load_lol(os.path.join(mod._URCSA, "lol", "checkpoint.pth"), dev)[0]), dev)
            i, o, l = "RCSAindoor.pth", "RCSAoutdoor.pth", "lol/checkpoint.pth"
            return dict(fn=table(ind, out, ind, out, ind, lol, lol, lol), ckpt=table(i, o, i, o, i, l, l, l))
        M["URCSA"] = RT.load_isolated("run_urcsa_v2.py", ur)

    if "DarkIR" in names:
        def dk(mod):
            f = frame_pred(mod.make_infer_fn(mod.load_model(dev)), dev)
            return dict(fn=table(*[f] * 8), ckpt=table(*["DarkIR_384.pt"] * 8))
        M["DarkIR"] = RT.load_isolated("run_darkir_v2.py", dk)

    if "HVI-CIDNet" in names:
        def hv(mod):
            ld = lambda k: frame_pred(mod.make_infer_fn(mod.load_model(mod.CKPT[k], dev)), dev)  # noqa: E731
            gen, l1, l2s, l2r = ld("Generalization"), ld("LOLv1"), ld("LOLv2_syn"), ld("LOLv2_real")
            g = "Generalization.bin"
            return dict(fn=table(gen, gen, gen, gen, gen, l1, l2s, l2r),
                        ckpt=table(g, g, g, g, g, "LOLv1-wperc.bin", "LOLv2_syn-wperc.bin",
                                   "LOLv2_real-bestPSNR.bin"))
        M["HVI-CIDNet"] = RT.load_isolated("run_hvi_cidnet_v2.py", hv)

    if "Zero-DCE" in names:
        def zd(mod):
            f = frame_pred(mod.make_infer_fn(mod.load_model(dev)), dev)
            return dict(fn=table(*[f] * 8), ckpt=table(*["Epoch99.pth"] * 8))
        M["Zero-DCE"] = RT.load_isolated("run_zerodce_v2.py", zd)

    if "SCI" in names:
        def sc(mod):
            f = frame_pred(mod.make_infer_fn(mod.load_model(dev)), dev)
            return dict(fn=table(*[f] * 8), ckpt=table(*["SCI medium.pt"] * 8))
        M["SCI"] = RT.load_isolated("run_sci_v2.py", sc)

    if "STCD" in names:
        st = RS.STCD(dev)
        clip = lambda lq_u8, lq: st.enhance_clip(lq, chunk=stcd_chunk)  # noqa: E731
        img = lambda lq_u8, lq: RSL.enhance_image(st.net, lq)  # noqa: E731
        fn = table(clip, clip, clip, clip, clip, img, img, img)
        ck = {k: ("indoor_G.pth, netG(x,x)[2]" if k in LOL else "indoor_G.pth, 2-frame clip") for k in fn}
        M["STCD"] = dict(fn=fn, ckpt=ck)
    return {m: M[m] for m in ALL_METHODS if m in M}


def to_dev(lq_u8, gt_u8, dev):
    lq = torch.from_numpy(np.ascontiguousarray(lq_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
    gt = torch.from_numpy(np.ascontiguousarray(gt_u8)).float().div(255).permute(0, 3, 1, 2).to(dev)
    return lq, gt


@torch.no_grad()
def fit_src_alpha(spec, fit_data, dev):
    acc = torch.zeros(3, 3, dtype=torch.float64)
    for i, (name, lq_u8, gt_u8) in enumerate(fit_data):
        lq, gt = to_dev(lq_u8, gt_u8, dev)
        pred = spec["fn"]["fit_indoor" if i < N_FIT_INDOOR else "fit_outdoor"](lq_u8, lq)
        acc += moments(pred, cp_surrogate(pred, lq), gt)
        del pred
    a, raw = alpha_from(acc)
    return a, raw


def old_row(old, method, split, name):
    try:
        if split == "L0_SDSD-indoor":
            return old["L0"]["per_seq"][split][method][name]
        if method == "STCD":
            return old["STCD"]["per_seq"][split][name]
        return old[method]["per_seq"][f"{OLD_V2_KEY[split]}__{name}"]
    except (KeyError, TypeError):
        return None


def summarize(per_seq):
    summ = {}
    for split, dm in per_seq.items():
        summ[split] = {}
        for m, dv in dm.items():
            summ[split][m] = {}
            for v, d in dv.items():
                if not d:
                    continue
                s = {k: float(np.mean([r[k] for r in d.values()])) for k in FKEYS}
                s["n"] = len(d)
                if split == "L0_SDSD-indoor":
                    c6 = [n for n in d if n in STCD_INDOOR_12]
                    if c6:
                        s["clean6_psnr_rgb"] = float(np.mean([d[n]["psnr_rgb"] for n in c6]))
                        s["clean6_lpips"] = float(np.mean([d[n]["lpips"] for n in c6]))
                summ[split][m][v] = s
    return summ


def print_table(out):
    for split in SPLITS:
        if split not in out["summary"]:
            continue
        print(f"\n== {split}   PSNR raw -> CP-src -> CP-tgt   |   LPIPS raw -> CP-src -> CP-tgt")
        for m in ALL_METHODS:
            s = out["summary"][split].get(m)
            if not s or "raw" not in s:
                continue
            g = lambda v, k: s[v][k] if v in s else float("nan")  # noqa: E731
            print(f"  {m:14s} {g('raw', 'psnr_rgb'):6.3f} -> {g('CP-src', 'psnr_rgb'):6.3f} -> "
                  f"{g('CP-tgt', 'psnr_rgb'):6.3f}   |   {g('raw', 'lpips'):.4f} -> {g('CP-src', 'lpips'):.4f} -> "
                  f"{g('CP-tgt', 'lpips'):.4f}   n={s['raw']['n']}")
    print("\nCP-src alpha (clipped / raw):")
    for m, a in out["alpha"].items():
        if "src" in a:
            print(f"  {m:14s} {np.round(a['src']['clipped'], 4).tolist()}  raw={np.round(a['src']['raw'], 4).tolist()}")


def header():
    return {
        "protocol": ("CP (cp_module.cp_surrogate / fuse_per_channel, BT.601 luma, eps=1e-3) on every baseline. "
                     "CP-src: per-channel alpha* fitted on the 15 SDSD train clips of "
                     "seer_eval_unified.iter_sdsd_train_fitset (8 indoor with the method's SDSD-indoor ckpt, "
                     "7 outdoor with its SDSD-outdoor ckpt), clipped to [0,1]. CP-tgt: 2-fold cross-fit on the "
                     "target split by scene (scenes = sorted scene_key(name), fold = index parity; each clip "
                     "fused with the alpha of the other fold), clipped to [0,1]. Splits/crops/metrics as the "
                     "unified ICLR eval (score_frames: uint8-RGB PSNR, LPIPS, DISTS, CLIPIQA; per clip -> "
                     "mean over clips). Per-split ckpts as run_temporal_all / run_*_v2.py / run_stcd_lol."),
        "script": os.path.abspath(__file__),
    }


def merge(parts, out_path):
    out = dict(header(), methods={}, per_seq={}, alpha={}, consistency={})
    for p in parts:
        d = json.load(open(p))
        out["methods"].update(d.get("methods", {}))
        out["alpha"].update(d.get("alpha", {}))
        out["consistency"].update(d.get("consistency", {}))
        for split, dm in d.get("per_seq", {}).items():
            out["per_seq"].setdefault(split, {}).update(dm)
    out["per_seq"] = {s: {m: out["per_seq"][s][m] for m in ALL_METHODS if m in out["per_seq"][s]}
                      for s in SPLITS if s in out["per_seq"]}
    out["summary"] = summarize(out["per_seq"])
    out["consistency_ok"] = all(c.get("ok", False) for cm in out["consistency"].values() for c in cm.values())
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[SAVE] {out_path}")
    print_table(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stcd_chunk", type=int, default=1)
    ap.add_argument("--merge", nargs="+", default=None)
    args = ap.parse_args()
    if args.merge:
        merge(args.merge, args.out)
        return

    dev = torch.device("cuda")
    t_all = time.time()
    methods = build_methods(dev, args.methods, args.stcd_chunk)
    old = {"L0": json.load(open(OLD_L0))}
    for m in methods:
        old[m] = json.load(open(RT.OLD_JSON[m]))

    out = dict(header(), methods={m: s["ckpt"] for m, s in methods.items()},
               per_seq={}, alpha={m: {} for m in methods}, consistency={m: {} for m in methods})

    fit_data = list(iter_sdsd_train_fitset())
    fit_names = [n for n, _, _ in fit_data]
    a_src = {}
    for m, spec in methods.items():
        a, raw = fit_src_alpha(spec, fit_data, dev)
        a_src[m] = a
        out["alpha"][m]["src"] = {"clipped": a.tolist(), "raw": raw, "fit_clips": fit_names,
                                  "fit_ckpt": {"indoor(8)": spec["ckpt"]["fit_indoor"],
                                               "outdoor(7)": spec["ckpt"]["fit_outdoor"]}}
        print(f"[CP-src] {m:14s} alpha raw={np.round(raw, 4).tolist()} clipped={np.round(a.tolist(), 4).tolist()} "
              f"({time.time() - t_all:.0f}s)", flush=True)

    for split in args.splits:
        _, it_fn = SPLITS[split]
        data = {}
        for i, (name, lq_u8, gt_u8) in enumerate(it_fn()):
            if args.limit is not None and i >= args.limit:
                break
            assert name not in data, f"duplicate clip name {name} in {split}"
            data[name] = (lq_u8, gt_u8)
        out["per_seq"][split] = {}
        for m, spec in methods.items():
            t0 = time.time()
            fn = spec["fn"][split]
            dv = {v: {} for v in VARIANTS}
            stash = {}
            with torch.no_grad():
                for name, (lq_u8, gt_u8) in data.items():
                    lq, gt = to_dev(lq_u8, gt_u8, dev)
                    pred = fn(lq_u8, lq)
                    surr = cp_surrogate(pred, lq)
                    dv["raw"][name] = score_frames(pred, gt)
                    dv["CP-src"][name] = score_frames(fuse_per_channel(pred, surr, a_src[m]), gt)
                    stash[name] = (pred.cpu(), moments(pred, surr, gt))
                    del pred, surr
                # ---- CP-tgt: 2-fold cross-fit by scene (same as seer_eval_unified "HMP+CPxfit") ----
                names = sorted(stash)
                scenes = sorted({scene_key(n) for n in names})
                fold = {n: scenes.index(scene_key(n)) % 2 for n in names}
                xf = {}
                for f in (0, 1):
                    other = [n for n in names if fold[n] != f]
                    if not other:
                        continue
                    a, raw = alpha_from(sum(stash[n][1] for n in other))
                    xf[str(f)] = {"clipped": a.tolist(), "raw": raw, "fit_on": other,
                                  "applied_to": [n for n in names if fold[n] == f]}
                    for n in names:
                        if fold[n] != f:
                            continue
                        lq, gt = to_dev(*data[n], dev)
                        pred = stash[n][0].to(dev)
                        dv["CP-tgt"][n] = score_frames(fuse_per_channel(pred, cp_surrogate(pred, lq), a), gt)
                        del pred
            del stash
            out["per_seq"][split][m] = dv
            out["alpha"][m][split] = xf
            # ---- consistency of raw vs existing JSONs ----
            diffs = {k: [] for k in FKEYS}
            miss = []
            for n, r in dv["raw"].items():
                o = old_row(old, m, split, n)
                if o is None:
                    miss.append(n)
                    continue
                for k in FKEYS:
                    if k in o:
                        diffs[k].append(abs(r[k] - o[k]))
            c = {"n_compared": len(dv["raw"]) - len(miss), "missing_in_old": miss}
            for k in FKEYS:
                c[f"max_abs_d{k}"] = float(max(diffs[k])) if diffs[k] else None
            c["ok"] = (not miss and c["max_abs_dpsnr_rgb"] is not None and c["max_abs_dpsnr_rgb"] < TOL
                       and c["max_abs_dlpips"] < TOL)
            out["consistency"][m][split] = c
            s = {v: {k: float(np.mean([r[k] for r in dv[v].values()])) for k in ("psnr_rgb", "lpips")}
                 for v in VARIANTS if dv[v]}
            print(f"  {split:16s} {m:14s} PSNR " + " -> ".join(f"{s[v]['psnr_rgb']:.3f}" for v in s)
                  + " | LPIPS " + " -> ".join(f"{s[v]['lpips']:.4f}" for v in s)
                  + f" | xfit a0={np.round(xf.get('0', {}).get('clipped', []), 3).tolist()} "
                  f"a1={np.round(xf.get('1', {}).get('clipped', []), 3).tolist()} | chk dPSNR={c['max_abs_dpsnr_rgb']} "
                  f"dLPIPS={c['max_abs_dlpips']} ok={c['ok']} ({time.time() - t0:.0f}s)", flush=True)
            out["summary"] = summarize(out["per_seq"])
            out["wall_s"] = round(time.time() - t_all, 1)
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1)
        del data
    out["consistency_ok"] = all(c["ok"] for cm in out["consistency"].values() for c in cm.values())
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[SAVE] {args.out}  consistency_ok={out['consistency_ok']}  wall={time.time() - t_all:.0f}s", flush=True)
    print_table(out)


if __name__ == "__main__":
    main()
