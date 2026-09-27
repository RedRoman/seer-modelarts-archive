"""Dump frames for the qualitative comparison figure (ICLR'27 Seer paper).

For each clip of a fixed list (all held out from Seer's training) this saves frame FRAME (0-based,
default 8) of the unified centre 16-frame 512x512 clip as PNG for: LQ, GT, every baseline
(Retinexformer, URCSA, DarkIR, HVI-CIDNet, STCD, Zero-DCE, SCI) and every --seer checkpoint
(raw output and "+CP-src"), plus <out>/qualitative.json with per-image PSNR/LPIPS of the dumped
frame vs GT.

Nothing is re-implemented:
  clips      run_temporal_all.SPLITS (= seer_eval_unified.iter_sdsd_indoor_seer for L0,
             eval_lib.iter_sdsd_outdoor for L1, eval_lib.iter_bvi for L2; T=16, 512 crop)
  baselines  run_temporal_all.build_methods (each run_*_v2.py loader + infer_fn, same per-domain
             weights: Retinexformer/URCSA indoor weights for L0 + BVI, outdoor weights for L1;
             HVI-CIDNet Generalization.bin; STCD indoor_G.pth via run_stcd_unified.STCD).
             Frame methods: run_temporal_all.run_frames (batch 1, frame by frame, all 16 frames).
             STCD: STCD.enhance_clip on the whole clip with chunk=1 (batch 1).
  Seer       seer_eval_unified.Seer (variant read from ckpt args), full 16-frame clip in one
             forward, then frame FRAME.  CP-src = seer_eval_unified.fit_alpha (closed-form alpha
             on the fixed SDSD *train* fit set) + cp_surrogate + fuse_per_channel, i.e. the
             "HMP+CPfit" row of seer_eval_unified.
  metrics    eval_lib.rgb_psnr_uint8 on (x*255).clip.astype(uint8) (same truncation as
             score_frames / run_frames; the PNG stores exactly this uint8 image) and
             eval_lib.lpips_batch on the float [0,1] frame.
Consistency check: the 16-frame clip mean PSNR/LPIPS of every row is also stored and compared with
the evaluated per-sequence numbers (baselines_temporal_unified_v2.json; for Seer optionally the
seer_eval_unified JSON given by --seer_ref).

Usage:
  python tools/dump_qualitative.py --out DIR \
      --seer "Seer=/path/final.pt" --seer "Seer w/o V-JEPA=/path/final.pt" \
      [--seer_ref "Seer=/path/eval_final.json"] [--force_no_vjepa "Seer w/o V-JEPA"]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

BASE = "/temp/zihaohe/workspace/baselines"
sys.path.insert(0, BASE)
import run_temporal_all as RT  # noqa: E402  (same import order as the baseline evals)
import seer_eval_unified as SE  # noqa: E402  (already imported by RT; same module object)
from common.eval_lib import rgb_psnr_uint8, lpips_batch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

EVDIR = "/temp/zihaohe/cache/checkpoint/runs/seer_iclr_evals"
BASELINE_REF = os.path.join(EVDIR, "baselines_temporal_unified_v2.json")
FRAME = 8
# (split key of run_temporal_all.SPLITS, sequence name)
CLIPS = [
    ("L0_SDSD-indoor", "pair60"),          # held out for Seer and for every released baseline ckpt
    ("L0_SDSD-indoor", "pair70"),          # idem
    ("L1_SDSD-outdoor", "MVI_1030"),       # Seer test split; also in Retinexformer-outdoor testing_dir
    ("L2_BVI-RLV", "S35_Bunny__10fps"),    # BVI-RLV test: toys + book cover, very dark LQ
    ("L2_BVI-RLV", "S37_Mario3__10fps"),   # BVI-RLV test: plush face + colour checker, grey wall
]
BASELINE_ORDER = ["Retinexformer", "URCSA", "DarkIR", "HVI-CIDNet", "STCD", "Zero-DCE", "SCI"]


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def to_u8(x):
    """(3,H,W) float [0,1] tensor -> (H,W,3) uint8, same truncation as score_frames/run_frames."""
    return (x.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def frame_metrics(pred, gt):
    """pred, gt: (T,3,H,W) float [0,1] on GPU -> per-frame psnr, lpips lists (score_frames recipe)."""
    ps, lp = [], []
    for t in range(pred.shape[0]):
        ps.append(rgb_psnr_uint8(to_u8(pred[t]), to_u8(gt[t])))
        lp.append(lpips_batch(pred[t:t + 1], gt[t:t + 1]))
    return ps, lp


def load_clips(clips):
    """Run the eval iterators once per split and keep the requested sequences (CPU uint8)."""
    want = {}
    for sp, name in clips:
        want.setdefault(sp, []).append(name)
    got = {}
    for sp, names in want.items():
        dom, it_fn = RT.SPLITS[sp]
        t0 = time.time()
        for name, lq_u8, gt_u8 in it_fn():
            if name in names:
                got[(sp, name)] = (dom, np.ascontiguousarray(lq_u8), np.ascontiguousarray(gt_u8))
        print(f"[data] {sp}: {[n for n in names if (sp, n) in got]} ({time.time() - t0:.0f}s)", flush=True)
    missing = [c for c in clips if c not in got]
    if missing:
        raise SystemExit(f"clips not found in their split iterators: {missing}")
    return got


class Dumper:
    def __init__(self, out, frame):
        self.out, self.frame = out, frame
        self.rec = {}

    def add(self, sp, name, method, pred_t, gt_t, png_u8=None, ref=None, extra=None):
        """pred_t, gt_t: (T,3,H,W) float [0,1] on GPU. Saves frame self.frame and its metrics."""
        f = self.frame
        ps, lp = frame_metrics(pred_t, gt_t)
        u8 = to_u8(pred_t[f]) if png_u8 is None else png_u8
        d = os.path.join(self.out, slug(name))
        os.makedirs(d, exist_ok=True)
        fn = f"{slug(method)}.png"
        imageio.imwrite(os.path.join(d, fn), u8)
        r = {"png": f"{slug(name)}/{fn}", "psnr_rgb": float(ps[f]), "lpips": float(lp[f]),
             "clip_psnr_rgb": float(np.mean(ps)), "clip_lpips": float(np.mean(lp)),
             "n_frames": len(ps)}
        if ref is not None:
            r["ref_clip_psnr_rgb"] = ref.get("psnr_rgb")
            r["ref_clip_lpips"] = ref.get("lpips")
            if ref.get("psnr_rgb") is not None:
                r["d_clip_psnr_vs_ref"] = r["clip_psnr_rgb"] - ref["psnr_rgb"]
        if extra:
            r.update(extra)
        self.rec.setdefault(name, {"split": sp, "methods": {}})["methods"][method] = r
        print(f"  {name:20s} {method:24s} frame{f}: PSNR={r['psnr_rgb']:.3f} LPIPS={r['lpips']:.4f} | "
              f"clip16 PSNR={r['clip_psnr_rgb']:.3f}"
              + (f" ref={ref['psnr_rgb']:.3f} (d={r['d_clip_psnr_vs_ref']:+.4f})"
                 if ref is not None and ref.get("psnr_rgb") is not None else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seer", action="append", default=[], metavar="LABEL=CKPT",
                    help="repeatable; e.g. --seer 'Seer=/p/final.pt' --seer 'Seer w/o V-JEPA=/q/final.pt'")
    ap.add_argument("--seer_ref", action="append", default=[], metavar="LABEL=EVAL_JSON",
                    help="optional seer_eval_unified JSON (or reference_v35stage4 JSON) for the clip-mean check")
    ap.add_argument("--force_no_vjepa", action="append", default=[], metavar="LABEL",
                    help="pass no_vjepa=True to Seer() for this label (only needed if the ckpt args lack it)")
    ap.add_argument("--clips", nargs="+", default=None, metavar="SPLIT:NAME",
                    help="override the clip list, e.g. L0_SDSD-indoor:pair60 L2_BVI-RLV:S35_Bunny__10fps")
    ap.add_argument("--frame", type=int, default=FRAME)
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()

    seers = []
    for s in args.seer:
        lab, ck = s.split("=", 1)
        seers.append((lab.strip(), ck.strip()))
    refs = {}
    for s in args.seer_ref:
        lab, p = s.split("=", 1)
        refs[lab.strip()] = json.load(open(p.strip()))
    clips = [tuple(c.split(":", 1)) for c in args.clips] if args.clips else CLIPS
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    t_all = time.time()
    data = load_clips(clips)
    dump = Dumper(args.out, args.frame)
    meta = {"frame_index": args.frame, "seer": {}, "baselines": {}}

    def gpu(lq_u8, gt_u8):
        lq = torch.from_numpy(lq_u8).float().div(255).permute(0, 3, 1, 2).to(dev)   # (T,3,H,W)
        gt = torch.from_numpy(gt_u8).float().div(255).permute(0, 3, 1, 2).to(dev)
        return lq, gt

    # ---- LQ / GT ----
    bref = json.load(open(BASELINE_REF))["per_seq"]
    for sp, name in clips:
        dom, lq_u8, gt_u8 = data[(sp, name)]
        lq, gt = gpu(lq_u8, gt_u8)
        dump.add(sp, name, "LQ", lq, gt, png_u8=lq_u8[args.frame], ref=bref[sp]["LQ"].get(name))
        dump.add(sp, name, "GT", gt, gt, png_u8=gt_u8[args.frame])
        dump.rec[name]["domain_weights"] = dom

    # ---- Seer checkpoints (one at a time, freed before the next) ----
    for lab, ck in seers:
        torch.cuda.reset_peak_memory_stats()
        model = SE.Seer(ck, lab in args.force_no_vjepa, dev)
        alpha_fit, alpha_raw, fit_names = SE.fit_alpha(model, dev)
        print(f"[{lab}] CP-src alpha (fit on {len(fit_names)} SDSD train clips) = {alpha_fit.tolist()}", flush=True)
        ref = refs.get(lab)
        for sp, name in clips:
            dom, lq_u8, gt_u8 = data[(sp, name)]
            with torch.no_grad():
                lq = torch.from_numpy(lq_u8).float().div(255).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
                gt = torch.from_numpy(gt_u8).float().div(255).permute(0, 3, 1, 2).to(dev)
                lq_p, H, W = SE.pad16(lq)
                pred = model(lq_p)[..., :H, :W]                    # (1,3,T,H,W), whole clip at once
                pred_t = pred[0].permute(1, 0, 2, 3).contiguous()   # (T,3,H,W)
                lq_t = lq[0].permute(1, 0, 2, 3).contiguous()
                cpf = SE.fuse_per_channel(pred_t, SE.cp_surrogate(pred_t, lq_t), alpha_fit)
            r_raw = r_cp = None
            if ref is not None:
                ps_ = ref["per_seq"].get(sp, {})
                r_raw = ps_.get("HMP-raw", {}).get(name)
                r_cp = ps_.get("HMP+CPfit", {}).get(name)
            dump.add(sp, name, lab, pred_t, gt, ref=r_raw)
            dump.add(sp, name, f"{lab}+CP-src", cpf, gt, ref=r_cp)
            del pred, pred_t, cpf, lq, lq_p
        meta["seer"][lab] = {"ckpt": ck, "variant": model.variant, "no_vjepa": model.no_vjepa,
                             "cp_src_alpha": {"clipped": alpha_fit.tolist(), "raw": alpha_raw,
                                              "fit_clips": fit_names},
                             "peak_mem_reserved_GB": torch.cuda.max_memory_reserved() / 2**30}
        print(f"[{lab}] peak reserved {meta['seer'][lab]['peak_mem_reserved_GB']:.1f} GB", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- baselines ----
    if not args.skip_baselines:
        methods = RT.build_methods(dev)
        for m in BASELINE_ORDER:
            meta["baselines"][m] = methods[m]["note"] + ("; run with chunk=1 (batch 1)" if m == "STCD" else "")
        for sp, name in clips:
            dom, lq_u8, gt_u8 = data[(sp, name)]
            lq, gt = gpu(lq_u8, gt_u8)
            for m in BASELINE_ORDER:
                spec = methods[m]
                with torch.no_grad():
                    if spec["kind"] == "frame":
                        pred, _, _ = RT.run_frames(spec[dom], lq_u8, gt_u8, dev)
                    else:
                        pred = spec[dom](lq, chunk=1)
                dump.add(sp, name, m, pred, gt, ref=bref[sp].get(m, {}).get(name))
                del pred
        del methods
        torch.cuda.empty_cache()

    out = {
        "protocol": ("frame %d (0-based) of the unified centre T=16, 512x512 clip (run_temporal_all.SPLITS "
                     "iterators); PSNR = eval_lib.rgb_psnr_uint8 on truncated uint8 (= the PNG), LPIPS = "
                     "eval_lib.lpips_batch (pyiqa alex) on float [0,1]; clip_* = mean over the 16 frames, "
                     "ref_clip_* = evaluated per-seq value" % args.frame),
        "clip_order": [n for _, n in clips],
        "method_order": ["LQ"] + [x for lab, _ in seers for x in (lab, lab + "+CP-src")]
                        + ([] if args.skip_baselines else BASELINE_ORDER) + ["GT"],
        "meta": meta,
        "clips": dump.rec,
        "wall_s": round(time.time() - t_all, 1),
    }
    with open(os.path.join(args.out, "qualitative.json"), "w") as f:
        json.dump(out, f, indent=1)
    worst = max((abs(r.get("d_clip_psnr_vs_ref", 0.0)), n, m) for n, c in dump.rec.items()
                for m, r in c["methods"].items())
    print(f"[SAVE] {os.path.join(args.out, 'qualitative.json')}  max|clip PSNR - ref| = {worst[0]:.4f} "
          f"({worst[1]}/{worst[2]})  wall {out['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
