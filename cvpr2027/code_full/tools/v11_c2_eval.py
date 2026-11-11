"""tools/v11_c2_eval.py — C-2-Direct full-condition eval (DAS3/DAS5/DAT5).

Loads a Stage 1 or Stage 2 ckpt, runs all conditions on LLVE-STCD-12 held-out,
reports primary + anti-cheat + diagnostic gates.

Conditions per DAS3:
  no_prior
  passive_zL
  adapted_zL
  shuffled_adapted
  zero_zL_hb_only          (DAT5: collapsed; same eval as h_b_only_adapter)
  gt_vjepa_diag            (DAR8: report only, never gate as oracle)

Spec: VJEPA_LLVE_CODE_REVIEW.md §DAS5 / §DAT5.
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from typing import Dict, List

import numpy as np
import torch
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from models.v11.content_attn_adapter import ContentAttnAdapter
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           build_split_seqs)


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device).eval()
    adapter = ContentAttnAdapter(
        bottleneck_channels=decoder.dims[-1],
        vjepa_grid=tuple(decoder.vjepa_grid),
        vjepa_dim=decoder.vjepa_dim,
        active_scales=tuple(int(s) for s in args.active_scales.split(",")),
        ref_layer=args.ref_layer,
        attn_heads=args.attn_heads,
        ffn_mult=args.ffn_mult,
    ).to(device).eval()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stage = sd.get("stage", "?")
    decoder.load_state_dict(sd["decoder"])
    adapter.load_state_dict(sd["adapter"])
    print(f"[ckpt] loaded stage={stage} from {args.ckpt}")
    print(f"[adapter] {adapter.num_trainable_params()/1e6:.2f}M, active_scales={adapter.active_scales}")

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)

    conditions = ["no_prior", "passive_zL", "adapted_zL",
                  "shuffled_adapted", "zero_zL_hb_only", "gt_vjepa_diag"]
    results = {}
    per_seq = {}
    audit_pairs = []

    for cond in conditions:
        rows = []
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq)
            # DAT2: V-JEPA-clean h_b/skips
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)

            if cond == "no_prior":
                kv_layers, kv_override = None, None
            elif cond == "passive_zL":
                kv_layers, kv_override = z_L, None
            elif cond == "adapted_zL":
                kv_layers, kv_override = None, adapter(z_L, h_b)
            elif cond == "shuffled_adapted":
                other = int(rng.integers(0, len(train_seqs)))
                donor_name, lq_o, gt_o = train_seqs[other]
                assert donor_name != name
                audit_pairs.append({"current_seq": name, "donor_seq": donor_name})
                lq_oo, _ = load_clip(lq_o, gt_o, crop_size=args.crop)
                z_L_o = encode(backbone, lq_oo.to(device))
                kv_layers, kv_override = None, adapter(z_L_o, h_b)
            elif cond == "zero_zL_hb_only":
                z_L_zeros = [torch.zeros_like(z) for z in z_L]
                kv_layers, kv_override = None, adapter(z_L_zeros, h_b)
            elif cond == "gt_vjepa_diag":
                z_H = encode(backbone, gt)
                kv_layers, kv_override = None, adapter(z_H, h_b)
            else:
                raise ValueError(cond)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq,
                                      kv_layers=kv_layers, delta_layers=None,
                                      kv_override=kv_override)
            psnr = mean_psnr(to_u8(pred.float()), to_u8(gt))
            psnr_lq = mean_psnr(to_u8(lq), to_u8(gt))
            rows.append({"seq": name, "lq_psnr": psnr_lq, "pred_psnr": psnr})
            print(f"    [{cond}/{name}] LQ={psnr_lq:.2f} pred={psnr:.3f}")
        per_seq[cond] = rows
        results[cond] = float(np.mean([r["pred_psnr"] for r in rows]))
        print(f"  -> {cond}: mean PSNR = {results[cond]:.3f}")

    # write per-condition CSVs
    for cond, rows in per_seq.items():
        cd = os.path.join(args.out_dir, cond); os.makedirs(cd, exist_ok=True)
        with open(os.path.join(cd, "per_seq.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
    if audit_pairs:
        with open(os.path.join(args.out_dir, "eval_shuffle_audit.csv"), "w") as f:
            f.write("current_seq,donor_seq\n")
            for r in audit_pairs: f.write(f"{r['current_seq']},{r['donor_seq']}\n")

    # gates
    no_prior   = results["no_prior"]
    passive    = results["passive_zL"]
    adapted    = results["adapted_zL"]
    shuffled   = results["shuffled_adapted"]
    zerov      = results["zero_zL_hb_only"]
    gt_diag    = results["gt_vjepa_diag"]

    s1_apass = adapted - passive  >= 2.0
    s1_anpr  = adapted - no_prior >= 4.0
    s2_apass = adapted - passive   >= 1.0
    s2_anpr  = adapted - no_prior  >= 2.0
    s2_sep   = adapted - shuffled  >= 1.5
    s2_skill = (shuffled - passive) <= 0.5
    s2_anti  = adapted - zerov    >= 1.5

    md = os.path.join(args.out_dir, "C2_EVAL_SUMMARY.md")
    with open(md, "w") as f:
        f.write(f"# C-2-Direct Eval (stage={stage})\n\n")
        f.write(f"ckpt: {args.ckpt}\n")
        f.write(f"adapter: {adapter.num_trainable_params()/1e6:.2f}M\n")
        f.write(f"active_scales: {adapter.active_scales}\n\n")
        f.write("| condition | mean PSNR |\n|---|---:|\n")
        for c in conditions:
            f.write(f"| {c} | {results[c]:.3f} |\n")
        f.write("\n## Stage 1 gates (DAS4)\n\n")
        f.write(f"- adapted - passive ≥ +2.0:   {adapted-passive:+.3f}  → **{'PASS' if s1_apass else 'FAIL'}**\n")
        f.write(f"- adapted - no_prior ≥ +4.0:  {adapted-no_prior:+.3f}  → **{'PASS' if s1_anpr else 'FAIL'}**\n")
        f.write("\n## Stage 2 primary gates (DAS5)\n\n")
        f.write(f"- adapted - passive ≥ +1.0:   {adapted-passive:+.3f}  → **{'PASS' if s2_apass else 'FAIL'}**\n")
        f.write(f"- adapted - no_prior ≥ +2.0:  {adapted-no_prior:+.3f}  → **{'PASS' if s2_anpr else 'FAIL'}**\n")
        f.write(f"- adapted - shuffled ≥ +1.5:  {adapted-shuffled:+.3f}  → **{'PASS' if s2_sep else 'FAIL'}**\n")
        f.write(f"- shuffled - passive ≤ +0.5:  {shuffled-passive:+.3f}  → **{'PASS' if s2_skill else 'FAIL'}**\n")
        f.write("\n## Anti-cheat gate (DAR6/DAT5)\n\n")
        f.write(f"- adapted - zero_zL_hb_only ≥ +1.5: {adapted-zerov:+.3f}  → **{'PASS' if s2_anti else 'FAIL'}**\n")
        f.write("\n## Diagnostic (DAR8, NOT gated)\n\n")
        f.write(f"- gt_vjepa_diag = {gt_diag:.3f}  (Δ vs adapted_zL = {gt_diag-adapted:+.3f})\n")

    print(f"\n=== SUMMARY ===")
    for c in conditions: print(f"  {c:25s} {results[c]:.3f}")
    print(f"\n  Stage 1 gates: adp-pass={adapted-passive:+.2f} ({'P' if s1_apass else 'F'}), adp-none={adapted-no_prior:+.2f} ({'P' if s1_anpr else 'F'})")
    print(f"  Stage 2 gates: adp-pass={adapted-passive:+.2f} ({'P' if s2_apass else 'F'}), adp-none={adapted-no_prior:+.2f} ({'P' if s2_anpr else 'F'}), "
          f"sep={adapted-shuffled:+.2f} ({'P' if s2_sep else 'F'}), killer={shuffled-passive:+.2f} ({'P' if s2_skill else 'F'}), "
          f"anti-cheat={adapted-zerov:+.2f} ({'P' if s2_anti else 'F'})")
    print(f"  Diagnostic: gt_vjepa_diag {gt_diag:.3f} (Δ vs adapted = {gt_diag-adapted:+.3f})")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--active_scales", default="3,4")
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--ffn_mult", type=float, default=1.0)
    main(ap.parse_args())
