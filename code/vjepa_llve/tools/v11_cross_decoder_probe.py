"""tools/v11_cross_decoder_probe.py — DV7.4 / DY1 attribution test.

Eval-only. Tests:
  C+ adapter (jointly trained with C+ decoder) × Probe C oracle decoder
                                                  (never saw C+ adapter outputs)

Question: is shuffled_predicted's high score a property of
  (A) the C+ adapter (it emits a generic restoration prior even with wrong z_L)
  or
  (B) the C+ decoder (it co-adapted to consume any plausible delta as brightness)
or
  (C) both.

Branches inform C++ design:
  A → emphasize NCE + adapter content-binding ranking
  B → emphasize delta dropout + decoder regularization
  C → both, but stay conservative

Spec: VJEPA_LLVE_CODE_REVIEW.md §DV7.4, §DY1.
"""
from __future__ import annotations
import argparse, csv, os, sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11 import JEPADeltaAdapter
from models.v11.scale_aware_adapter import ScaleAwareDeltaAdapter
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_oracle_delta_probe import (
    load_clip, to_u8, mean_psnr, encode, build_split_seqs,
)


@torch.no_grad()
def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    # Adapter: load from ckpt that was trained jointly with a different decoder
    base_adapter = JEPADeltaAdapter(n_layers=4, in_dim=1024, bot_dim=256, depth=4, heads=4)
    adapter = ScaleAwareDeltaAdapter(base_adapter).to(device)
    sd = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    if "adapter" in sd:
        adapter.load_state_dict(sd["adapter"])
    else:
        # Probe B' format only had base_adapter
        base_adapter.load_state_dict(sd["adapter"] if "adapter" in sd else sd)
        adapter = ScaleAwareDeltaAdapter(base_adapter).to(device)
    adapter.eval()
    for p in adapter.parameters(): p.requires_grad_(False)
    print(f"[adapter] loaded from {args.adapter_ckpt}")

    # Decoder: load from a DIFFERENT ckpt (potentially trained with different adapter)
    decoder = DeltaModulatedDecoder().to(device)
    sd = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    decoder.load_state_dict(sd["decoder"])
    decoder.eval()
    for p in decoder.parameters(): p.requires_grad_(False)
    print(f"[decoder] loaded from {args.decoder_ckpt}")

    train_seqs, test_seqs = build_split_seqs()
    print(f"[split] train={len(train_seqs)} held-out={len(test_seqs)}")

    rng_e = np.random.default_rng(args.seed)
    shuffle_audit = []

    def run_condition(condition: str):
        rows = []
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq)
            z_H = encode(backbone, gt)
            if condition == "identity_delta":
                kv = z_L; delta = [torch.zeros_like(z) for z in z_L]
            elif condition == "oracle_delta":
                kv = z_L; delta = [zh - zl for zh, zl in zip(z_H, z_L)]
            elif condition == "predicted_delta":
                kv = z_L; delta = adapter.predict_delta(z_L)
            elif condition == "shuffled_predicted_delta":
                other = int(rng_e.integers(0, len(train_seqs)))
                while other == 0 and len(train_seqs) > 1: pass  # any valid donor
                donor_name, lq_o, _ = train_seqs[other]
                shuffle_audit.append({"current_seq": name, "donor_seq": donor_name})
                lq_oo, _ = load_clip(lq_o, gt_p, crop_size=args.crop)
                z_L_o = encode(backbone, lq_oo.to(device))
                kv = z_L; delta = adapter.predict_delta(z_L_o)
            else:
                raise ValueError(condition)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder(lq, kv_layers=kv, delta_layers=delta)
            pred_u8, gt_u8, lq_u8 = to_u8(pred.float()), to_u8(gt), to_u8(lq)
            psnr = mean_psnr(pred_u8, gt_u8); psnr_lq = mean_psnr(lq_u8, gt_u8)
            rows.append({"seq": name, "lq_psnr": psnr_lq, "pred_psnr": psnr})
            print(f"    [{condition}/{name}] LQ={psnr_lq:.2f} pred={psnr:.3f}")
        mean = float(np.mean([r["pred_psnr"] for r in rows]))
        return mean, rows

    results = {}
    for cond in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
        print(f"\n--- {cond} ---")
        mean, rows = run_condition(cond)
        results[cond] = mean
        out = os.path.join(args.out_dir, cond)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "per_seq.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"  -> {cond}: mean PSNR = {mean:.3f}")

    # shuffle audit
    if shuffle_audit:
        with open(os.path.join(args.out_dir, "eval_shuffle_audit.csv"), "w") as f:
            f.write("current_seq,donor_seq\n")
            for r in shuffle_audit: f.write(f"{r['current_seq']},{r['donor_seq']}\n")

    # summary
    md = os.path.join(args.out_dir, "CROSS_DECODER_SUMMARY.md")
    with open(md, "w") as f:
        f.write("# v11 Cross-Decoder Probe (DV7.4 / DY1)\n\n")
        f.write(f"adapter: {args.adapter_ckpt}\n")
        f.write(f"decoder: {args.decoder_ckpt}\n")
        f.write(f"eval: crop={args.crop}, LLVE-STCD-12 ({len(test_seqs)} seqs), seq-disjoint held-out\n\n")
        f.write("| condition | mean PSNR | Δ vs identity |\n|---|---:|---:|\n")
        ident = results.get("identity_delta", float("nan"))
        for c in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
            d = results[c] - ident if not np.isnan(ident) else float("nan")
            f.write(f"| {c} | {results[c]:.3f} | {d:+.3f} |\n")
        f.write("\n## Attribution branch\n\n")
        if all(c in results for c in ["identity_delta","predicted_delta","shuffled_predicted_delta"]):
            d_pred = results["predicted_delta"] - results["identity_delta"]
            d_shuf = results["shuffled_predicted_delta"] - results["identity_delta"]
            ratio = d_shuf / max(d_pred, 1e-6)
            f.write(f"- predicted - identity = {d_pred:+.3f} dB\n")
            f.write(f"- shuffled - identity = {d_shuf:+.3f} dB\n")
            f.write(f"- leakage_ratio = {ratio:.2f}\n\n")
            if d_shuf > 1.0 and d_pred > 0.5:
                branch = "A (adapter-side leakage): C+ adapter emits content-weak/global action"
            elif d_shuf < 0.5:
                branch = "B (decoder-side co-adapt): C+ decoder collapsed to 'any-delta-helps'"
            elif d_pred < 0.5:
                branch = "C (both): C+ depends on adapter+decoder co-adaptation"
            else:
                branch = "intermediate — ambiguous; consider longer probe or InfoNCE direct test"
            f.write(f"**Branch: {branch}**\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/v11_cross_decoder_probe")
    ap.add_argument("--adapter_ckpt", default="/cache/zihaohe/checkpoint/v11_probes/v11_jdap_cplus_1000/ckpt.pt")
    ap.add_argument("--decoder_ckpt", default="/cache/zihaohe/checkpoint/v11_probes/probe_c_long_crop256/oracle_delta/ckpt.pt")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
