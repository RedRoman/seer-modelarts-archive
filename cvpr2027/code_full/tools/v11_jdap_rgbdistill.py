"""tools/v11_jdap_rgbdistill.py — DP6 main path: RGB-aligned JDAP student training.

Trains:
  - ScaleAwareDeltaAdapter (init from probe_b_prime, fully trainable)
  - DeltaModulatedDecoder   (init from probe_c oracle_delta if available, trainable)

with:
  - frozen V-JEPA encoder
  - scheduled action forcing m: 1.0 → 0.0 over args.iters
    delta_train = m * delta_oracle + (1-m) * stopgrad(delta_pred)
  - RGB-first loss
  - ranking loss with shuffled negative
  - per-layer gain+confidence priors (Probe A evidence)

Eval at end on LLVE-STCD-12 (sequence-disjoint held-out, crop=256):
  identity_delta
  oracle_delta
  predicted_delta              ← v11 main result
  shuffled_predicted_delta

Pass criteria (DP6):
  predicted - identity ≥ +0.7 dB
  recovery ≥ 30% of (oracle - identity)
  shuffled_predicted ≤ identity

Spec: VJEPA_LLVE_CODE_REVIEW.md §DP6.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.common import read_image_01
from models.vjepa_backbone import build_vjepa_backbone
from models.v11 import JEPADeltaAdapter
from models.v11.scale_aware_adapter import ScaleAwareDeltaAdapter
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_delta_stats import (
    LLVE_STCD_12, INDOOR_ROOT, OUTDOOR_ROOT, index_seqs,
)
from tools.v11_oracle_delta_probe import (
    load_clip, to_u8, mean_psnr, encode, charbonnier, build_split_seqs,
)


# ----- losses --------------------------------------------------------
def latent_dir_loss(d_pred, d_true):
    return (1.0 - F.cosine_similarity(d_pred, d_true, dim=-1)).mean()

def latent_mag_loss(d_pred, d_true):
    n_p = d_pred.norm(dim=-1).mean()
    n_t = d_true.norm(dim=-1).mean().clamp_min(1e-8)
    return (n_p / n_t - 1.0).abs()

def latent_l2_loss(d_pred, d_true):
    return ((d_pred - d_true).pow(2).mean() / d_true.pow(2).mean().clamp_min(1e-8)).sqrt()


# ----- scheduled action forcing schedule -----------------------------
def m_schedule(iter_idx: int, total: int, mode: str = "linear",
                rng: Optional[np.random.Generator] = None) -> float:
    """m schedule for action forcing.
    Modes:
      linear: 1.0 at iter 0 -> 0.0 at iter total
      linear_floor: 1.0 -> 0.3 (preserves oracle path; DS C+)
      cosine: 1.0 -> 0.0 cosine anneal
      random_mix: random per-batch from {1.0, 0.7, 0.3} (DT2 corrected C+)
    """
    if mode == "random_mix":
        choices = [1.0, 0.7, 0.3]
        if rng is None: rng = np.random.default_rng(iter_idx)
        return float(rng.choice(choices))
    p = min(1.0, iter_idx / max(1, total))
    if mode == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if mode == "linear_floor":
        return 1.0 - 0.7 * p   # 1.0 -> 0.3
    return 1.0 - p   # linear (DR default)


# ----- main train + eval ---------------------------------------------
def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print("[backbone] V-JEPA frozen")

    # Adapter: init from probe_b_prime, then wrap with scale-aware
    base_adapter = JEPADeltaAdapter(n_layers=4, in_dim=1024, bot_dim=256, depth=4, heads=4)
    if args.adapter_init and os.path.isfile(args.adapter_init):
        sd = torch.load(args.adapter_init, map_location="cpu", weights_only=False)
        base_adapter.load_state_dict(sd["adapter"])
        print(f"[adapter] base init from {args.adapter_init}")
    adapter = ScaleAwareDeltaAdapter(base_adapter).to(device)
    n_a = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[adapter] ScaleAwareDeltaAdapter trainable: {n_a/1e6:.2f}M")

    # Decoder: optional init from oracle_delta probe_c ckpt
    decoder = DeltaModulatedDecoder().to(device)
    if args.decoder_init and os.path.isfile(args.decoder_init):
        sd = torch.load(args.decoder_init, map_location="cpu", weights_only=False)
        decoder.load_state_dict(sd["decoder"])
        print(f"[decoder] init from {args.decoder_init}")
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[decoder] DeltaModulatedDecoder trainable: {n_d/1e6:.2f}M")

    # Optim
    optim = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    print(f"[split] train={len(train_seqs)} held-out={len(test_seqs)} disjoint=YES")

    # ----- training ------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()
    iter_idx = 0
    while iter_idx < args.iters:
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        # encode
        with torch.no_grad():
            z_L = encode(backbone, lq)
            z_H = encode(backbone, gt)
        delta_oracle = [zh - zl for zh, zl in zip(z_H, z_L)]

        # student delta
        delta_pred = adapter.predict_delta(z_L)

        # scheduled action forcing
        m = m_schedule(iter_idx, args.iters, args.m_anneal, rng=rng)
        delta_train = [m * d_o + (1.0 - m) * d_p.detach() for d_o, d_p in zip(delta_oracle, delta_pred)]

        # decoder forward (good)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder(lq, kv_layers=z_L, delta_layers=delta_train)

        # ranking: shuffled-donor delta
        L_rank = torch.zeros((), device=device)
        if args.use_ranking and args.ranking_weight > 0:
            other = int(rng.integers(0, len(train_seqs)))
            tries = 0
            while other == idx and tries < 5:
                other = int(rng.integers(0, len(train_seqs))); tries += 1
            _, lq_o, gt_o = train_seqs[other]
            lq_oo, gt_oo = load_clip(lq_o, gt_o, crop_size=args.crop)
            with torch.no_grad():
                z_L_o = encode(backbone, lq_oo.to(device))
                z_H_o = encode(backbone, gt_oo.to(device))
            delta_wrong = [zh_o - zl_o for zh_o, zl_o in zip(z_H_o, z_L_o)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_wrong = decoder(lq, kv_layers=z_L, delta_layers=delta_wrong)
            l_good = charbonnier(pred.float(), gt)
            l_wrong = charbonnier(pred_wrong.float(), gt)
            L_rank = F.relu(args.ranking_margin + l_good - l_wrong)
        else:
            l_good = charbonnier(pred.float(), gt)

        # latent regularizers (per layer)
        L_dir = sum(latent_dir_loss(d_p, d_o) for d_p, d_o in zip(delta_pred, delta_oracle)) / 4.0
        L_mag = sum(latent_mag_loss(d_p, d_o) for d_p, d_o in zip(delta_pred, delta_oracle)) / 4.0
        L_l2  = sum(latent_l2_loss(d_p, d_o)  for d_p, d_o in zip(delta_pred, delta_oracle)) / 4.0

        L = (args.rgb_weight * l_good
             + args.dir_weight * L_dir
             + args.mag_weight * L_mag
             + args.l2_weight * L_l2
             + args.ranking_weight * L_rank)

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(decoder.parameters()), 1.0)
        optim.step(); sched.step()

        if (iter_idx + 1) % args.log_every == 0:
            print(f"  it {iter_idx+1}/{args.iters} L={L.item():.4f} "
                  f"(rgb={l_good.item():.4f} dir={L_dir.item():.3f} mag={L_mag.item():.3f} "
                  f"l2={L_l2.item():.3f} rank={L_rank.item():.3f}) "
                  f"m={m:.2f} lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            print(f"    [scale] {adapter.gain_confidence_summary()}")
            log.append({"iter": iter_idx+1, "L": L.item(), "L_rgb": l_good.item(),
                        "L_dir": L_dir.item(), "L_mag": L_mag.item(),
                        "L_l2": L_l2.item(), "L_rank": L_rank.item(), "m": m,
                        "lr": optim.param_groups[0]["lr"]})
            t0 = time.time()
        iter_idx += 1

    # save
    torch.save({"adapter": adapter.state_dict(),
                "decoder": decoder.state_dict(),
                "args": vars(args), "log": log},
               os.path.join(args.out_dir, "ckpt.pt"))

    # ----- eval ----------------------------------------------------------
    print("\n=== EVAL on LLVE-STCD-12 held-out (crop=256) ===")
    adapter.eval(); decoder.eval()
    _shuffle_audit_rows = []  # DU4 eval-time donor pair audit

    @torch.no_grad()
    def run_condition(condition: str, rng_e: np.random.Generator):
        rows = []
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq); z_H = encode(backbone, gt)

            if condition == "identity_delta":
                kv = z_L; delta = [torch.zeros_like(z) for z in z_L]
            elif condition == "oracle_delta":
                kv = z_L; delta = [zh - zl for zh, zl in zip(z_H, z_L)]
            elif condition == "predicted_delta":
                kv = z_L; delta = adapter.predict_delta(z_L)
            elif condition == "shuffled_predicted_delta":
                # adapter's prediction from a different sequence
                other = int(rng_e.integers(0, len(train_seqs)))
                donor_name, lq_o, _ = train_seqs[other]
                # DU4: log eval-time donor pair
                assert donor_name != name, "DU4 audit: eval donor must differ"
                _shuffle_audit_rows.append({"current_seq": name, "donor_seq": donor_name})
                lq_oo, _ = load_clip(lq_o, gt_p, crop_size=args.crop)  # only need lq_o for z_L_o
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

    rng_e = np.random.default_rng(args.seed + 1)
    results = {}
    for cond in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
        print(f"\n--- {cond} ---")
        mean, rows = run_condition(cond, rng_e)
        results[cond] = mean
        cond_dir = os.path.join(args.out_dir, cond)
        os.makedirs(cond_dir, exist_ok=True)
        with open(os.path.join(cond_dir, "per_seq.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"  -> {cond}: mean PSNR = {mean:.3f}")

    # DU4: write eval shuffle audit
    if _shuffle_audit_rows:
        audit_path = os.path.join(args.out_dir, "eval_shuffle_audit.csv")
        with open(audit_path, "w") as f:
            f.write("current_seq,donor_seq\n")
            for r in _shuffle_audit_rows:
                f.write(f"{r['current_seq']},{r['donor_seq']}\n")
        print(f"wrote eval shuffle audit: {audit_path} ({len(_shuffle_audit_rows)} pairs)")

    # summary
    md = os.path.join(args.out_dir, "JDAP_RGBDISTILL_SUMMARY.md")
    with open(md, "w") as f:
        f.write("# v11 JDAP-RGB Distill Results (DP6)\n\n")
        f.write(f"adapter trainable: {n_a/1e6:.2f}M | decoder trainable: {n_d/1e6:.2f}M\n")
        f.write(f"iters: {args.iters} | lr: {args.lr} | crop: {args.crop}\n")
        f.write(f"m anneal: {args.m_anneal}, ranking: {args.use_ranking} (margin={args.ranking_margin}, w={args.ranking_weight})\n")
        f.write(f"loss weights: rgb={args.rgb_weight} dir={args.dir_weight} mag={args.mag_weight} l2={args.l2_weight}\n\n")
        f.write("| condition | mean PSNR | Δ vs identity |\n|---|---:|---:|\n")
        ident = results.get("identity_delta", float("nan"))
        for c in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
            d = results[c] - ident if not np.isnan(ident) else float("nan")
            f.write(f"| {c} | {results[c]:.3f} | {d:+.3f} |\n")
        f.write("\n## DP6 gates\n\n")
        if "predicted_delta" in results and "identity_delta" in results:
            pi = results["predicted_delta"] - results["identity_delta"]
            f.write(f"- predicted - identity = {pi:+.3f} dB (gate ≥ +0.7) → **{'PASS' if pi>=0.7 else 'FAIL'}**\n")
        if all(c in results for c in ["predicted_delta","oracle_delta","identity_delta"]):
            full = results["oracle_delta"] - results["identity_delta"]
            pred = results["predicted_delta"] - results["identity_delta"]
            rec = pred / full if abs(full) > 1e-6 else float("nan")
            f.write(f"- recovery = {rec*100:.0f}% of oracle gain (gate ≥ 30%) → **{'PASS' if rec>=0.30 else 'FAIL'}**\n")
        if "shuffled_predicted_delta" in results and "identity_delta" in results:
            si = results["shuffled_predicted_delta"] - results["identity_delta"]
            f.write(f"- shuffled_predicted - identity = {si:+.3f} dB (should ≤ 0) → **{'PASS' if si<=0.0 else 'FAIL'}**\n")
        f.write(f"\nFinal scale-aware adapter: {adapter.gain_confidence_summary()}\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/v11_jdap_rgbdistill_500")
    ap.add_argument("--adapter_init", default="/cache/zihaohe/checkpoint/v11_probes/probe_b_prime/delta_adapter.pt")
    ap.add_argument("--decoder_init", default="/cache/zihaohe/checkpoint/v11_probes/probe_c_long_crop256/oracle_delta/ckpt.pt")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m_anneal", default="linear",
                    choices=["linear", "linear_floor", "cosine", "random_mix"])
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--dir_weight", type=float, default=0.05)
    ap.add_argument("--mag_weight", type=float, default=0.02)
    ap.add_argument("--l2_weight", type=float, default=0.01)
    ap.add_argument("--use_ranking", action="store_true", default=True)
    ap.add_argument("--ranking_margin", type=float, default=0.005)
    ap.add_argument("--ranking_weight", type=float, default=0.1)
    main(ap.parse_args())
