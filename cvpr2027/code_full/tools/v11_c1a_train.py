"""tools/v11_c1a_train.py — C-1a Bottleneck-Conditioned Adapter trainer (DAI).

Setup (per DAI6):
  V-JEPA: frozen
  decoder: frozen V2 oracle decoder (delta_in_encoder=False)
  adapter: BottleneckCrossAttnAdapter (full N_v query tokens, 1 cross-attn block,
           layers 17/23 active only)

Loss:
  RGB Charbonnier through frozen V2 decoder
  + RGB ranking through frozen V2 decoder
  + InfoNCE on layers 17/23
  + light dir/mag/L2 latent regularizers

Eval gates (DAG6):
  predicted - identity >= +3.0
  shuffled - identity <= +0.5
  predicted - shuffled >= +1.5
  oracle - identity >= +10.0
  leakage_ratio <= 0.20

Spec: VJEPA_LLVE_CODE_REVIEW.md §DAG / §DAH / §DAI.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from models.v11.bottleneck_cross_attn_adapter import BottleneckCrossAttnAdapter
from tools.v11_oracle_delta_probe import (
    load_clip, to_u8, mean_psnr, encode, charbonnier, build_split_seqs,
)
from tools.v11_jdap_cplusplus import (
    descriptor, FIFOQueue, info_nce,
)
from tools.v11_jdap_rgbdistill import (
    latent_dir_loss, latent_mag_loss, latent_l2_loss,
)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})

    # --- frozen V-JEPA backbone ---
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print("[backbone] V-JEPA frozen")

    # --- frozen V2 oracle decoder (delta_in_encoder=False) ---
    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    sd = torch.load(args.decoder_init, map_location="cpu", weights_only=False)
    if "decoder" in sd: sd = sd["decoder"]
    decoder.load_state_dict(sd)
    for p in decoder.parameters(): p.requires_grad_(False)
    decoder.eval()
    print(f"[decoder] V2 oracle FROZEN; loaded from {args.decoder_init}")
    assert decoder.delta_in_encoder is False, "must use V2 decoder for C-1a"

    # --- trainable bottleneck cross-attn adapter ---
    adapter = BottleneckCrossAttnAdapter(
        bottleneck_channels=decoder.dims[-1],   # 256
        vjepa_grid=tuple(decoder.vjepa_grid),
        vjepa_dim=decoder.vjepa_dim,
        n_layers=4,
        active_layer_idx=tuple(int(x) for x in args.active_layer_idx.split(",")),
        attn_heads=args.attn_heads,
        attn_depth=args.attn_depth,
    ).to(device)
    n_a = adapter.num_trainable_params()
    print(f"[adapter] BottleneckCrossAttnAdapter trainable: {n_a/1e6:.2f}M, "
          f"active_layer_idx={adapter.active_layer_idx}, attn_depth={adapter.attn_depth}")

    # --- log shapes (DAI7) ---
    with torch.no_grad():
        lq_dummy = torch.rand(1, 3, 16, args.crop, args.crop, device=device)
        z_L_dummy = encode(backbone, lq_dummy)
        _, h_b_dummy = decoder(lq_dummy, kv_layers=z_L_dummy, delta_layers=None,
                                return_bottleneck=True)
        q_dummy = adapter._q_tokens(h_b_dummy)
        delta_dummy = adapter.predict_delta(h_b_dummy, z_L_dummy)
        print(f"[shapes] h_b={tuple(h_b_dummy.shape)}  "
              f"q_tokens={tuple(q_dummy.shape)}  "
              f"delta_layers={[tuple(d.shape) for d in delta_dummy]}")

    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr * 0.05)

    train_seqs, test_seqs = build_split_seqs()

    # NCE setup (per active layer)
    nce_layers = list(adapter.active_layer_idx)
    queues = {l: FIFOQueue(dim=decoder.vjepa_dim, max_size=args.nce_queue_size, device=device)
              for l in nce_layers} if args.use_nce else {}
    if args.use_nce:
        print(f"[nce] layers={nce_layers} T={args.nce_T} w={args.nce_w} "
              f"queue={args.nce_queue_size} warmup_until_iter={args.nce_warmup}")

    rng = np.random.default_rng(args.seed)
    log = []
    nce_history = []
    rank_active_history = deque(maxlen=100)
    t0 = time.time()
    iter_idx = 0

    while iter_idx < args.iters:
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.no_grad():
            z_L = encode(backbone, lq)
            z_H = encode(backbone, gt)
            # h_b is delta-independent under V2 (delta_in_encoder=False); compute once.
            h_b, skips = decoder.encode(lq, kv_layers=z_L, delta_layers=None)
        delta_oracle = [zh - zl for zh, zl in zip(z_H, z_L)]

        # student delta prediction (trainable)
        delta_pred = adapter.predict_delta(h_b, z_L)

        # decoder forward (frozen) using predicted delta
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=z_L, delta_layers=delta_pred)
        l_good = charbonnier(pred.float(), gt)

        # ranking: shuffled-donor delta through frozen decoder
        L_rank = torch.zeros((), device=device)
        rank_active_this_step = 0.0
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
                pred_wrong = decoder.decode(h_b, skips, lq, kv_layers=z_L,
                                            delta_layers=delta_wrong)
            l_wrong = charbonnier(pred_wrong.float(), gt)
            L_rank = F.relu(args.ranking_margin + l_good - l_wrong)
            rank_active_this_step = float((L_rank > 0).item())
        rank_active_history.append(rank_active_this_step)

        # latent regularizers (only on active layers)
        active = adapter.active_layer_idx
        L_dir = sum(latent_dir_loss(delta_pred[l], delta_oracle[l]) for l in active) / len(active)
        L_mag = sum(latent_mag_loss(delta_pred[l], delta_oracle[l]) for l in active) / len(active)
        L_l2  = sum(latent_l2_loss(delta_pred[l],  delta_oracle[l]) for l in active) / len(active)

        # InfoNCE on active layers
        L_nce = torch.zeros((), device=device)
        nce_per_layer_this_step = {}
        if args.use_nce and iter_idx >= args.nce_warmup:
            terms = []
            for l in nce_layers:
                z_q = descriptor(delta_pred[l])
                z_pos = descriptor(delta_oracle[l])
                q = queues[l].get()
                term = info_nce(z_q, z_pos, q, T=args.nce_T)
                terms.append(term)
                nce_per_layer_this_step[l] = float(term.item())
                queues[l].push(z_pos)
            L_nce = sum(terms) / max(1, len(terms))
        elif args.use_nce:
            with torch.no_grad():
                for l in nce_layers:
                    queues[l].push(descriptor(delta_oracle[l]))

        L = (args.rgb_weight * l_good
             + args.dir_weight * L_dir
             + args.mag_weight * L_mag
             + args.l2_weight * L_l2
             + args.ranking_weight * L_rank
             + args.nce_w * L_nce)

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optim.step(); sched.step()

        if (iter_idx + 1) % args.log_every == 0:
            rank_frac = float(np.mean(rank_active_history)) if rank_active_history else 0.0
            print(f"  it {iter_idx+1}/{args.iters} L={L.item():.4f} "
                  f"(rgb={l_good.item():.4f} dir={L_dir.item():.3f} mag={L_mag.item():.3f} "
                  f"l2={L_l2.item():.3f} rank={L_rank.item():.3f} nce={L_nce.item():.4f}) "
                  f"rank_active_frac={rank_frac:.2f} "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            log.append({"iter": iter_idx + 1, "L": L.item(), "L_rgb": l_good.item(),
                        "L_dir": L_dir.item(), "L_mag": L_mag.item(),
                        "L_l2": L_l2.item(), "L_rank": L_rank.item(),
                        "L_nce": L_nce.item(),
                        "rank_active_frac": rank_frac,
                        "lr": optim.param_groups[0]["lr"]})
            if nce_per_layer_this_step:
                nce_history.append({"iter": iter_idx + 1,
                                    **{f"L{l}": v for l, v in nce_per_layer_this_step.items()}})
            t0 = time.time()
        iter_idx += 1

    torch.save({"adapter": adapter.state_dict(),
                "args": vars(args), "log": log, "iter": iter_idx},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved adapter ({n_a/1e6:.2f}M) to {args.out_dir}/ckpt.pt")

    if nce_history:
        with open(os.path.join(args.out_dir, "nce_history.csv"), "w") as f:
            keys = sorted({k for r in nce_history for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in nce_history: w.writerow(r)

    # ---- full eval ----
    print("\n=== FULL EVAL on LLVE-STCD-12 held-out (crop=256) ===")
    adapter.eval()
    rng_e = np.random.default_rng(args.seed + 1)
    audit_pairs = []
    results = {}
    per_seq = {}
    for cond in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
        rows = []
        with torch.no_grad():
            for name, lq_p, gt_p in test_seqs:
                lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
                lq, gt = lq.to(device), gt.to(device)
                z_L = encode(backbone, lq)
                z_H = encode(backbone, gt)
                h_b, skips = decoder.encode(lq, kv_layers=z_L, delta_layers=None)
                if cond == "identity_delta":
                    delta = [torch.zeros_like(z) for z in z_L]
                elif cond == "oracle_delta":
                    delta = [zh - zl for zh, zl in zip(z_H, z_L)]
                elif cond == "predicted_delta":
                    delta = adapter.predict_delta(h_b, z_L)
                else:
                    other = int(rng_e.integers(0, len(train_seqs)))
                    donor_name, lq_o, gt_o = train_seqs[other]
                    assert donor_name != name, "shuffle audit: donor must differ"
                    audit_pairs.append({"current_seq": name, "donor_seq": donor_name})
                    lq_oo, _ = load_clip(lq_o, gt_o, crop_size=args.crop)
                    z_L_o = encode(backbone, lq_oo.to(device))
                    # IMPORTANT: also need h_b for the donor — but we eval against
                    # a frozen decoder, and adapter.predict_delta is conditioned on
                    # h_b derived from CURRENT lq's decoder forward. Use current h_b
                    # with donor z_L_o so the adapter sees inconsistent inputs (the
                    # killer signal).
                    delta = adapter.predict_delta(h_b, z_L_o)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = decoder.decode(h_b, skips, lq, kv_layers=z_L, delta_layers=delta)
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

    # write audit
    if audit_pairs:
        with open(os.path.join(args.out_dir, "eval_shuffle_audit.csv"), "w") as f:
            f.write("current_seq,donor_seq\n")
            for r in audit_pairs: f.write(f"{r['current_seq']},{r['donor_seq']}\n")

    # ---- summary ----
    md = os.path.join(args.out_dir, "C1A_SUMMARY.md")
    ident = results["identity_delta"]
    pred_g = results["predicted_delta"] - ident
    shuf_g = results["shuffled_predicted_delta"] - ident
    oracle_g = results["oracle_delta"] - ident
    sep = pred_g - shuf_g
    leak = shuf_g / max(1e-6, pred_g)
    with open(md, "w") as f:
        f.write("# v11 C-1a Bottleneck-Conditioned Adapter Results (DAI)\n\n")
        f.write(f"adapter: {n_a/1e6:.2f}M | decoder: V2 frozen oracle (33.92M, no grad)\n")
        f.write(f"iters: {args.iters} | lr: {args.lr} | crop: {args.crop}\n")
        f.write(f"active_layer_idx: {adapter.active_layer_idx} | attn_depth: {adapter.attn_depth}\n")
        f.write(f"ranking: w={args.ranking_weight} margin={args.ranking_margin}\n")
        f.write(f"NCE: layers={nce_layers} T={args.nce_T} w={args.nce_w} queue={args.nce_queue_size} warmup_iter={args.nce_warmup}\n\n")
        f.write("| condition | mean PSNR | Δ vs identity |\n|---|---:|---:|\n")
        for c in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
            d = results[c] - ident
            f.write(f"| {c} | {results[c]:.3f} | {d:+.3f} |\n")
        f.write("\n## Gates (DAG6)\n\n")
        f.write(f"- C1-B1 usefulness (predicted - identity ≥ +3.0):  {pred_g:+.3f} → **{'PASS' if pred_g >= 3.0 else 'FAIL'}**\n")
        f.write(f"- C1-B2 killer (shuffled - identity ≤ +0.5):        {shuf_g:+.3f} → **{'PASS' if shuf_g <= 0.5 else 'FAIL'}**\n")
        f.write(f"- C1-B3 separation (predicted - shuffled ≥ +1.5):   {sep:+.3f} → **{'PASS' if sep >= 1.5 else 'FAIL'}**\n")
        f.write(f"- C1-B4 oracle sanity (oracle - identity ≥ +10.0):  {oracle_g:+.3f} → **{'PASS' if oracle_g >= 10.0 else 'FAIL'}**\n")
        f.write(f"- C1-B5 leakage ratio (≤ 0.20):                     {leak:.3f} → **{'PASS' if leak <= 0.20 else 'FAIL'}**\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/v11_c1a_bottleneck_adapter")
    ap.add_argument("--decoder_init", default="/cache/zihaohe/checkpoint/v11_probes/probe_c_v2_oracle/oracle_delta/ckpt.pt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--dir_weight", type=float, default=0.05)
    ap.add_argument("--mag_weight", type=float, default=0.02)
    ap.add_argument("--l2_weight", type=float, default=0.01)
    ap.add_argument("--use_ranking", action="store_true", default=True)
    ap.add_argument("--ranking_margin", type=float, default=0.05)
    ap.add_argument("--ranking_weight", type=float, default=1.0)
    ap.add_argument("--use_nce", action="store_true", default=True)
    ap.add_argument("--nce_T", type=float, default=0.07)
    ap.add_argument("--nce_w", type=float, default=0.1)
    ap.add_argument("--nce_queue_size", type=int, default=32)
    ap.add_argument("--nce_warmup", type=int, default=32)
    ap.add_argument("--active_layer_idx", default="2,3", help="DAI5: 2,3 = V-JEPA layers 17,23")
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--attn_depth", type=int, default=1, help="DAI4: 1 block MVP")
    main(ap.parse_args())
