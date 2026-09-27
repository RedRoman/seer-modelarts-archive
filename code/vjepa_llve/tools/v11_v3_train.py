"""tools/v11_v3_train.py — V3-B shuffle-adversarial decoder scaffold (DAN9).

Architecture: DeltaModulatedDecoder(delta_in_encoder=False) (V2-style base).
Loss:
  L_oracle = Charbonnier(D(LQ, z_L, delta_oracle), HQ)
  L_shuf   = Charbonnier(D(LQ, z_L, delta_shuf),   HQ)
  L_adv    = ReLU(margin + L_oracle - L_shuf)            # decoder must FAIL on shuf
  L        = L_oracle + lambda_adv * L_adv

DAN5 gate (eval in matched delta_in_encoder=False regime):
  oracle - identity >= +10.0 → PASS
  shuffled - identity <= +0.5 → PASS
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print("[backbone] V-JEPA frozen")

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[decoder] V3-B (V2-style, delta_in_encoder=False), trainable={n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()

    rng = np.random.default_rng(args.seed)
    log = []
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
        delta_oracle = [zh - zl for zh, zl in zip(z_H, z_L)]

        # shuffled donor delta from a different sequence
        other = int(rng.integers(0, len(train_seqs)))
        tries = 0
        while other == idx and tries < 5:
            other = int(rng.integers(0, len(train_seqs))); tries += 1
        _, lq_o, gt_o = train_seqs[other]
        lq_oo, gt_oo = load_clip(lq_o, gt_o, crop_size=args.crop)
        with torch.no_grad():
            z_L_o = encode(backbone, lq_oo.to(device))
            z_H_o = encode(backbone, gt_oo.to(device))
        delta_shuf = [zh_o - zl_o for zh_o, zl_o in zip(z_H_o, z_L_o)]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_oracle = decoder(lq, kv_layers=z_L, delta_layers=delta_oracle)
            pred_shuf   = decoder(lq, kv_layers=z_L, delta_layers=delta_shuf)
        l_oracle = charbonnier(pred_oracle.float(), gt)
        l_shuf   = charbonnier(pred_shuf.float(),   gt)
        l_adv    = F.relu(args.margin + l_oracle - l_shuf)

        L = args.lambda_oracle * l_oracle + args.lambda_adv * l_adv

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step(); sched.step()

        if (iter_idx + 1) % args.log_every == 0:
            adv_active = float((l_adv > 0).item())
            gap = (l_shuf - l_oracle).item()
            print(f"  it {iter_idx+1}/{args.iters} L={L.item():.4f} "
                  f"(L_o={l_oracle.item():.4f} L_s={l_shuf.item():.4f} "
                  f"L_adv={l_adv.item():.4f} adv_active={adv_active:.0f} gap={gap:+.4f}) "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            log.append({"iter": iter_idx + 1, "L": L.item(),
                        "L_oracle": l_oracle.item(), "L_shuf": l_shuf.item(),
                        "L_adv": l_adv.item(), "adv_active": adv_active,
                        "gap": gap, "lr": optim.param_groups[0]["lr"]})
            t0 = time.time()
        iter_idx += 1

    torch.save({"decoder": decoder.state_dict(),
                "args": vars(args), "log": log, "iter": iter_idx},
               os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "train_log.csv"), "w") as f:
        if log:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader()
            for r in log: w.writerow(r)

    # ---- eval in matched regime (delta_in_encoder=False already) ----
    print("\n=== EVAL on LLVE-STCD-12 held-out (crop=256, V3-B) ===")
    decoder.eval()
    rng_e = np.random.default_rng(args.seed + 1)
    audit_pairs = []
    results = {}
    per_seq = {}
    for cond in ["identity_delta", "oracle_delta", "shuffled_delta", "no_delta"]:
        rows = []
        with torch.no_grad():
            for name, lq_p, gt_p in test_seqs:
                lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
                lq, gt = lq.to(device), gt.to(device)
                zL = encode(backbone, lq); zH = encode(backbone, gt)
                if cond == "identity_delta":
                    delta = [torch.zeros_like(z) for z in zL]
                elif cond == "oracle_delta":
                    delta = [zh - zl for zh, zl in zip(zH, zL)]
                elif cond == "shuffled_delta":
                    other = int(rng_e.integers(0, len(train_seqs)))
                    donor_name, lq_o, gt_o = train_seqs[other]
                    assert donor_name != name
                    audit_pairs.append({"current_seq": name, "donor_seq": donor_name})
                    lq_oo, gt_oo = load_clip(lq_o, gt_o, crop_size=args.crop)
                    zL_o = encode(backbone, lq_oo.to(device))
                    zH_o = encode(backbone, gt_oo.to(device))
                    delta = [zh_o - zl_o for zh_o, zl_o in zip(zH_o, zL_o)]
                else:
                    delta = None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = decoder(lq, kv_layers=zL, delta_layers=delta)
                psnr = mean_psnr(to_u8(pred.float()), to_u8(gt))
                psnr_lq = mean_psnr(to_u8(lq), to_u8(gt))
                rows.append({"seq": name, "lq_psnr": psnr_lq, "pred_psnr": psnr})
                print(f"    [{cond}/{name}] LQ={psnr_lq:.2f} pred={psnr:.3f}")
        per_seq[cond] = rows
        results[cond] = float(np.mean([r["pred_psnr"] for r in rows]))
        print(f"  -> {cond} mean PSNR = {results[cond]:.3f}")

    # save artifacts
    for cond, rows in per_seq.items():
        cd = os.path.join(args.out_dir, cond); os.makedirs(cd, exist_ok=True)
        with open(os.path.join(cd, "per_seq.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
    with open(os.path.join(args.out_dir, "eval_shuffle_audit.csv"), "w") as f:
        f.write("current_seq,donor_seq\n")
        for r in audit_pairs: f.write(f"{r['current_seq']},{r['donor_seq']}\n")

    ident = results["identity_delta"]
    oracle_g = results["oracle_delta"] - ident
    shuf_g = results["shuffled_delta"] - ident
    nodelta_g = results["no_delta"] - ident
    pass_oracle = oracle_g >= 10.0
    pass_killer = shuf_g <= 0.5
    partial = (oracle_g >= 10.0) and (0.5 < shuf_g <= 1.0)
    if pass_oracle and pass_killer: verdict = "PASS — launch C-1a"
    elif partial:                    verdict = "PARTIAL — DAN5 allows one bump (lambda_adv -> 1.0)"
    else:                            verdict = "FAIL — move to C-2 non-additive interface"

    md = os.path.join(args.out_dir, "V3B_SUMMARY.md")
    with open(md, "w") as f:
        f.write("# V3-B Shuffle-Adversarial Decoder Results (DAN)\n\n")
        f.write(f"decoder: DeltaModulatedDecoder(delta_in_encoder=False) (V2-style)\n")
        f.write(f"trainable: {n_params/1e6:.2f}M\n")
        f.write(f"iters: {args.iters} | lr: {args.lr} | crop: {args.crop}\n")
        f.write(f"lambda_oracle: {args.lambda_oracle} | lambda_adv: {args.lambda_adv} | margin: {args.margin}\n\n")
        f.write("| condition | mean PSNR | Δ vs identity |\n|---|---:|---:|\n")
        for c in ["identity_delta","oracle_delta","shuffled_delta","no_delta"]:
            f.write(f"| {c} | {results[c]:.3f} | {results[c]-ident:+.3f} |\n")
        f.write(f"\n## Gates (DAN5)\n\n")
        f.write(f"- oracle_gain ≥ +10.0:  {oracle_g:+.3f}  → **{'PASS' if pass_oracle else 'FAIL'}**\n")
        f.write(f"- shuf_gap ≤ +0.5:      {shuf_g:+.3f}  → **{'PASS' if pass_killer else 'FAIL'}**\n")
        f.write(f"- no_delta - identity:  {nodelta_g:+.3f}  (informational)\n")
        f.write(f"\n## Verdict\n\n**{verdict}**\n")
    print(f"\n[V3-B GATES] oracle_gain={oracle_g:+.3f} ({'PASS' if pass_oracle else 'FAIL'})  "
          f"shuf_gap={shuf_g:+.3f} ({'PASS' if pass_killer else 'FAIL'})  "
          f"verdict: {verdict}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/v3b_shuffle_adv")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_oracle", type=float, default=1.0)
    ap.add_argument("--lambda_adv", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.05)
    main(ap.parse_args())
