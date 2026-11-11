"""PATH C: passive V-JEPA cross-attn baseline.

Decoder trained from scratch with kv_layers=[z_LQ] at TRAIN and INFERENCE.
No REPA, no projector, no refiner. Pure Charbonnier.

Tests if runtime V-JEPA cross-attn alone (no training-time alignment loss)
beats E2b's 19.556 (which uses REPA training, no V-JEPA at inference).
"""
from __future__ import annotations
import argparse, os, sys, time

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
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[PATH C] decoder {n_d/1e6:.2f}M trainable; V-JEPA encoder 304M frozen")
    print(f"         kv = z_LQ at layer {args.ref_layer} at TRAIN and INFERENCE")

    optim = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        _, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.no_grad():
            z_L = encode(backbone, lq)[args.ref_layer].float()

        kv_layers = [z_L]
        h_b, skips = decoder.encode(lq, kv_layers=kv_layers, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
        L = charbonnier(pred.float(), gt)

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item()})
            t0 = time.time()

    torch.save({"decoder": decoder.state_dict(), "args": vars(args), "log": log,
                "stage": "passive_vjepa"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # 4-mode eval per DBI5: passive_zL, no_prior, shuffled_zL, zero_zL
    decoder.eval()
    print("\n=== EVAL: 4 controls per DBI5 ===")
    results = {}
    for mode in ["passive_zL", "no_prior", "shuffled_zL", "zero_zL"]:
        psnrs = _eval_mode(decoder, backbone, test_seqs, device, args.crop,
                           args.ref_layer, mode)
        m = float(np.mean(psnrs))
        results[mode] = m
        print(f"  [{mode:14s}] mean PSNR = {m:.3f}")

    md = os.path.join(args.out_dir, "PATH_C_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# PATH C: passive V-JEPA cross-attn baseline\n\n")
        f.write(f"decoder: {n_d/1e6:.2f}M trainable, V-JEPA encoder 304M frozen\n")
        f.write(f"trained from scratch with kv=[z_LQ] at every iter, lr={args.lr}, "
                f"iters={args.iters}, seed={args.seed}\n\n")
        f.write(f"## PSNR by mode (DBI5 controls)\n\n")
        f.write(f"| mode | mean PSNR | δ vs passive_zL |\n|---|---|---|\n")
        for mode in ["passive_zL", "no_prior", "shuffled_zL", "zero_zL"]:
            d = results[mode] - results["passive_zL"]
            f.write(f"| {mode} | {results[mode]:.3f} | {d:+.3f} |\n")
        f.write(f"\n## DBI5 mainline gates\n")
        gates = [
            ("passive_zL >= 19.5", results["passive_zL"] >= 19.5),
            ("passive_zL - no_prior >= +1.5", (results["passive_zL"] - results["no_prior"]) >= 1.5),
            ("passive_zL - shuffled_zL >= +0.5", (results["passive_zL"] - results["shuffled_zL"]) >= 0.5),
            ("strong: passive_zL - shuffled_zL >= +1.0", (results["passive_zL"] - results["shuffled_zL"]) >= 1.0),
            ("weak/dead: passive_zL < 19.0", results["passive_zL"] < 19.0),
        ]
        for desc, ok in gates:
            f.write(f"- {desc}: {'PASS' if ok else 'FAIL'}\n")
    print(f"\nPATH C passive_zL = {results['passive_zL']:.3f}")
    print(f"  vs no_prior     = {results['no_prior']:.3f}  (δ={results['passive_zL']-results['no_prior']:+.3f})")
    print(f"  vs shuffled_zL  = {results['shuffled_zL']:.3f}  (δ={results['passive_zL']-results['shuffled_zL']:+.3f})")
    print(f"  vs zero_zL      = {results['zero_zL']:.3f}  (δ={results['passive_zL']-results['zero_zL']:+.3f})")
    print(f"wrote {md}")


@torch.no_grad()
def _eval_mode(decoder, backbone, test_seqs, device, crop, ref_layer, mode):
    psnrs = []
    n = len(test_seqs)
    for i, (name, lq_p, gt_p) in enumerate(test_seqs):
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        z_L = encode(backbone, lq)[ref_layer].float()

        if mode == "passive_zL":
            kv = [z_L]
        elif mode == "no_prior":
            kv = None
        elif mode == "shuffled_zL":
            j = (i + 1) % n
            _, lq_d, gt_d = test_seqs[j]
            lq_donor, _ = load_clip(lq_d, gt_d, crop_size=crop)
            lq_donor = lq_donor.to(device)
            z_L_donor = encode(backbone, lq_donor)[ref_layer].float()
            kv = [z_L_donor]
        elif mode == "zero_zL":
            kv = [torch.zeros_like(z_L)]
        else:
            raise ValueError(mode)

        h_b, skips = decoder.encode(lq, kv_layers=kv, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv, delta_layers=None)
        psnrs.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
    return psnrs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    main(ap.parse_args())
