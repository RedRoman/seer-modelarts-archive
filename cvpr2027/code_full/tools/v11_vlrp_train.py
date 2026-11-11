"""J1: VLR-P (V-JEPA Latent Restoration via Predictor).

Novel framework: V-JEPA's own predictor head does the restoration in latent
space. The decoder is reduced to a "renderer" that maps refined V-JEPA
tokens → pixels. V-JEPA is used at INFERENCE (unlike REPA which only used
it at training).

Pipeline:
    LQ pixels (B,3,T,H,W)
        │
        ▼  V-JEPA encoder (frozen, 304M)
    z_LQ tokens (B, N=2048, 1024)  at layer 23 (deepest)
        │
        ▼  CounterfactualRefiner (predictor body frozen + LoRA-rank-16, ~1.5M)
    z_HQ_pred (B, N, 1024)          identity-at-init via gate
        │
        ▼  DeltaModulatedDecoder cross-attn @ scales 3,4 with kv = z_HQ_pred
    HQ pixels (B,3,T,H,W)

Trainable: LoRA (~1.5M) + decoder (~33M) = ~35M
Frozen: V-JEPA encoder (304M) + V-JEPA predictor body (~150M)

Loss: Charbonnier on pixels (no REPA, no projector).
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
from models.predictor_lora import CounterfactualRefiner
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


VJEPA_REPO  = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT  = "/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    refiner = CounterfactualRefiner(
        local_repo=VJEPA_REPO, ckpt_path=VJEPA_CKPT,
        encoder_embed_dim=1024, image_size=args.crop, num_frames=16,
        patch_size=16, tubelet_size=2,
        predictor_embed_dim=384, depth=12, num_heads=12,
        lora_rank=args.lora_rank, freeze_body=True,
    ).to(device)
    n_refiner = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    n_refiner_total = sum(p.numel() for p in refiner.parameters())
    print(f"[refiner] {n_refiner/1e6:.2f}M trainable / {n_refiner_total/1e6:.1f}M total")

    # Decoder configured to cross-attn the refined V-JEPA tokens at scales 3+4.
    # Set vjepa_layer_per_scale to all-0 so it always indexes kv_layers[0]
    # (we pass kv_layers=[z_HQ_pred] as a single-element list).
    decoder = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[VLR-P] decoder {n_d/1e6:.2f}M; total trainable {(n_d + n_refiner)/1e6:.2f}M; "
          f"V-JEPA encoder + predictor frozen ({(304 + n_refiner_total/1e6 - n_refiner/1e6):.0f}M)")

    train_params = list(decoder.parameters()) + list(refiner.trainable_parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

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

        # 1. V-JEPA encode LQ (frozen, no grad)
        with torch.no_grad():
            z_L = encode(backbone, lq)
        z_LQ_top = z_L[args.ref_layer].float()      # (B, N=2048, 1024)

        # 2. Predictor + LoRA refines z_LQ → z_HQ_pred
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_HQ_pred = refiner(z_LQ_top.bfloat16())
            z_HQ_pred = z_HQ_pred.float()

        # 3. Decoder cross-attn with refined tokens
        kv_layers = [z_HQ_pred]
        h_b, skips = decoder.encode(lq, kv_layers=kv_layers, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
        L = charbonnier(pred.float(), gt)

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            gate_val = float(refiner.gate.detach().abs().item())
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} "
                  f"|gate|={gate_val:.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(), "gate": gate_val})
            t0 = time.time()

    torch.save({"decoder": decoder.state_dict(), "refiner": refiner.state_dict(),
                "args": vars(args), "log": log, "stage": "vlrp"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval (V-JEPA + predictor + decoder all run at inference)
    decoder.eval(); refiner.eval()
    print("\n=== EVAL: SDSD held-out (VLR-P, V-JEPA+predictor used at inference) ===")
    psnrs = []
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq)
            z_LQ_top = z_L[args.ref_layer].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_HQ_pred = refiner(z_LQ_top.bfloat16()).float()
                kv_layers = [z_HQ_pred]
                h_b, skips = decoder.encode(lq, kv_layers=kv_layers, delta_layers=None)
                pred = decoder.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
            p = mean_psnr(to_u8(pred.float()), to_u8(gt))
            psnrs.append(p)
            print(f"    [{name}] LQ→pred = {p:.3f}")
    mean_p = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "VLRP_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# VLR-P Eval (V-JEPA Latent Restoration via Predictor)\n\n")
        f.write(f"V-JEPA encoder 304M (frozen) + V-JEPA predictor body ~150M (frozen)\n")
        f.write(f"+ LoRA {n_refiner/1e6:.2f}M trainable + decoder {n_d/1e6:.2f}M trainable\n\n")
        f.write(f"## SDSD held-out PSNR = {mean_p:.3f}  (vs E2b 19.556, vs pure_rgb 18.533)\n")
        f.write(f"Final LoRA gate magnitude: {float(refiner.gate.detach().abs().item()):.4f}\n")
    print(f"\nVLR-P SDSD PSNR = {mean_p:.3f}  (vs E2b 19.556)")
    print(f"final |gate| = {float(refiner.gate.detach().abs().item()):.4f}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--ref_layer", type=int, default=3, help="V-JEPA layer to feed predictor")
    main(ap.parse_args())
