"""Stage 2 variant: drop_h_b adapter (DBB5.B).

Difference from v11_c2_stage2_train.py: uses ContentAttnAdapterNoHb (no h_b input).
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from models.v11.content_attn_adapter_v2 import ContentAttnAdapterNoHb
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    sd = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    decoder.load_state_dict(sd["decoder"])
    for p in decoder.parameters(): p.requires_grad_(False)
    decoder.eval()
    print(f"[decoder] FROZEN, loaded from {args.stage1_ckpt}")

    adapter = ContentAttnAdapterNoHb(
        vjepa_grid=tuple(decoder.vjepa_grid),
        vjepa_dim=decoder.vjepa_dim,
        active_scales=tuple(int(s) for s in args.active_scales.split(",")),
        ref_layer=args.ref_layer,
        attn_heads=args.attn_heads,
        ffn_mult=args.ffn_mult,
    ).to(device)
    n_a = adapter.num_trainable_params()
    print(f"[adapter] ContentAttnAdapterNoHb (DBB5.B): {n_a/1e6:.2f}M (no h_b input)")

    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    log = []
    rank_active_count = 0
    t0 = time.time()
    iter_idx = 0

    while iter_idx < args.iters:
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.no_grad():
            z_L = encode(backbone, lq)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)

        other = int(rng.integers(0, len(train_seqs)))
        tries = 0
        while other == idx and tries < 5:
            other = int(rng.integers(0, len(train_seqs))); tries += 1
        _, lq_o, gt_o = train_seqs[other]
        lq_oo, _ = load_clip(lq_o, gt_o, crop_size=args.crop)
        with torch.no_grad():
            z_L_o = encode(backbone, lq_oo.to(device))

        kv_good = adapter(z_L)        # NO h_b input
        kv_bad = adapter(z_L_o)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred_good = decoder.decode(h_b, skips, lq, kv_layers=None,
                                       delta_layers=None, kv_override=kv_good)
            pred_bad  = decoder.decode(h_b, skips, lq, kv_layers=None,
                                       delta_layers=None, kv_override=kv_bad)
        l_good = charbonnier(pred_good.float(), gt)
        l_bad  = charbonnier(pred_bad.float(),  gt)
        l_rank = F.relu(args.margin + l_good - l_bad)
        L = l_good + args.lambda_rank * l_rank

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optim.step(); sched.step()

        rank_active_count += int(l_rank.item() > 0)

        if (iter_idx + 1) % args.log_every == 0:
            print(f"  it {iter_idx+1}/{args.iters} L={L.item():.4f} "
                  f"L_good={l_good.item():.4f} L_bad={l_bad.item():.4f} "
                  f"L_rank={l_rank.item():.4f} rank_active={rank_active_count}/{iter_idx+1} "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            log.append({"iter": iter_idx + 1, "L": L.item(),
                        "L_good": l_good.item(), "L_bad": l_bad.item(),
                        "L_rank": l_rank.item(),
                        "rank_active_count": rank_active_count,
                        "lr": optim.param_groups[0]["lr"]})
            t0 = time.time()
        iter_idx += 1

    torch.save({"decoder": decoder.state_dict(),
                "adapter": adapter.state_dict(),
                "args": vars(args), "log": log, "iter": iter_idx,
                "stage": 2,
                "variant": "DBB5.B_no_hb",
                "stage1_ckpt": args.stage1_ckpt},
               os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "train_log.csv"), "w") as f:
        if log:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader()
            for r in log: w.writerow(r)
    print(f"[ckpt] saved Stage 2 (no_hb) → {args.out_dir}/ckpt.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/c2_stage2_nohb")
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--active_scales", default="3,4")
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--ffn_mult", type=float, default=1.0)
    main(ap.parse_args())
