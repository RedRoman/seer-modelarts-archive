"""tools/v11_c2_stage1_train.py — C-2-Direct Stage 1 (condition-mixed scaffold).

Per DAS4 / DAR3 / DAT2:
  trainable: decoder + adapter
  frozen:    V-JEPA
  conditions sampled per iter: adapted_zL 50%, passive_zL 25%, no_prior 25%
  loss:      RGB Charbonnier on the sampled condition
  iters:     1500 default

DAT2: h_b/skips MUST be from decoder.encode(LQ, kv_layers=None) — V-JEPA-clean.

Stage 1 gates (DAS4):
  adapted_zL - passive_zL >= +2.0 dB    PASS
  adapted_zL - no_prior   >= +4.0 dB    PASS

Spec: VJEPA_LLVE_CODE_REVIEW.md §DAS4 / §DAU.
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
from models.v11.content_attn_adapter import ContentAttnAdapter
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


DEFAULT_PROBS = {"adapted_zL": 0.50, "passive_zL": 0.25, "no_prior": 0.25}
FALLBACK_PROBS = {"adapted_zL": 0.40, "passive_zL": 0.30, "no_prior": 0.30}  # DAT6
CONDITIONS = list(DEFAULT_PROBS.keys())
RATIO_CHECK_ITER = 500   # DAT6: at iter 500, compare passive_zL vs no_prior


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print("[backbone] V-JEPA frozen")

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[decoder] V2-style trainable: {n_d/1e6:.2f}M")

    adapter = ContentAttnAdapter(
        bottleneck_channels=decoder.dims[-1],
        vjepa_grid=tuple(decoder.vjepa_grid),
        vjepa_dim=decoder.vjepa_dim,
        active_scales=tuple(int(s) for s in args.active_scales.split(",")),
        ref_layer=args.ref_layer,
        attn_heads=args.attn_heads,
        ffn_mult=args.ffn_mult,
    ).to(device)
    n_a = adapter.num_trainable_params()
    print(f"[adapter] ContentAttnAdapter trainable: {n_a/1e6:.2f}M")

    optim = torch.optim.AdamW(
        list(decoder.parameters()) + list(adapter.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()

    rng = np.random.default_rng(args.seed)
    log = []
    cond_running = {c: [] for c in CONDITIONS}
    t0 = time.time()
    iter_idx = 0
    probs_dict = dict(DEFAULT_PROBS)
    probs = [probs_dict[c] for c in CONDITIONS]
    ratio_switched = False
    print(f"[DAT6] starting sampling probs: {probs_dict}")

    while iter_idx < args.iters:
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.no_grad():
            z_L = encode(backbone, lq)

        # DAT2: V-JEPA-clean h_b/skips
        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)

        # sample condition
        cond = str(rng.choice(CONDITIONS, p=probs))
        if cond == "no_prior":
            kv_layers, kv_override = None, None
        elif cond == "passive_zL":
            kv_layers, kv_override = z_L, None
        else:  # adapted_zL
            kv_layers = None
            kv_override = adapter(z_L, h_b)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq,
                                  kv_layers=kv_layers, delta_layers=None,
                                  kv_override=kv_override)
        l_rgb = charbonnier(pred.float(), gt)
        L = l_rgb

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decoder.parameters()) + list(adapter.parameters()), 1.0)
        optim.step(); sched.step()

        cond_running[cond].append(l_rgb.item())

        # DAT6: at iter RATIO_CHECK_ITER, switch to fallback if passive_zL lags no_prior
        if (not ratio_switched) and (iter_idx + 1 == RATIO_CHECK_ITER):
            recent = 50
            pass_loss = float(np.mean(cond_running["passive_zL"][-recent:])) if cond_running["passive_zL"] else float("nan")
            none_loss = float(np.mean(cond_running["no_prior"][-recent:])) if cond_running["no_prior"] else float("nan")
            if np.isfinite(pass_loss) and np.isfinite(none_loss) and pass_loss > none_loss:
                probs_dict = dict(FALLBACK_PROBS)
                probs = [probs_dict[c] for c in CONDITIONS]
                ratio_switched = True
                msg = (f"[DAT6 SWITCH @ iter {iter_idx+1}] passive_zL loss={pass_loss:.4f} > no_prior loss={none_loss:.4f}; "
                       f"switching probs to {probs_dict}")
            else:
                msg = (f"[DAT6 KEEP   @ iter {iter_idx+1}] passive_zL loss={pass_loss:.4f}, no_prior loss={none_loss:.4f}; "
                       f"keeping default probs")
            print(msg)
            with open(os.path.join(args.out_dir, "ratio_check.txt"), "w") as _f:
                _f.write(msg + "\n")

        if (iter_idx + 1) % args.log_every == 0:
            running_means = {c: float(np.mean(cond_running[c][-50:])) if cond_running[c] else float('nan')
                             for c in CONDITIONS}
            print(f"  it {iter_idx+1}/{args.iters} L={L.item():.4f} cond={cond} "
                  f"L_rgb={l_rgb.item():.4f}  "
                  f"running[adp={running_means['adapted_zL']:.4f} "
                  f"pass={running_means['passive_zL']:.4f} "
                  f"none={running_means['no_prior']:.4f}]  "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            log.append({"iter": iter_idx + 1, "L": L.item(),
                        "cond": cond, "L_rgb": l_rgb.item(),
                        **{f"running_{c}": running_means[c] for c in CONDITIONS},
                        "lr": optim.param_groups[0]["lr"]})
            t0 = time.time()
        iter_idx += 1

    torch.save({"decoder": decoder.state_dict(),
                "adapter": adapter.state_dict(),
                "args": vars(args), "log": log, "iter": iter_idx,
                "stage": 1},
               os.path.join(args.out_dir, "ckpt.pt"))
    with open(os.path.join(args.out_dir, "train_log.csv"), "w") as f:
        if log:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader()
            for r in log: w.writerow(r)
    print(f"[ckpt] saved Stage 1 → {args.out_dir}/ckpt.pt")
    print("\nNext: run tools/v11_c2_eval.py against this ckpt for Stage 1 gates.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/c2_stage1")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--active_scales", default="3,4")
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--attn_heads", type=int, default=8)
    ap.add_argument("--ffn_mult", type=float, default=1.0)
    main(ap.parse_args())
