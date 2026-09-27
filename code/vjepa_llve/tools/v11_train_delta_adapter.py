"""tools/v11_train_delta_adapter.py — CZ4 Probe B.

Train JEPADeltaAdapter A: z_L -> z_hat_H (= z_L + delta_hat).
Compare against identity baseline (z_hat_H = z_L) on held-out clips.

Loss per layer:
  L_layer = (1 - cos(z_hat_H, z_H))_mean  +  alpha * ||z_hat_H - z_H||^2_mean

Eval metrics on held-out (LLVE-STCD-12):
  per-layer cos(z_hat_H, z_H), per-layer ||z_hat_H - z_H||
  identity baseline = cos(z_L, z_H), ||z_L - z_H||

Pass criterion (CZ Probe B):
  adapter mean cos > identity mean cos  AND
  adapter L2 < identity L2  on held-out, with margin > 5% relative.

Spec: VJEPA_LLVE_CODE_REVIEW.md §CZ4.B.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.sdsd_dataset import _detect_pair_dirs
from data.common import list_sequences, list_images, read_image_01
from models.vjepa_backbone import build_vjepa_backbone
from models.v11 import JEPADeltaAdapter
from tools.v11_delta_stats import (
    LLVE_STCD_12, ORIGINAL_INDOOR_14, INDOOR_ROOT, OUTDOOR_ROOT, index_seqs,
)


def load_clip(lq_paths, gt_paths, n_frames=16):
    n_total = len(lq_paths); T = min(n_frames, n_total)
    start = (n_total - T) // 2
    lq = np.stack([read_image_01(lq_paths[start+i]) for i in range(T)], axis=0)
    gt = np.stack([read_image_01(gt_paths[start+i]) for i in range(T)], axis=0)
    lq = torch.from_numpy(lq).permute(3,0,1,2).unsqueeze(0).float()
    gt = torch.from_numpy(gt).permute(3,0,1,2).unsqueeze(0).float()
    return lq, gt


def build_split_seqs() -> Tuple[List, List]:
    """DA3 fix: held-out = LLVE_STCD_12 names; train MUST exclude them.
    index_seqs() scans whole disk regardless of train/test split, so we
    explicitly filter by name."""
    indoor_full  = index_seqs(INDOOR_ROOT,  "train")  # scans whole disk
    outdoor_full = index_seqs(OUTDOOR_ROOT, "train")
    test_names = set(LLVE_STCD_12)

    train_seqs = []
    for name, (lq, gt) in indoor_full.items():
        if name in test_names: continue          # DA3: exclude LLVE-STCD-12
        train_seqs.append((name, lq, gt))
    for name, (lq, gt) in outdoor_full.items():
        if name in test_names: continue
        train_seqs.append((name, lq, gt))

    test_seqs = [(name, *indoor_full[name]) for name in LLVE_STCD_12
                 if name in indoor_full]

    train_names = set(s[0] for s in train_seqs)
    held_names  = set(s[0] for s in test_seqs)
    inter = train_names & held_names
    assert not inter, f"DA3 violation: train/test intersect on {inter}"
    print(f"[split] train={len(train_seqs)}  held-out={len(test_seqs)}  "
          f"disjoint=YES (DA3 assert passed)")
    return train_seqs, test_seqs


@torch.no_grad()
def encode(backbone, vid):
    return [f.float() for f in backbone(vid)["feats"]]


def cosine_loss(z_hat, z_H):
    return (1.0 - F.cosine_similarity(z_hat, z_H, dim=-1)).mean()


def train(args):
    device = torch.device("cuda")
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    adapter = JEPADeltaAdapter(n_layers=4, in_dim=1024, bot_dim=args.bot_dim,
                               depth=args.depth, heads=args.heads).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"[train] adapter params: {n_params/1e6:.1f}M")

    train_seqs, test_seqs = build_split_seqs()
    print(f"[train] train_seqs={len(train_seqs)}, test_seqs={len(test_seqs)}")

    optim = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    # DA5: compute mean-delta and per-layer affine baselines on TRAIN data
    print("[train] computing mean-delta + affine baselines on train (DA5 baselines)...")
    sum_delta = [None]*4; n_used = 0
    sum_zL = [None]*4; sum_zH = [None]*4
    for name, lq_p, gt_p in train_seqs[:min(64, len(train_seqs))]:
        lq, gt = load_clip(lq_p, gt_p)
        zL = encode(backbone, lq.to(device))
        zH = encode(backbone, gt.to(device))
        for li in range(4):
            d = (zH[li] - zL[li]).mean(dim=(0,1))   # (D,)
            sum_delta[li] = d if sum_delta[li] is None else sum_delta[li] + d
            zLm = zL[li].mean(dim=(0,1)); zHm = zH[li].mean(dim=(0,1))
            sum_zL[li] = zLm if sum_zL[li] is None else sum_zL[li] + zLm
            sum_zH[li] = zHm if sum_zH[li] is None else sum_zH[li] + zHm
        n_used += 1
    mean_delta = [sd / n_used for sd in sum_delta]   # (D,) per layer
    mean_zL    = [sl / n_used for sl in sum_zL]
    mean_zH    = [sh / n_used for sh in sum_zH]
    # Affine: z_hat = (z_L - mean_zL) * scale + mean_zH; scale = ||mean_zH|| / ||mean_zL||
    affine_scale = [zhm.norm() / zlm.norm().clamp_min(1e-8) for zhm, zlm in zip(mean_zH, mean_zL)]
    print(f"[baselines] computed on {n_used} train clips; per-layer ||mean_delta||: "
          f"{[md.norm().item() for md in mean_delta]}")

    # Pre-encode test set ONCE (small, fits in mem)
    print("[train] caching test z_L,z_H ...")
    test_cache = []
    for name, lq, gt in test_seqs:
        lq_t, gt_t = load_clip(lq, gt)
        z_L = encode(backbone, lq_t.to(device))
        z_H = encode(backbone, gt_t.to(device))
        test_cache.append((name, z_L, z_H))

    @torch.no_grad()
    def eval_holdout():
        adapter.eval()
        per_layer = {li: {"cos_id": [], "cos_ad": [], "l2_id": [], "l2_ad": []}
                     for li in range(4)}
        for name, z_L, z_H in test_cache:
            z_hat = adapter(z_L)
            for li in range(4):
                cos_ad = F.cosine_similarity(z_hat[li], z_H[li], dim=-1).mean().item()
                cos_id = F.cosine_similarity(z_L[li],  z_H[li], dim=-1).mean().item()
                l2_ad  = (z_hat[li] - z_H[li]).pow(2).mean().sqrt().item()
                l2_id  = (z_L[li]  - z_H[li]).pow(2).mean().sqrt().item()
                # DA5: delta-direction cos + magnitude ratio
                delta_true = z_H[li] - z_L[li]
                delta_hat  = z_hat[li] - z_L[li]
                cos_delta = F.cosine_similarity(delta_hat, delta_true, dim=-1).mean().item()
                mag_ratio = (delta_hat.norm(dim=-1).mean() / delta_true.norm(dim=-1).mean().clamp_min(1e-8)).item()
                per_layer[li]["cos_ad"].append(cos_ad); per_layer[li]["cos_id"].append(cos_id)
                per_layer[li]["l2_ad"].append(l2_ad);   per_layer[li]["l2_id"].append(l2_id)
                per_layer[li].setdefault("cos_delta", []).append(cos_delta)
                per_layer[li].setdefault("mag_ratio", []).append(mag_ratio)
                per_layer[li].setdefault("l2_mean_baseline", [])
                per_layer[li].setdefault("l2_affine_baseline", [])
                # mean-delta baseline: z_hat_mean = z_L + mean_delta
                z_hat_mean = z_L[li] + mean_delta[li].view(1, 1, -1)
                l2_mean = (z_hat_mean - z_H[li]).pow(2).mean().sqrt().item()
                cos_mean = F.cosine_similarity(z_hat_mean, z_H[li], dim=-1).mean().item()
                per_layer[li]["l2_mean_baseline"].append(l2_mean)
                per_layer[li].setdefault("cos_mean_baseline", []).append(cos_mean)
                # affine baseline: z_hat_affine = (z_L - mean_zL) * scale + mean_zH
                z_hat_aff = (z_L[li] - mean_zL[li]) * affine_scale[li] + mean_zH[li]
                l2_aff = (z_hat_aff - z_H[li]).pow(2).mean().sqrt().item()
                cos_aff = F.cosine_similarity(z_hat_aff, z_H[li], dim=-1).mean().item()
                per_layer[li]["l2_affine_baseline"].append(l2_aff)
                per_layer[li].setdefault("cos_affine_baseline", []).append(cos_aff)
        adapter.train()
        return per_layer

    log = []
    rng = np.random.default_rng(args.seed)
    iter_idx = 0
    t0 = time.time()
    while iter_idx < args.iters:
        # sample one train clip
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p)
        z_L = encode(backbone, lq.to(device))
        z_H = encode(backbone, gt.to(device))

        z_hat = adapter(z_L)
        loss_cos = sum(cosine_loss(zh, zg) for zh, zg in zip(z_hat, z_H)) / 4.0
        loss_l2 = sum((zh - zg).pow(2).mean() for zh, zg in zip(z_hat, z_H)) / 4.0
        loss = loss_cos + args.alpha * loss_l2

        optim.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optim.step(); sched.step()

        if (iter_idx + 1) % args.log_every == 0:
            print(f"  it {iter_idx+1}/{args.iters} loss={loss.item():.4f} "
                  f"(cos={loss_cos.item():.4f} l2={loss_l2.item():.5f}) "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            log.append({"iter": iter_idx+1, "loss": loss.item(),
                        "loss_cos": loss_cos.item(), "loss_l2": loss_l2.item()})
            t0 = time.time()

        if (iter_idx + 1) % args.eval_every == 0 or iter_idx + 1 == args.iters:
            ev = eval_holdout()
            print(f"  [eval @ {iter_idx+1}]")
            for li in range(4):
                cos_id = float(np.mean(ev[li]["cos_id"])); cos_ad = float(np.mean(ev[li]["cos_ad"]))
                l2_id  = float(np.mean(ev[li]["l2_id"]));  l2_ad  = float(np.mean(ev[li]["l2_ad"]))
                d_cos = cos_ad - cos_id; d_l2 = l2_ad - l2_id
                verdict = ("PASS" if (cos_ad > cos_id and l2_ad < l2_id) else "FAIL")
                cos_de = float(np.mean(ev[li]["cos_delta"])); mag_r = float(np.mean(ev[li]["mag_ratio"]))
                cos_mn = float(np.mean(ev[li]["cos_mean_baseline"])); l2_mn = float(np.mean(ev[li]["l2_mean_baseline"]))
                cos_af = float(np.mean(ev[li]["cos_affine_baseline"])); l2_af = float(np.mean(ev[li]["l2_affine_baseline"]))
                print(f"     layer {li}: cos id={cos_id:.4f} ad={cos_ad:.4f} mean={cos_mn:.4f} aff={cos_af:.4f}  "
                      f"L2 id={l2_id:.3f} ad={l2_ad:.3f} mean={l2_mn:.3f} aff={l2_af:.3f}  "
                      f"cos_Δ={cos_de:.4f} mag_ratio={mag_r:.3f}  -> {verdict}")
        iter_idx += 1

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(),
                "args": vars(args), "log": log},
               os.path.join(out_dir, "delta_adapter.pt"))
    # final eval table
    ev = eval_holdout()
    md = os.path.join(out_dir, "probe_b_summary.md")
    with open(md, "w") as f:
        f.write("# v11 Probe B′ — Delta Adapter vs identity / mean-delta / affine (DB3 verdict)\n\n")
        f.write(f"adapter params: {n_params/1e6:.1f}M\n")
        f.write(f"train_iters: {args.iters} | sequence-disjoint held-out: LLVE-STCD-12 ({len(test_cache)} seqs)\n")
        f.write(f"baselines: mean-delta + affine computed on first {n_used} train clips (DC3 disclosure)\n")
        f.write("(DA3 split-fix asserted disjoint; DA5 metrics added; DB3 baseline comparison required.)\n\n")

        f.write("## Cosine similarity (higher = better)\n\n")
        f.write("| layer | cos_id | cos_mean | cos_affine | cos_adapter | Δ vs id | Δ vs mean | Δ vs affine |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for li in range(4):
            cos_id = float(np.mean(ev[li]["cos_id"])); cos_ad = float(np.mean(ev[li]["cos_ad"]))
            cos_mn = float(np.mean(ev[li]["cos_mean_baseline"])); cos_af = float(np.mean(ev[li]["cos_affine_baseline"]))
            f.write(f"| {li} | {cos_id:.4f} | {cos_mn:.4f} | {cos_af:.4f} | {cos_ad:.4f} | "
                    f"{cos_ad-cos_id:+.4f} | {cos_ad-cos_mn:+.4f} | {cos_ad-cos_af:+.4f} |\n")

        f.write("\n## RMS L2 to z_H (lower = better)\n\n")
        f.write("| layer | L2_id | L2_mean | L2_affine | L2_adapter | Δ vs id | Δ vs mean | Δ vs affine |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for li in range(4):
            l2_id = float(np.mean(ev[li]["l2_id"])); l2_ad = float(np.mean(ev[li]["l2_ad"]))
            l2_mn = float(np.mean(ev[li]["l2_mean_baseline"])); l2_af = float(np.mean(ev[li]["l2_affine_baseline"]))
            f.write(f"| {li} | {l2_id:.3f} | {l2_mn:.3f} | {l2_af:.3f} | {l2_ad:.3f} | "
                    f"{l2_ad-l2_id:+.3f} | {l2_ad-l2_mn:+.3f} | {l2_ad-l2_af:+.3f} |\n")

        f.write("\n## DA5 delta-direction metrics\n\n")
        f.write("| layer | cos(delta_hat, true_delta) | ||delta_hat||/||delta|| (mag ratio) |\n")
        f.write("|---:|---:|---:|\n")
        for li in range(4):
            cd = float(np.mean(ev[li]["cos_delta"])); mr = float(np.mean(ev[li]["mag_ratio"]))
            f.write(f"| {li} | {cd:.4f} | {mr:.3f} |\n")

        f.write("\n## DB3 verdict: adapter must beat ALL THREE (identity, mean-delta, affine)\n\n")
        per_layer_pass = []
        for li in range(4):
            cos_id = float(np.mean(ev[li]["cos_id"])); cos_ad = float(np.mean(ev[li]["cos_ad"]))
            cos_mn = float(np.mean(ev[li]["cos_mean_baseline"])); cos_af = float(np.mean(ev[li]["cos_affine_baseline"]))
            l2_id = float(np.mean(ev[li]["l2_id"])); l2_ad = float(np.mean(ev[li]["l2_ad"]))
            l2_mn = float(np.mean(ev[li]["l2_mean_baseline"])); l2_af = float(np.mean(ev[li]["l2_affine_baseline"]))
            beats_cos = cos_ad > cos_id and cos_ad > cos_mn and cos_ad > cos_af
            beats_l2  = l2_ad  < l2_id  and l2_ad  < l2_mn  and l2_ad  < l2_af
            verd = "PASS" if (beats_cos and beats_l2) else "FAIL"
            per_layer_pass.append(verd == "PASS")
            f.write(f"- layer {li}: cos {beats_cos and 'beats all' or 'NOT all'}, "
                    f"L2 {beats_l2 and 'beats all' or 'NOT all'} -> **{verd}**\n")
        all_pass = all(per_layer_pass)
        # DC2: layer-23-only is the only Probe-A-supported narrowing; report as
        # EXPLORATORY only. Do NOT use it as a decision gate unless V11_JDAP_SPEC.md
        # is patched BEFORE judging (DC2 / Gate B condition).
        layer23_only = per_layer_pass[3]
        layer17_23   = per_layer_pass[2] and per_layer_pass[3]
        f.write(f"\n**Probe B′ all-4-layer verdict (Gate A, strict / pre-registered): "
                f"{'PASS' if all_pass else 'FAIL'}**\n")
        f.write(f"\n*Exploratory only (NOT a decision gate per DC2):*\n")
        f.write(f"- Layer-23-only (Gate B; would require spec patch before judging): "
                f"{'pass' if layer23_only else 'fail'}\n")
        f.write(f"- Layers 17+23 (Gate C; would require spec patch before judging): "
                f"{'pass' if layer17_23 else 'fail'}\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/probe_b")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="DA5 fix: weight on L2 of (z_hat_H, z_H). 1.0 balances cos+L2.")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--bot_dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    train(ap.parse_args())
