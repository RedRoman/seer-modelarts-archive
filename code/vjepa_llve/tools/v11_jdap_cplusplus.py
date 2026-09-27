"""tools/v11_jdap_cplusplus.py — C++ run per DAC5.

Extends v11_jdap_rgbdistill with:
  - per-layer delta_dropout (Branch B regularizer; DAB1/DY4 p=0.20)
  - InfoNCE on adapter outputs at layers 17+23 (DAB3 health log)
  - frozen-decoder RGB ranking (DX2/DY2 single-lane stop-grad path)
  - snapshots at iter_000 (save only), iter_400 (save + 3-seq sanity), final (save + full eval)
  - iter-400 mini-eval is soft-only — no early stop unless oracle collapse / NaN / dead ranking (DAB1)
  - portability eval at iter_400 and final using a separately-loaded oracle decoder
    (Probe C oracle_delta), reporting portable_pred_gain + portable_sep (DAA3)

Spec: VJEPA_LLVE_CODE_REVIEW.md §DAA / §DAB / §DAC.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, os, sys, time
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
from models.v11 import JEPADeltaAdapter
from models.v11.scale_aware_adapter import ScaleAwareDeltaAdapter
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_oracle_delta_probe import (
    load_clip, to_u8, mean_psnr, encode, charbonnier, build_split_seqs,
)
from tools.v11_jdap_rgbdistill import (
    latent_dir_loss, latent_mag_loss, latent_l2_loss, m_schedule,
)


# ----- Branch B regularizers ----------------------------------------
def per_layer_delta_dropout(delta_layers: List[torch.Tensor], p: float,
                            training: bool, rng: np.random.Generator
                            ) -> Tuple[List[torch.Tensor], List[int]]:
    """Independently zero each layer's delta with probability p.
    Safety: if all layers dropped, restore the deepest one so decoder
    sees at least one delta signal.
    Returns (out, mask) where mask[i] = 1 if kept, 0 if dropped.
    """
    if not training or p <= 0.0:
        return delta_layers, [1] * len(delta_layers)
    mask = [1 if rng.random() > p else 0 for _ in delta_layers]
    if sum(mask) == 0:
        mask[-1] = 1   # safety
    out = [d if m else torch.zeros_like(d) for d, m in zip(delta_layers, mask)]
    return out, mask


def descriptor(d: torch.Tensor) -> torch.Tensor:
    """Pool a delta tensor to a global L2-normalized descriptor.
    d: (B, N, D) where N = T*H*W tokens. Returns (B, D)."""
    v = d.float().mean(dim=1)   # (B, D)
    return F.normalize(v, dim=-1, eps=1e-6)


class FIFOQueue:
    """Simple FIFO queue of (D,) descriptors stored on device."""
    def __init__(self, dim: int, max_size: int, device):
        self.max = max_size
        self.buf = torch.zeros(0, dim, device=device)

    def push(self, x: torch.Tensor):
        # x: (B, D)
        with torch.no_grad():
            self.buf = torch.cat([self.buf, x.detach()], dim=0)
            if self.buf.size(0) > self.max:
                self.buf = self.buf[-self.max:]

    def get(self) -> torch.Tensor:
        return self.buf

    def __len__(self):
        return self.buf.size(0)


def info_nce(z_q: torch.Tensor, z_pos: torch.Tensor,
             queue: torch.Tensor, T: float) -> torch.Tensor:
    """Standard InfoNCE.
    z_q: (B, D), z_pos: (B, D), queue: (Nq, D). All L2-normalized.
    """
    B = z_q.size(0)
    pos = (z_q * z_pos).sum(dim=-1, keepdim=True)         # (B, 1)
    if queue.size(0) > 0:
        neg = z_q @ queue.T                                # (B, Nq)
        logits = torch.cat([pos, neg], dim=-1) / T         # (B, 1+Nq)
    else:
        logits = pos / T                                   # degenerate; CE -> 0
    target = torch.zeros(B, dtype=torch.long, device=z_q.device)
    return F.cross_entropy(logits, target)


# ----- main ---------------------------------------------------------
def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    print("[backbone] V-JEPA frozen")

    base_adapter = JEPADeltaAdapter(n_layers=4, in_dim=1024, bot_dim=256, depth=4, heads=4)
    adapter = ScaleAwareDeltaAdapter(base_adapter).to(device)
    if args.adapter_init and os.path.isfile(args.adapter_init):
        sd = torch.load(args.adapter_init, map_location="cpu", weights_only=False)
        if "adapter" in sd:
            adapter.load_state_dict(sd["adapter"])
            print(f"[adapter] full ScaleAware init from {args.adapter_init}")
        else:
            base_adapter.load_state_dict(sd)
            print(f"[adapter] base-only init from {args.adapter_init}")

    n_a = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[adapter] ScaleAwareDeltaAdapter trainable: {n_a/1e6:.2f}M")

    decoder = DeltaModulatedDecoder().to(device)
    if args.decoder_init and os.path.isfile(args.decoder_init):
        sd = torch.load(args.decoder_init, map_location="cpu", weights_only=False)
        decoder.load_state_dict(sd["decoder"])
        print(f"[decoder] init from {args.decoder_init}")
    if args.freeze_decoder:
        for p in decoder.parameters(): p.requires_grad_(False)
        decoder.eval()
        n_d = 0
        print("[decoder] FROZEN (Option B / DAE4 ceiling test)")
        # Use main (frozen) decoder for ranking too — no separate decoder_frozen needed.
        decoder_frozen = decoder
        optim_params = list(adapter.parameters())
    else:
        n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"[decoder] DeltaModulatedDecoder trainable: {n_d/1e6:.2f}M")
        # Frozen reference decoder for DX2/DY2 ranking — same init weights, all params frozen
        decoder_frozen = DeltaModulatedDecoder().to(device)
        decoder_frozen.load_state_dict(decoder.state_dict())
        for p in decoder_frozen.parameters():
            p.requires_grad_(False)
        decoder_frozen.eval()
        # NB: do NOT wrap forward in no_grad — we need adapter grad to flow.
        print("[decoder_frozen] ready (params frozen, no autograd on its own params)")
        optim_params = list(adapter.parameters()) + list(decoder.parameters())

    optim = torch.optim.AdamW(
        optim_params,
        lr=args.lr, weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    print(f"[split] train={len(train_seqs)} held-out={len(test_seqs)} disjoint=YES")

    # NCE setup — queue per layer
    nce_layers = [int(x) for x in args.nce_layers.split(",") if x]
    if args.use_nce:
        nce_dim = base_adapter.bot_dim if hasattr(base_adapter, "bot_dim") else 1024
        # adapter outputs are in V-JEPA dim (1024). descriptor() pools to same dim.
        nce_dim = 1024
        queues = {l: FIFOQueue(dim=nce_dim, max_size=args.nce_queue_size, device=device)
                  for l in nce_layers}
        print(f"[nce] layers={nce_layers} T={args.nce_T} w={args.nce_w} "
              f"queue={args.nce_queue_size} warmup_until_iter={args.nce_warmup}")
    else:
        queues = {}

    # snapshot scheduling
    save_iters = sorted(set([int(x) for x in args.save_iters.split(",") if x]))
    sanity_iter = args.sanity_iter

    # ----- save iter_000 snapshot before training -----
    if 0 in save_iters:
        torch.save({"adapter": adapter.state_dict(),
                    "decoder": decoder.state_dict(),
                    "args": vars(args), "iter": 0},
                   os.path.join(args.out_dir, "iter_000.pt"))
        print("[snapshot] iter_000.pt saved")

    rng = np.random.default_rng(args.seed)
    log = []
    nce_history = []   # list of (iter, layer, nce_loss)
    rank_active_history = deque(maxlen=100)
    t0 = time.time()
    iter_idx = 0
    dropout_kept_count = [0, 0, 0, 0]   # how often each layer survived dropout
    dropout_total = 0

    while iter_idx < args.iters:
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        with torch.no_grad():
            z_L = encode(backbone, lq)
            z_H = encode(backbone, gt)
        delta_oracle = [zh - zl for zh, zl in zip(z_H, z_L)]

        delta_pred = adapter.predict_delta(z_L)

        m = m_schedule(iter_idx, args.iters, args.m_anneal, rng=rng)
        delta_train = [m * d_o + (1.0 - m) * d_p.detach() for d_o, d_p in zip(delta_oracle, delta_pred)]

        # per-layer delta dropout (Branch B regularizer)
        delta_train, drop_mask = per_layer_delta_dropout(
            delta_train, args.delta_dropout, training=True, rng=rng,
        )
        for i, kept in enumerate(drop_mask):
            dropout_kept_count[i] += kept
        dropout_total += 1

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder(lq, kv_layers=z_L, delta_layers=delta_train)
        l_good = charbonnier(pred.float(), gt)

        # ranking: use FROZEN decoder so gradient does not push decoder
        # toward over-discriminating pred vs wrong (DX2/DY2)
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
            # Frozen decoder forward (params frozen but autograd active so adapter gets grad)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_good_fz = decoder_frozen(lq, kv_layers=z_L,
                                              delta_layers=[m * d_o + (1.0 - m) * d_p
                                                            for d_o, d_p in zip(delta_oracle, delta_pred)])
                pred_wrong_fz = decoder_frozen(lq, kv_layers=z_L,
                                               delta_layers=delta_wrong)
            l_good_fz = charbonnier(pred_good_fz.float(), gt)
            l_wrong_fz = charbonnier(pred_wrong_fz.float(), gt)
            L_rank = F.relu(args.ranking_margin + l_good_fz - l_wrong_fz)
            rank_active_this_step = float((L_rank > 0).item())
        rank_active_history.append(rank_active_this_step)

        # latent regularizers
        L_dir = sum(latent_dir_loss(d_p, d_o) for d_p, d_o in zip(delta_pred, delta_oracle)) / 4.0
        L_mag = sum(latent_mag_loss(d_p, d_o) for d_p, d_o in zip(delta_pred, delta_oracle)) / 4.0
        L_l2  = sum(latent_l2_loss(d_p, d_o)  for d_p, d_o in zip(delta_pred, delta_oracle)) / 4.0

        # InfoNCE on adapter outputs
        L_nce = torch.zeros((), device=device)
        nce_per_layer_this_step = {}
        if args.use_nce and iter_idx >= args.nce_warmup:
            nce_terms = []
            for l in nce_layers:
                z_q = descriptor(delta_pred[l])              # (B, D)
                z_pos = descriptor(delta_oracle[l])           # (B, D)
                q = queues[l].get()
                term = info_nce(z_q, z_pos, q, T=args.nce_T)
                nce_terms.append(term)
                nce_per_layer_this_step[l] = float(term.item())
                # push positive into queue (detach)
                queues[l].push(z_pos)
            L_nce = sum(nce_terms) / max(1, len(nce_terms))
        elif args.use_nce:
            # warmup phase: still push descriptors into queue so it fills
            with torch.no_grad():
                for l in nce_layers:
                    z_pos = descriptor(delta_oracle[l])
                    queues[l].push(z_pos)

        L = (args.rgb_weight * l_good
             + args.dir_weight * L_dir
             + args.mag_weight * L_mag
             + args.l2_weight * L_l2
             + args.ranking_weight * L_rank
             + args.nce_w * L_nce)

        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(optim_params, 1.0)
        optim.step(); sched.step()

        if (iter_idx + 1) % args.log_every == 0:
            rank_frac = float(np.mean(rank_active_history)) if rank_active_history else 0.0
            print(f"  it {iter_idx+1}/{args.iters} L={L.item():.4f} "
                  f"(rgb={l_good.item():.4f} dir={L_dir.item():.3f} mag={L_mag.item():.3f} "
                  f"l2={L_l2.item():.3f} rank={L_rank.item():.3f} nce={L_nce.item():.4f}) "
                  f"m={m:.2f} drop={sum(drop_mask)}/4 rank_active_frac={rank_frac:.2f} "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            print(f"    [scale] {adapter.gain_confidence_summary()}")
            log.append({"iter": iter_idx+1, "L": L.item(), "L_rgb": l_good.item(),
                        "L_dir": L_dir.item(), "L_mag": L_mag.item(),
                        "L_l2": L_l2.item(), "L_rank": L_rank.item(),
                        "L_nce": L_nce.item(), "m": m,
                        "rank_active_frac": rank_frac,
                        "drop_kept": sum(drop_mask),
                        "lr": optim.param_groups[0]["lr"]})
            if nce_per_layer_this_step:
                nce_history.append({"iter": iter_idx+1, **{f"L{l}": v for l, v in nce_per_layer_this_step.items()}})
            t0 = time.time()

        # snapshots + iter-400 sanity
        cur_iter = iter_idx + 1
        if cur_iter in save_iters:
            torch.save({"adapter": adapter.state_dict(),
                        "decoder": decoder.state_dict(),
                        "args": vars(args), "iter": cur_iter},
                       os.path.join(args.out_dir, f"iter_{cur_iter:03d}.pt"))
            print(f"[snapshot] iter_{cur_iter:03d}.pt saved")

        if cur_iter == sanity_iter:
            sanity_results = run_sanity_eval(
                adapter, decoder, backbone, test_seqs[:3],
                args.crop, device, train_seqs, rng,
            )
            print(f"[sanity it {cur_iter}] {sanity_results}")
            log.append({"iter": cur_iter, "sanity": sanity_results})
            # DAB1 hard-stop only on broken state
            broken = (
                (not math.isfinite(L.item())) or
                (sanity_results["oracle_minus_identity"] < args.sanity_oracle_min) or
                (rank_frac == 0.0 and args.use_ranking and args.ranking_weight > 0)
            )
            if broken:
                print(f"[sanity it {cur_iter}] HARD STOP — broken state detected:")
                print(f"  L_finite={math.isfinite(L.item())}  oracle-id={sanity_results['oracle_minus_identity']:.3f}  rank_frac={rank_frac:.2f}")
                break
            else:
                print(f"[sanity it {cur_iter}] soft-only signals; continuing per DAB1")

        iter_idx += 1

    # save final
    torch.save({"adapter": adapter.state_dict(),
                "decoder": decoder.state_dict(),
                "args": vars(args), "log": log, "iter": iter_idx},
               os.path.join(args.out_dir, "ckpt.pt"))
    if iter_idx in save_iters or "final" in args.save_iters:
        torch.save({"adapter": adapter.state_dict(),
                    "decoder": decoder.state_dict(),
                    "args": vars(args), "iter": iter_idx},
                   os.path.join(args.out_dir, "final.pt"))
        print("[snapshot] final.pt saved")

    # NCE health log
    if nce_history:
        with open(os.path.join(args.out_dir, "nce_history.csv"), "w") as f:
            keys = sorted({k for r in nce_history for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in nce_history: w.writerow(r)
        # health verdict
        early = [r for r in nce_history if 32 < r["iter"] <= 100]
        mid = [r for r in nce_history if 200 < r["iter"] <= 400]
        nce_health = {}
        for l in nce_layers:
            k = f"L{l}"
            if early and mid and k in early[0]:
                e = float(np.mean([r[k] for r in early if k in r]))
                m_v = float(np.mean([r[k] for r in mid if k in r]))
                nce_health[f"L{l}_early_mean"] = e
                nce_health[f"L{l}_mid_mean"] = m_v
                nce_health[f"L{l}_drop"] = e - m_v
                nce_health[f"L{l}_healthy"] = bool((e - m_v) >= 0.05)
        print(f"[nce_health] {nce_health}")
        with open(os.path.join(args.out_dir, "nce_health.json"), "w") as f:
            json.dump(nce_health, f, indent=2)

    # dropout stats
    if dropout_total > 0:
        kept_frac = [c / dropout_total for c in dropout_kept_count]
        print(f"[dropout] per-layer keep frac: {[f'{x:.2f}' for x in kept_frac]} (target ~{1-args.delta_dropout:.2f})")

    # ----- full eval -----
    print("\n=== FULL EVAL on LLVE-STCD-12 held-out (crop=256) ===")
    adapter.eval(); decoder.eval()
    results, audit = run_full_eval(
        adapter, decoder, backbone, test_seqs, train_seqs, args.crop, device,
        rng=np.random.default_rng(args.seed + 1),
    )

    # write per-condition CSVs
    for cond, rows in audit["per_seq"].items():
        cond_dir = os.path.join(args.out_dir, cond)
        os.makedirs(cond_dir, exist_ok=True)
        with open(os.path.join(cond_dir, "per_seq.csv"), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)

    # eval-time donor pair audit
    if audit["shuffle_pairs"]:
        with open(os.path.join(args.out_dir, "eval_shuffle_audit.csv"), "w") as f:
            f.write("current_seq,donor_seq\n")
            for r in audit["shuffle_pairs"]:
                f.write(f"{r['current_seq']},{r['donor_seq']}\n")

    # summary
    write_summary(args, results, n_a, n_d, adapter, dropout_kept_count, dropout_total)
    print("\nC++ training + same-decoder eval complete.")
    print("Run portability eval: python tools/v11_cross_decoder_probe.py --cplus_dir <out_dir>")


# ----- helpers -------------------------------------------------------
@torch.no_grad()
def run_sanity_eval(adapter, decoder, backbone, seqs, crop, device, train_seqs, rng):
    """3-seq sanity. Returns dict with identity, oracle, predicted, shuffled means
    and key derived metrics."""
    adapter_was_training = adapter.training; decoder_was_training = decoder.training
    adapter.eval(); decoder.eval()
    means = {}
    rng_e = np.random.default_rng(int(rng.integers(0, 1e9)))
    for cond in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
        psnrs = []
        for name, lq_p, gt_p in seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq); z_H = encode(backbone, gt)
            if cond == "identity_delta":
                delta = [torch.zeros_like(z) for z in z_L]
            elif cond == "oracle_delta":
                delta = [zh - zl for zh, zl in zip(z_H, z_L)]
            elif cond == "predicted_delta":
                delta = adapter.predict_delta(z_L)
            else:
                other = int(rng_e.integers(0, len(train_seqs)))
                _, lq_o, gt_o = train_seqs[other]
                lq_oo, _ = load_clip(lq_o, gt_o, crop_size=crop)
                z_L_o = encode(backbone, lq_oo.to(device))
                delta = adapter.predict_delta(z_L_o)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder(lq, kv_layers=z_L, delta_layers=delta)
            psnrs.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
        means[cond] = float(np.mean(psnrs))
    if adapter_was_training: adapter.train()
    if decoder_was_training: decoder.train()
    ident = means["identity_delta"]
    return {
        "identity": means["identity_delta"],
        "oracle": means["oracle_delta"],
        "predicted": means["predicted_delta"],
        "shuffled": means["shuffled_predicted_delta"],
        "predicted_minus_identity": means["predicted_delta"] - ident,
        "shuffled_minus_identity": means["shuffled_predicted_delta"] - ident,
        "oracle_minus_identity": means["oracle_delta"] - ident,
        "leakage_ratio": (means["shuffled_predicted_delta"] - ident) / max(1e-6, means["predicted_delta"] - ident),
    }


@torch.no_grad()
def run_full_eval(adapter, decoder, backbone, test_seqs, train_seqs, crop, device, rng):
    audit = {"per_seq": {}, "shuffle_pairs": []}
    results = {}
    for cond in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
        rows = []
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
            lq, gt = lq.to(device), gt.to(device)
            z_L = encode(backbone, lq); z_H = encode(backbone, gt)
            if cond == "identity_delta":
                delta = [torch.zeros_like(z) for z in z_L]
            elif cond == "oracle_delta":
                delta = [zh - zl for zh, zl in zip(z_H, z_L)]
            elif cond == "predicted_delta":
                delta = adapter.predict_delta(z_L)
            else:
                other = int(rng.integers(0, len(train_seqs)))
                donor_name, lq_o, gt_o = train_seqs[other]
                assert donor_name != name, "shuffle audit: donor must differ"
                audit["shuffle_pairs"].append({"current_seq": name, "donor_seq": donor_name})
                lq_oo, _ = load_clip(lq_o, gt_o, crop_size=crop)
                z_L_o = encode(backbone, lq_oo.to(device))
                delta = adapter.predict_delta(z_L_o)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder(lq, kv_layers=z_L, delta_layers=delta)
            psnr = mean_psnr(to_u8(pred.float()), to_u8(gt))
            psnr_lq = mean_psnr(to_u8(lq), to_u8(gt))
            rows.append({"seq": name, "lq_psnr": psnr_lq, "pred_psnr": psnr})
            print(f"    [{cond}/{name}] LQ={psnr_lq:.2f} pred={psnr:.3f}")
        audit["per_seq"][cond] = rows
        results[cond] = float(np.mean([r["pred_psnr"] for r in rows]))
        print(f"  -> {cond}: mean PSNR = {results[cond]:.3f}")
    return results, audit


def write_summary(args, results, n_a, n_d, adapter, dropout_kept_count, dropout_total):
    md = os.path.join(args.out_dir, "JDAP_CPLUSPLUS_SUMMARY.md")
    with open(md, "w") as f:
        f.write("# v11 JDAP-C++ Results (DAA/DAB/DAC)\n\n")
        f.write(f"adapter: {n_a/1e6:.2f}M | decoder: {n_d/1e6:.2f}M\n")
        f.write(f"iters: {args.iters} | lr: {args.lr} | crop: {args.crop}\n")
        f.write(f"m anneal: {args.m_anneal}, ranking: w={args.ranking_weight} margin={args.ranking_margin} (FROZEN-decoder DX2/DY2)\n")
        f.write(f"delta_dropout: p={args.delta_dropout} per-layer\n")
        f.write(f"NCE: layers={args.nce_layers} T={args.nce_T} w={args.nce_w} queue={args.nce_queue_size} warmup_iter={args.nce_warmup}\n\n")
        f.write("| condition | mean PSNR | Δ vs identity |\n|---|---:|---:|\n")
        ident = results.get("identity_delta", float("nan"))
        for c in ["identity_delta", "oracle_delta", "predicted_delta", "shuffled_predicted_delta"]:
            d = results[c] - ident if not np.isnan(ident) else float("nan")
            f.write(f"| {c} | {results[c]:.3f} | {d:+.3f} |\n")

        f.write("\n## Gates (DAA2 / DAC criteria)\n\n")
        ident = results["identity_delta"]
        pred = results["predicted_delta"] - ident
        shuf = results["shuffled_predicted_delta"] - ident
        oracle = results["oracle_delta"] - ident
        leak = shuf / max(1e-6, pred)
        sep = pred - shuf
        f.write(f"- G1 floor (predicted - identity ≥ +3.0):  {pred:+.3f} → **{'PASS' if pred >= 3.0 else 'FAIL'}**\n")
        f.write(f"- G1 strong (predicted - identity ≥ +5.0): {pred:+.3f} → **{'PASS' if pred >= 5.0 else 'FAIL'}**\n")
        f.write(f"- G2 killer (shuffled - identity ≤ +0.5):  {shuf:+.3f} → **{'PASS' if shuf <= 0.5 else 'FAIL'}**\n")
        f.write(f"- G3 separation (predicted - shuffled ≥ +2.0): {sep:+.3f} → **{'PASS' if sep >= 2.0 else 'FAIL'}**  (DAB4: implied by other gates)\n")
        f.write(f"- G4 oracle (oracle ≥ identity):           {oracle:+.3f} → **{'PASS' if oracle >= 0.0 else 'FAIL'}**\n")
        f.write(f"- G5 leakage ratio (≤ 0.20):               {leak:.3f} → **{'PASS' if leak <= 0.20 else 'FAIL'}**\n")

        f.write("\n## Diagnostics\n\n")
        f.write(f"final scale-aware adapter: {adapter.gain_confidence_summary()}\n\n")
        if dropout_total > 0:
            kept = [c / dropout_total for c in dropout_kept_count]
            f.write(f"dropout per-layer keep frac: {[round(x, 2) for x in kept]} (target ~{1 - args.delta_dropout:.2f})\n")

        f.write("\n## Portability\n\n")
        f.write("Run `tools/v11_cross_decoder_probe.py` against this run for cross-decoder gains and `portable_pred_gain` / `portable_sep`.\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/v11_jdap_cplusplus_1000")
    ap.add_argument("--adapter_init", default="/cache/zihaohe/checkpoint/v11_probes/v11_jdap_cplus_1000/ckpt.pt")
    ap.add_argument("--decoder_init", default="/cache/zihaohe/checkpoint/v11_probes/v11_jdap_cplus_1000/ckpt.pt")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m_anneal", default="random_mix",
                    choices=["linear", "linear_floor", "cosine", "random_mix"])
    ap.add_argument("--rgb_weight", type=float, default=1.0)
    ap.add_argument("--dir_weight", type=float, default=0.05)
    ap.add_argument("--mag_weight", type=float, default=0.02)
    ap.add_argument("--l2_weight", type=float, default=0.01)
    ap.add_argument("--use_ranking", action="store_true", default=True)
    ap.add_argument("--ranking_margin", type=float, default=0.05)
    ap.add_argument("--ranking_weight", type=float, default=1.0)
    # NEW for C++
    ap.add_argument("--delta_dropout", type=float, default=0.20,
                    help="per-layer dropout prob; DAB1/DY4 = 0.20")
    ap.add_argument("--use_nce", action="store_true", default=True)
    ap.add_argument("--nce_layers", default="2,3", help="adapter indices (0..3)")
    ap.add_argument("--nce_T", type=float, default=0.07)
    ap.add_argument("--nce_w", type=float, default=0.1)
    ap.add_argument("--nce_queue_size", type=int, default=32)
    ap.add_argument("--nce_warmup", type=int, default=32)
    ap.add_argument("--save_iters", default="0,400", help="iters at which to save snapshot (final always saved)")
    ap.add_argument("--sanity_iter", type=int, default=400, help="iter at which to run 3-seq sanity (DAB1: soft-only)")
    ap.add_argument("--freeze_decoder", action="store_true", default=False,
                    help="Option B / DAE4: freeze decoder, train adapter only against fixed oracle decoder")
    ap.add_argument("--sanity_oracle_min", type=float, default=2.0,
                    help="DAD1/CN-fix: 3-seq sanity oracle-id threshold; was 5.0 (12-seq full eval gate); 2.0 for sanity")
    main(ap.parse_args())
