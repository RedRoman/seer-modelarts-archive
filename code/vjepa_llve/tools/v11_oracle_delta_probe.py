"""tools/v11_oracle_delta_probe.py — CZ4 / DH5 Probe C orchestrator.

Trains the SAME DeltaModulatedDecoder under different prior conditions and
evaluates RGB-PSNR on sequence-disjoint LLVE-STCD-12 held-out.

Conditions (DH5):
  no_prior        : decoder(LQ)
  passive_zL      : decoder(LQ, kv=z_L)
  identity_delta  : decoder(LQ, kv=z_L, delta=0)
  shuffled_delta  : decoder(LQ, kv=z_L, delta=delta_from_other_seq_in_train)
  oracle_zH       : decoder(LQ, kv=z_H)
  oracle_delta    : decoder(LQ, kv=z_L, delta=z_H-z_L)
  predicted_delta : decoder(LQ, kv=z_L, delta=A(z_L))   [adapter frozen]

Spec: V11_PROBE_C_SPEC.md
"""
from __future__ import annotations
import argparse, csv, os, sys, time
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
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.v11_delta_stats import (
    LLVE_STCD_12, INDOOR_ROOT, OUTDOOR_ROOT, index_seqs,
)


CONDITIONS = ["no_prior", "passive_zL", "identity_delta", "shuffled_delta",
              "oracle_delta", "predicted_delta", "oracle_zH"]


def load_clip(lq_paths, gt_paths, n_frames=16, crop_size=256):
    n_total = len(lq_paths); T = min(n_frames, n_total)
    start = (n_total - T) // 2
    lq = np.stack([read_image_01(lq_paths[start+i]) for i in range(T)], axis=0)
    gt = np.stack([read_image_01(gt_paths[start+i]) for i in range(T)], axis=0)
    lq = torch.from_numpy(lq).permute(3,0,1,2).unsqueeze(0).float()
    gt = torch.from_numpy(gt).permute(3,0,1,2).unsqueeze(0).float()
    # random or center crop to crop_size for training stability and v-jepa fixed-size
    H, W = lq.shape[-2:]
    if H >= crop_size and W >= crop_size:
        h0, w0 = (H-crop_size)//2, (W-crop_size)//2
        lq = lq[..., h0:h0+crop_size, w0:w0+crop_size]
        gt = gt[..., h0:h0+crop_size, w0:w0+crop_size]
    return lq, gt


def load_clip_eval(lq_paths, gt_paths, n_frames=16, crop_size=256):
    """DJ3: fixed center-crop 256 eval. Same protocol across all conditions.
    Avoids OOM and matches V-JEPA wrapper internal resize. Recorded in summary."""
    return load_clip(lq_paths, gt_paths, n_frames=n_frames, crop_size=crop_size)


def to_u8(x): return (x[0].permute(1,2,3,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)


def rgb_psnr_uint8(a, b, eps=1e-10):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(255**2 / max(mse, eps))


def mean_psnr(p, g): return float(np.mean([rgb_psnr_uint8(p[t], g[t]) for t in range(p.shape[0])]))


@torch.no_grad()
def encode(backbone, vid):
    return [f.float() for f in backbone(vid)["feats"]]


@torch.no_grad()
def get_kv_delta(condition: str, z_L: List[torch.Tensor], z_H: List[torch.Tensor],
                 z_L_other: Optional[List[torch.Tensor]] = None,
                 z_H_other: Optional[List[torch.Tensor]] = None,
                 adapter: Optional[nn.Module] = None
                 ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    if condition == "no_prior":
        return None, None
    if condition == "passive_zL":
        return z_L, None
    if condition == "identity_delta":
        return z_L, [torch.zeros_like(z) for z in z_L]
    if condition == "shuffled_delta":
        # delta from a different sequence's z_H - z_L
        delta = [zh_o - zl_o for zh_o, zl_o in zip(z_H_other, z_L_other)]
        return z_L, delta
    if condition == "oracle_zH":
        return z_H, None
    if condition == "oracle_delta":
        delta = [zh - zl for zh, zl in zip(z_H, z_L)]
        return z_L, delta
    if condition == "predicted_delta":
        assert adapter is not None
        delta = adapter.predict_delta(z_L)   # returns delta only (not z_L+delta)
        return z_L, delta
    raise ValueError(f"unknown condition: {condition}")


def charbonnier(x, y, eps=1e-3):
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def train_one_condition(condition: str, args, backbone, adapter, device,
                        train_seqs, test_seqs, out_dir,
                        base_state=None, init_sha=None):
    os.makedirs(out_dir, exist_ok=True)
    decoder = DeltaModulatedDecoder(delta_in_encoder=getattr(args, "delta_in_encoder", True)).to(device)
    if base_state is not None:
        # DK4 fix: load identical initial weights so condition differences come
        # from the input, not from random init.
        decoder.load_state_dict({k: v.to(device) for k, v in base_state.items()})
        with open(os.path.join(out_dir, "init_audit.txt"), "w") as _f:
            _f.write(f"init_sha256_first16={init_sha}\nshared_init=YES\n")
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)
    print(f"\n=== condition: {condition} (decoder {n_params/1e6:.1f}M) ===")

    # DK4 fix: same per-iter seq sampling order across conditions (same train rng).
    rng = np.random.default_rng(args.seed)
    iter_idx = 0
    log = []
    t0 = time.time()
    while iter_idx < args.iters:
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)
        with torch.no_grad():
            z_L = encode(backbone, lq)
            z_H = encode(backbone, gt)
        z_L_o = z_H_o = None
        donor_name = None
        if condition == "shuffled_delta":
            other = int(rng.integers(0, len(train_seqs)))
            while other == idx and len(train_seqs) > 1:
                other = int(rng.integers(0, len(train_seqs)))
            assert other != idx, "DJ6 shuffle violation"
            donor_name, lq_o, gt_o = train_seqs[other]
            lq_oo, gt_oo = load_clip(lq_o, gt_o, crop_size=args.crop)
            with torch.no_grad():
                z_L_o = encode(backbone, lq_oo.to(device))
                z_H_o = encode(backbone, gt_oo.to(device))

        kv, delta = get_kv_delta(condition, z_L, z_H, z_L_o, z_H_o, adapter)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder(lq, kv_layers=kv, delta_layers=delta)
            loss = charbonnier(pred.float(), gt)
        optim.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optim.step(); sched.step()

        if (iter_idx + 1) % args.log_every == 0:
            print(f"  it {iter_idx+1}/{args.iters} loss={loss.item():.4f} "
                  f"lr={optim.param_groups[0]['lr']:.2e} dt={time.time()-t0:.1f}s")
            log.append({"iter": iter_idx+1, "loss": loss.item(),
                        "lr": optim.param_groups[0]['lr']})
            t0 = time.time()
        # DJ6: write donor audit for first 10 train iters of shuffled_delta
        if condition == "shuffled_delta" and iter_idx < 10:
            audit_path = os.path.join(out_dir, "shuffle_audit.csv")
            mode = "w" if iter_idx == 0 else "a"
            with open(audit_path, mode) as _f:
                if iter_idx == 0: _f.write("iter,current_seq,donor_seq\n")
                _f.write(f"{iter_idx},{name},{donor_name}\n")
        iter_idx += 1

    # save and eval
    torch.save({"decoder": decoder.state_dict(),
                "condition": condition, "args": vars(args), "log": log},
               os.path.join(out_dir, "ckpt.pt"))

    # eval RGB-PSNR on held-out
    print(f"  evaluating {condition} on {len(test_seqs)} held-out clips...")
    decoder.eval()
    rows = []
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip_eval(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)
        with torch.no_grad():
            z_L = encode(backbone, lq)
            z_H = encode(backbone, gt)
            z_L_o = z_H_o = None
            if condition == "shuffled_delta":
                other = int(rng.integers(0, len(train_seqs)))
                _, lq_o, gt_o = train_seqs[other]
                lq_oo, gt_oo = load_clip(lq_o, gt_o, crop_size=args.crop)
                z_L_o = encode(backbone, lq_oo.to(device))
                z_H_o = encode(backbone, gt_oo.to(device))
            kv, delta = get_kv_delta(condition, z_L, z_H, z_L_o, z_H_o, adapter)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder(lq, kv_layers=kv, delta_layers=delta)
        pred_u8, gt_u8, lq_u8 = to_u8(pred.float()), to_u8(gt), to_u8(lq)
        psnr = mean_psnr(pred_u8, gt_u8); psnr_lq = mean_psnr(lq_u8, gt_u8)
        rows.append({"seq": name, "lq_psnr": psnr_lq, "pred_psnr": psnr,
                     "delta_vs_lq": psnr - psnr_lq})
        print(f"    [{name}] LQ={psnr_lq:.2f} pred={psnr:.3f}")
    csv_path = os.path.join(out_dir, "per_seq.csv")
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    mean_psnr_val = float(np.mean([r["pred_psnr"] for r in rows]))
    mean_lq_val = float(np.mean([r["lq_psnr"] for r in rows]))
    print(f"  -> {condition}: mean PSNR = {mean_psnr_val:.3f} (LQ={mean_lq_val:.3f})")
    return mean_psnr_val, rows


def build_split_seqs():
    """Same as Probe B': train = SDSD (in+out) MINUS LLVE-STCD-12; held-out = LLVE-STCD-12."""
    indoor_full = index_seqs(INDOOR_ROOT, "train")
    outdoor_full = index_seqs(OUTDOOR_ROOT, "train")
    test_names = set(LLVE_STCD_12)
    train_seqs = []
    for name, (lq, gt) in indoor_full.items():
        if name not in test_names: train_seqs.append((name, lq, gt))
    for name, (lq, gt) in outdoor_full.items():
        if name not in test_names: train_seqs.append((name, lq, gt))
    test_seqs = [(name, *indoor_full[name]) for name in LLVE_STCD_12 if name in indoor_full]
    train_names = set(s[0] for s in train_seqs)
    held_names = set(s[0] for s in test_seqs)
    assert not (train_names & held_names), 'leakage'
    print(f"[split] train={len(train_seqs)} held-out={len(test_seqs)} disjoint=YES")
    return train_seqs, test_seqs


def main(args):
    device = torch.device("cuda")
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)
    # DJ7 audit: backbone frozen
    assert all(not p.requires_grad for p in backbone.parameters()), "backbone not frozen"

    adapter = None
    if any(c == "predicted_delta" for c in args.conditions.split(",")):
        adapter = JEPADeltaAdapter().to(device).eval()
        sd = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(sd["adapter"])
        for p in adapter.parameters(): p.requires_grad_(False)
        print(f"[adapter] loaded frozen from {args.adapter_ckpt}")
        assert all(not p.requires_grad for p in adapter.parameters()), "adapter not frozen"

    train_seqs, test_seqs = build_split_seqs()
    # DK4 fix: ONE base decoder init shared by every condition.
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    _ref_decoder = DeltaModulatedDecoder(delta_in_encoder=getattr(args, "delta_in_encoder", True))
    _ref_n = sum(p.numel() for p in _ref_decoder.parameters() if p.requires_grad)
    assert _ref_n <= 80_000_000, f"DI4 violation: decoder {_ref_n/1e6:.1f}M > 80M target"
    base_init_path = os.path.join(args.out_dir, "init_decoder.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(_ref_decoder.state_dict(), base_init_path)
    base_state = {k: v.clone() for k, v in _ref_decoder.state_dict().items()}
    del _ref_decoder
    import hashlib
    init_sha = hashlib.sha256(b"".join(t.cpu().numpy().tobytes() for t in base_state.values())).hexdigest()[:16]
    print(f"[audit] decoder trainable params: {_ref_n/1e6:.2f}M (≤ 80M target ✓)")
    print(f"[audit] base init saved to {base_init_path}; init_sha256[:16]={init_sha}")

    out_root = args.out_dir
    os.makedirs(out_root, exist_ok=True)
    results = {}
    conds = args.conditions.split(",")
    for c in conds:
        c = c.strip()
        if c not in CONDITIONS:
            print(f"WARN unknown condition {c!r}, skipping"); continue
        out_dir = os.path.join(out_root, c)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "param_count.txt"), "w") as _f:
            _f.write(f"decoder_trainable_params={_ref_n}\n")
        with open(os.path.join(out_dir, "split_audit.txt"), "w") as _f:
            _f.write(f"train={len(train_seqs)}\nheld_out={len(test_seqs)}\nintersection=0\n")
        if os.path.isfile(os.path.join(out_dir, "per_seq.csv")) and not args.overwrite:
            import csv as _csv
            with open(os.path.join(out_dir, "per_seq.csv")) as f:
                rows = list(_csv.DictReader(f))
            mean = float(np.mean([float(r["pred_psnr"]) for r in rows]))
            print(f"[skip-existing] {c} mean PSNR = {mean:.3f}")
            results[c] = mean; continue
        mean, _ = train_one_condition(c, args, backbone, adapter, device,
                                       train_seqs, test_seqs, out_dir,
                                       base_state=base_state, init_sha=init_sha)
        results[c] = mean
        torch.cuda.empty_cache()

    # cross-condition summary
    md = os.path.join(out_root, "PROBE_C_SUMMARY.md")
    with open(md, "w") as f:
        f.write("# Probe C — Oracle-Delta Decoder Results\n\n")
        f.write(f"decoder: DeltaModulatedDecoder (~34M trainable)\n")
        f.write(f"iters per condition: {args.iters} | lr={args.lr} | crop={args.crop}\n")
        f.write(f"held-out: LLVE-STCD-12 ({len(test_seqs)} seqs), seq-disjoint\n")
        f.write(f"eval protocol: fixed center crop {args.crop} (DJ3) — NOT tile=512 full-frame\n")
        f.write(f"shared decoder init (DK4): YES, sha256[:16]={init_sha}\n")
        if args.iters < 500:
            f.write(f"\n**LABEL**: smoke-only / crop-{args.crop} pre-gate — NOT final Probe C gate.\n\n")
        else:
            f.write(f"\n")
        f.write("| condition | mean RGB-PSNR | Δ vs passive_zL |\n|---|---:|---:|\n")
        passive = results.get("passive_zL", float("nan"))
        for c in CONDITIONS:
            if c not in results: continue
            d = results[c] - passive if not np.isnan(passive) else float("nan")
            f.write(f"| {c} | {results[c]:.3f} | {d:+.3f} |\n")
        f.write("\n## DH7 gates\n\n")
        if "passive_zL" in results and "oracle_delta" in results:
            gain = results["oracle_delta"] - results["passive_zL"]
            f.write(f"- Primary: oracle_delta - passive_zL = {gain:+.3f} dB "
                    f"(gate: ≥ +0.7) -> **{'PASS' if gain>=0.7 else 'FAIL'}**\n")
        if "passive_zL" in results and "predicted_delta" in results and "oracle_delta" in results:
            full = results["oracle_delta"] - results["passive_zL"]
            pred = results["predicted_delta"] - results["passive_zL"]
            recover = pred / full if abs(full) > 1e-6 else float("nan")
            f.write(f"- Secondary: predicted recovers {recover*100:.0f}% of oracle gain "
                    f"(gate: ≥ 30-40%) -> **{'PASS' if recover>=0.30 else 'FAIL'}**\n")
        if "passive_zL" in results and "shuffled_delta" in results:
            shuf_gap = results["shuffled_delta"] - results["passive_zL"]
            killer_pass = abs(shuf_gap) <= 0.15
            f.write(f"- Killer: shuffled_delta vs passive_zL = {shuf_gap:+.3f} dB "
                    f"(should be ~0) -> **{'PASS' if killer_pass else 'FAIL — JDAP CLAIM DIES'}**\n")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", default="/cache/zihaohe/checkpoint/v11_probes/probe_c")
    ap.add_argument("--adapter_ckpt", default="/cache/zihaohe/checkpoint/v11_probes/probe_b_prime/delta_adapter.pt")
    ap.add_argument("--conditions", default="passive_zL,identity_delta,shuffled_delta,oracle_delta,predicted_delta,oracle_zH",
                    help="comma-separated subset of CONDITIONS")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--delta_in_encoder", type=int, default=1, help="DAH2.C/DAI1: set 0 for V2 (delta only in decoder)")
    args = ap.parse_args()
    main(args)
