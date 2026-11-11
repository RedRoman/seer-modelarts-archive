"""J1-v2: VLR-P with DBE/DBF amendments.

Key changes vs J1:
  - gate init = 0.3 (not 0.0) so predictor path engages from step 0
  - Add L_align = -cos(z_HQ_pred, z_H_target) where z_H_target = E_vjepa(GT)[3]
  - Add L_anchor = mean ||z_HQ_pred - z_LQ||² / N for bounded predictor update
  - Separate AdamW group: LoRA + new_proj + gate at lr=5e-4; decoder at lr=2e-4
  - Per-100-iter latent diagnostic on a fixed val clip:
      cos(z_HQ_pred, z_H), cos(z_LQ, z_H), delta_cos
  - Smoke G6: LoRA + new_proj grad norm > 0
  - Mid-train abort at iter 500: PSNR<17 OR delta_cos<0 OR |gate|<0.02
  - Final eval has 5 load-bearing controls per DBF5:
      full_vlrp / gate_zero / raw_zL / shuffled_refined / no_prior
"""
from __future__ import annotations
import argparse, copy, os, sys, time
from typing import List

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


def cos_mean(a, b):
    """Token-wise cosine, mean over (B,N). Handles bf16/fp32."""
    a = F.normalize(a.float(), dim=-1, eps=1e-3)
    b = F.normalize(b.float(), dim=-1, eps=1e-3)
    return float((a * b).sum(dim=-1).clamp(-1.5, 1.5).mean().item())


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


def diagnostic_cos(refiner, backbone, lq, gt, ref_layer):
    """Returns (cos_pred_H, cos_L_H, delta_cos) on a fixed val clip."""
    with torch.no_grad():
        z_L = encode(backbone, lq)[ref_layer].float()
        z_H = encode(backbone, gt)[ref_layer].float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_HQ_pred = refiner(z_L.bfloat16()).float()
        cph = cos_mean(z_HQ_pred, z_H)
        clh = cos_mean(z_L, z_H)
    return cph, clh, cph - clh


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
    # DBF1: gate init = 0.3 (override the default 0.0 in CounterfactualRefiner)
    with torch.no_grad():
        refiner.gate.fill_(args.gate_init)
    n_refiner = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    n_refiner_total = sum(p.numel() for p in refiner.parameters())
    print(f"[refiner] gate init={float(refiner.gate.item()):.3f} | "
          f"{n_refiner/1e6:.2f}M trainable / {n_refiner_total/1e6:.1f}M total")

    decoder = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[VLR-P-v2] decoder {n_d/1e6:.2f}M; total trainable {(n_d + n_refiner)/1e6:.2f}M")

    # Separate optim groups: LoRA + new_proj + gate at higher lr
    refiner_params = [p for p in refiner.parameters() if p.requires_grad]
    decoder_params = list(decoder.parameters())
    optim = torch.optim.AdamW(
        [
            {"params": decoder_params, "lr": args.lr},
            {"params": refiner_params, "lr": args.lr_refiner},
        ],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Fix one held-out sample for diagnostic
    _name_diag, _lq_diag, _gt_diag = test_seqs[0]
    diag_lq, diag_gt = load_clip(_lq_diag, _gt_diag, crop_size=args.crop)
    diag_lq, diag_gt = diag_lq.to(device), diag_gt.to(device)
    print(f"[diag] using {_name_diag} as fixed validation clip")

    log = []
    smoke_record = {}
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        _, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)

        # 1. V-JEPA encode LQ AND GT (frozen)
        with torch.no_grad():
            z_L = encode(backbone, lq)[args.ref_layer].float()
            z_H_target = encode(backbone, gt)[args.ref_layer].float()

        # 2. Predictor + LoRA refines
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_HQ_pred = refiner(z_L.bfloat16())
            z_HQ_pred = z_HQ_pred.float()

        # 3. Decoder
        kv_layers = [z_HQ_pred]
        h_b, skips = decoder.encode(lq, kv_layers=kv_layers, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)

        # 4. Losses
        L_rgb = charbonnier(pred.float(), gt)
        L_align = safe_cos_loss(z_HQ_pred, z_H_target)
        L_anchor = ((z_HQ_pred - z_L) ** 2).mean()
        L_total = L_rgb + args.lambda_align * L_align + args.lambda_anchor * L_anchor

        optim.zero_grad(set_to_none=True); L_total.backward()

        # G6 diagnostic: LoRA/new_proj grad norm
        ref_grad_norm = 0.0
        for p in refiner_params:
            if p.grad is not None:
                ref_grad_norm += float(p.grad.norm().item()) ** 2
        ref_grad_norm = ref_grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(decoder_params + refiner_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            cos_p_H = cos_mean(z_HQ_pred.detach(), z_H_target)
            cos_L_H = cos_mean(z_L, z_H_target)
            dcos = cos_p_H - cos_L_H
            row = {
                "iter": it+1, "L_total": L_total.item(),
                "L_rgb": L_rgb.item(), "L_align": L_align.item(),
                "L_anchor": L_anchor.item(),
                "gate": float(refiner.gate.detach().abs().item()),
                "cos_pH": cos_p_H, "cos_LH": cos_L_H, "dcos": dcos,
                "ref_grad_norm": ref_grad_norm,
            }
            log.append(row)
            print(f"  it {it+1}/{args.iters} L_rgb={L_rgb.item():.4f} L_align={L_align.item():.4f} "
                  f"L_anchor={L_anchor.item():.4f} |gate|={row['gate']:.4f} "
                  f"cos(pred,H)={cos_p_H:.4f} cos(L,H)={cos_L_H:.4f} Δcos={dcos:+.4f} "
                  f"refgrad={ref_grad_norm:.4f} dt={time.time()-t0:.1f}s")
            t0 = time.time()
            # Smoke records at iter 5 and iter 30
            if (it + 1) in (5, 30):
                cph, clh, dc = diagnostic_cos(refiner, backbone, diag_lq, diag_gt, args.ref_layer)
                smoke_record[it + 1] = {
                    "L_align": L_align.item(),
                    "gate": row["gate"],
                    "dcos_diag": dc,
                    "cos_pH_diag": cph,
                    "cos_LH_diag": clh,
                    "ref_grad_norm": ref_grad_norm,
                    "L_rgb": L_rgb.item(),
                }

        # DBF7 mid-train abort
        if (it + 1) == 500 and args.iters > 500:
            psnr_mid, _ = _eval_mode(refiner, decoder, backbone, test_seqs[:3],
                                     device, args.crop, args.ref_layer, "full")
            cph, clh, dc = diagnostic_cos(refiner, backbone, diag_lq, diag_gt, args.ref_layer)
            gate_now = float(refiner.gate.detach().abs().item())
            print(f"\n[MID-EVAL @500] PSNR(3 seqs)={psnr_mid:.3f}  Δcos={dc:+.4f}  |gate|={gate_now:.4f}")
            if psnr_mid < 17.0 or dc < 0 or gate_now < 0.02:
                print(f"[ABORT] mid-train gates failed; PSNR<17 OR Δcos<0 OR gate<0.02")
                _write_abort_md(args.out_dir, psnr_mid, dc, gate_now, log, smoke_record)
                sys.exit(2)

    torch.save({"decoder": decoder.state_dict(), "refiner": refiner.state_dict(),
                "args": vars(args), "log": log, "smoke": smoke_record,
                "stage": "vlrp_v2"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # ----- Final eval with 5 controls per DBF5 -----
    decoder.eval(); refiner.eval()
    print("\n=== FINAL EVAL: 5 load-bearing controls ===")
    results = {}
    for mode in ["full", "gate_zero", "raw_zL", "shuffled", "no_prior"]:
        psnr, per = _eval_mode(refiner, decoder, backbone, test_seqs, device,
                               args.crop, args.ref_layer, mode)
        results[mode] = {"mean": psnr, "per_seq": per}
        print(f"  [{mode:14s}] mean PSNR = {psnr:.3f}")

    cph, clh, dc = diagnostic_cos(refiner, backbone, diag_lq, diag_gt, args.ref_layer)
    gate_final = float(refiner.gate.detach().abs().item())

    md = os.path.join(args.out_dir, "VLRP_V2_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# VLR-P v2 Final Eval (with DBF5 load-bearing controls)\n\n")
        f.write(f"V-JEPA encoder 304M + predictor body ~150M (frozen)\n")
        f.write(f"+ refiner trainable {n_refiner/1e6:.2f}M + decoder {n_d/1e6:.2f}M\n\n")
        f.write(f"## Final diagnostics\n")
        f.write(f"- |gate| final           = {gate_final:.4f}\n")
        f.write(f"- cos(z_HQ_pred, z_H)    = {cph:.4f}\n")
        f.write(f"- cos(z_LQ,     z_H)     = {clh:.4f}\n")
        f.write(f"- Δcos (mechanism move)  = {dc:+.4f}\n\n")
        f.write(f"## PSNR by mode (DBF5 controls)\n\n")
        f.write(f"| mode | mean PSNR | δ vs full |\n|---|---|---|\n")
        for mode in ["full", "gate_zero", "raw_zL", "shuffled", "no_prior"]:
            d = results[mode]["mean"] - results["full"]["mean"]
            f.write(f"| {mode} | {results[mode]['mean']:.3f} | {d:+.3f} |\n")
        f.write(f"\n## DBF6 mainline gates\n")
        f.write(f"- full >= 19.56              : {'PASS' if results['full']['mean'] >= 19.56 else 'FAIL'}\n")
        f.write(f"- |gate| >= 0.05            : {'PASS' if gate_final >= 0.05 else 'FAIL'}\n")
        f.write(f"- Δcos > 0                  : {'PASS' if dc > 0 else 'FAIL'}\n")
        f.write(f"- full - gate_zero >= +0.5 : {'PASS' if (results['full']['mean'] - results['gate_zero']['mean']) >= 0.5 else 'FAIL'}\n")
        f.write(f"- full - raw_zL >= +0.3    : {'PASS' if (results['full']['mean'] - results['raw_zL']['mean']) >= 0.3 else 'FAIL'}\n")
        f.write(f"- full - shuffled >= +0.5  : {'PASS' if (results['full']['mean'] - results['shuffled']['mean']) >= 0.5 else 'FAIL'}\n")
    print(f"\nVLR-P v2 full PSNR = {results['full']['mean']:.3f}")
    print(f"  vs gate_zero        = {results['gate_zero']['mean']:.3f}  (δ={results['full']['mean']-results['gate_zero']['mean']:+.3f})")
    print(f"  vs raw_zL           = {results['raw_zL']['mean']:.3f}  (δ={results['full']['mean']-results['raw_zL']['mean']:+.3f})")
    print(f"  vs shuffled_refined = {results['shuffled']['mean']:.3f}  (δ={results['full']['mean']-results['shuffled']['mean']:+.3f})")
    print(f"  vs no_prior         = {results['no_prior']['mean']:.3f}  (δ={results['full']['mean']-results['no_prior']['mean']:+.3f})")
    print(f"  Δcos final = {dc:+.4f}, |gate| final = {gate_final:.4f}")
    print(f"wrote {md}")


@torch.no_grad()
def _eval_mode(refiner, decoder, backbone, test_seqs, device, crop, ref_layer, mode):
    psnrs = []
    n_seqs = len(test_seqs)
    for i, (name, lq_p, gt_p) in enumerate(test_seqs):
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        z_L = encode(backbone, lq)[ref_layer].float()
        if mode == "full":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_HQ_pred = refiner(z_L.bfloat16()).float()
            kv = [z_HQ_pred]
        elif mode == "gate_zero":
            saved = refiner.gate.data.clone()
            refiner.gate.data.zero_()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_HQ_pred = refiner(z_L.bfloat16()).float()
            refiner.gate.data.copy_(saved)
            kv = [z_HQ_pred]
        elif mode == "raw_zL":
            kv = [z_L]
        elif mode == "shuffled":
            j = (i + 1) % n_seqs   # use next sequence's z_L → refined
            _, lq_d_p, gt_d_p = test_seqs[j]
            lq_d, _ = load_clip(lq_d_p, gt_d_p, crop_size=crop)
            lq_d = lq_d.to(device)
            z_L_donor = encode(backbone, lq_d)[ref_layer].float()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_HQ_pred = refiner(z_L_donor.bfloat16()).float()
            kv = [z_HQ_pred]
        elif mode == "no_prior":
            kv = None
        else:
            raise ValueError(mode)

        h_b, skips = decoder.encode(lq, kv_layers=kv, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=kv, delta_layers=None)
        psnrs.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
    return float(np.mean(psnrs)), psnrs


def _write_abort_md(out_dir, psnr_mid, dcos, gate, log, smoke):
    md = os.path.join(out_dir, "VLRP_V2_ABORT.md")
    with open(md, "w") as f:
        f.write(f"# VLR-P v2 ABORTED at iter 500\n\n")
        f.write(f"PSNR(3-seq mid)= {psnr_mid:.3f}, Δcos={dcos:+.4f}, |gate|={gate:.4f}\n")
        f.write(f"Smoke records: {smoke}\n")
        f.write(f"Last log row: {log[-1] if log else 'none'}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_refiner", type=float, default=5e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--gate_init", type=float, default=0.3)
    ap.add_argument("--lambda_align", type=float, default=0.5)
    ap.add_argument("--lambda_anchor", type=float, default=0.01)
    main(ap.parse_args())
