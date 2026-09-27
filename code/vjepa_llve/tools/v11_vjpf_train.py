"""VJ-Predictor-Flow (v3): use V-JEPA's pretrained PREDICTOR as the velocity field.

Architecture:
  V-JEPA encoder (frozen 304M)
    LQ ─► z_LQ (B, 2048, 1024)
    GT ─► z_HQ (B, 2048, 1024)

  CounterfactualRefiner (V-JEPA predictor body ~150M FROZEN + LoRA ~1.5M)
    Wrapper acts as the rectified-flow velocity field:
      v_pred(z_t, t) = refiner(z_t * pos_t + bias_t)
    where pos_t/bias_t are tiny time-conditioning module (~50K params)
    output: v_pred is the flow velocity, NOT a direct refined latent

  Rectified flow training:
    sample t ~ U(0,1), z_t = (1-t)·z_LQ + t·z_HQ, v_target = z_HQ - z_LQ
    L_flow = MSE(v_pred, v_target)
  Inference: 4-step Euler ODE same as VJLF/VJM

  DeltaModulatedDecoder (33M render head, identical to VJLF v1 / VJM v1)
    train: kv=[z_HQ] (oracle decoupled)
    eval:  kv=[z_HQ_pred from refiner-flow]
    REPA + HASTE on bottleneck h_b ↔ V-JEPA(GT) layer 3

Key novelty:
  - VJLF v1: from-scratch DiT learns velocity field
  - VJ-Mamba v1: from-scratch Mamba learns velocity field
  - VJ-Predictor-Flow: REUSE V-JEPA's own pretrained predictor (frozen+LoRA) as velocity field
    → most V-JEPA-native of the three; predictor was trained on ~1B video tokens
"""
from __future__ import annotations
import argparse, math, os, sys, time

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
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


VJEPA_REPO  = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT  = "/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"


class TimeConditioner(nn.Module):
    """Inject flow-time t into latent tokens via FiLM-style scale + shift."""
    def __init__(self, dim: int = 1024, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, 2 * dim),
        )
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _sin(t, dim):
        half = dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        emb = torch.cat([torch.cos(t.float()[:, None] * f[None, :]),
                          torch.sin(t.float()[:, None] * f[None, :])], dim=-1)
        if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, z, t):
        sh, sc = self.mlp(self._sin(t, self.freq_dim)).chunk(2, dim=-1)
        return z * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)


class REPAProjector(nn.Module):
    def __init__(self, in_ch=256, out_dim=1024):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=3, padding=1)
    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        m = x.mean(dim=1, keepdim=True); s = x.std(dim=1, keepdim=True) + 1e-6
        return (x - m) / s


def safe_cos_loss(z_stu, z_tea):
    z_stu = F.normalize(z_stu.float(), dim=-1, eps=1e-3)
    z_tea = F.normalize(z_tea.float(), dim=-1, eps=1e-3)
    cos = (z_stu * z_tea).sum(dim=-1).clamp(-1.5, 1.5)
    return -cos.mean()


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
    # Set gate to 1.0 — we want full refiner output as velocity (not residual)
    with torch.no_grad():
        refiner.gate.fill_(1.0)
    n_ref = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    n_ref_total = sum(p.numel() for p in refiner.parameters())

    time_cond = TimeConditioner(dim=1024, freq_dim=256).to(device)
    n_tc = sum(p.numel() for p in time_cond.parameters())

    render = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_r = sum(p.numel() for p in render.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=render.dims[-1], out_dim=render.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[VJPF] V-JEPA encoder 304M frozen")
    print(f"       Refiner: {n_ref/1e6:.2f}M trainable (LoRA+gate) / {n_ref_total/1e6:.1f}M total (predictor body frozen)")
    print(f"       TimeConditioner: {n_tc/1e6:.3f}M")
    print(f"       Render: {n_r/1e6:.2f}M, REPAProj: {n_p/1e6:.2f}M")
    print(f"       Total trainable: {(n_ref+n_tc+n_r+n_p)/1e6:.2f}M")
    print(f"       Loss: L_flow + {args.lambda_pixel}·L_pixel + {args.lambda_repa}·L_repa "
          f"(HASTE off @ iter {args.early_stop_iter})")

    train_params = ([p for p in refiner.parameters() if p.requires_grad]
                    + list(time_cond.parameters())
                    + list(render.parameters())
                    + list(projector.parameters()))
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr*0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        refiner.load_state_dict(ckpt["refiner"])
        time_cond.load_state_dict(ckpt["time_cond"])
        render.load_state_dict(ckpt["render"])
        projector.load_state_dict(ckpt["projector"])
        print(f"[eval-only] loaded ckpt → {args.out_dir}/ckpt.pt")
    else:
        t0 = time.time()
        for it in range(args.iters):
            idx = int(rng.integers(0, len(train_seqs)))
            _, lq_p, gt_p = train_seqs[idx]
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)

            with torch.no_grad():
                z_LQ = encode(backbone, lq)[args.ref_layer].float()
                z_HQ = encode(backbone, gt)[args.ref_layer].float()

            B = z_LQ.shape[0]
            t_sample = torch.rand(B, device=device)
            t_view = t_sample.view(B, 1, 1)
            z_t = (1.0 - t_view) * z_LQ + t_view * z_HQ
            v_target = z_HQ - z_LQ

            # Time-condition then send through V-JEPA predictor as velocity field
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_t_cond = time_cond(z_t, t_sample)
                v_pred = refiner(z_t_cond.bfloat16()).float() - z_t  # refiner returns refined; subtract input → delta as velocity

            L_flow = F.mse_loss(v_pred, v_target)

            kv_layers = [z_HQ]
            h_b, skips = render.encode(lq, kv_layers=kv_layers, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=kv_layers, delta_layers=None)
            L_pixel = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_stu = projector(h_b, tuple(render.vjepa_grid))
                m = z_HQ.mean(dim=1, keepdim=True); s = z_HQ.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_HQ - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_flow + args.lambda_pixel * L_pixel + args.lambda_repa * L_repa

            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_flow={L_flow.item():.4f} "
                      f"L_pixel={L_pixel.item():.4f} L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(), "L_flow": L_flow.item(),
                            "L_pixel": L_pixel.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"refiner": refiner.state_dict(), "time_cond": time_cond.state_dict(),
                    "render": render.state_dict(), "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "vj_predictor_flow"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Final eval
    print(f"\n=== FINAL EVAL: VJ-Predictor-Flow ({args.eval_steps}-step ODE) ===")
    psnr_p, psnr_o, mean_iqa = _quick_eval(refiner, time_cond, render, backbone, test_seqs,
                                            device, args.crop, args.ref_layer,
                                            args.eval_steps, verbose=True)
    md = os.path.join(args.out_dir, "VJPF_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# VJ-Predictor-Flow: V-JEPA predictor as velocity field\n\n")
        f.write(f"Refiner {n_ref/1e6:.2f}M trainable LoRA (predictor body {n_ref_total/1e6:.0f}M FROZEN)\n")
        f.write(f"TimeConditioner {n_tc/1e6:.3f}M, Render {n_r/1e6:.2f}M, Projector {n_p/1e6:.2f}M\n")
        f.write(f"V-JEPA encoder 304M frozen\n\n")
        f.write(f"## SDSD held-out (predictor-pipeline)\n\n")
        f.write(f"- PSNR    = {psnr_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"## Oracle-render PSNR = {psnr_o:.3f}\n\n")
        f.write(f"vs E2b 19.556, vs PATH P 19.724, vs pure_rgb 18.533\n")
    print(f"\nVJ-Predictor-Flow PSNR={psnr_p:.3f} {fmt_iqa(mean_iqa)}  oracle={psnr_o:.3f}")
    print(f"wrote {md}")


@torch.no_grad()
def _quick_eval(refiner, time_cond, render, backbone, test_seqs, device, crop, ref_layer,
                num_steps, verbose=False, with_iqa=True):
    refiner.eval(); time_cond.eval(); render.eval()
    psnrs_p, psnrs_o = [], []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []} if with_iqa else None
    for name, lq_p, gt_p in test_seqs:
        lq, gt = load_clip(lq_p, gt_p, crop_size=crop)
        lq, gt = lq.to(device), gt.to(device)
        z_LQ = encode(backbone, lq)[ref_layer].float()
        z_HQ = encode(backbone, gt)[ref_layer].float()

        # Predictor-flow ODE: at each step, time-condition + refiner-as-velocity
        z = z_LQ.clone()
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_k = torch.full((z.shape[0],), k * dt + 0.5 * dt,
                             device=z.device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_cond = time_cond(z, t_k)
                v = refiner(z_cond.bfloat16()).float() - z
            z = z + dt * v
        z_HQ_pred = z

        h_b, skips = render.encode(lq, kv_layers=[z_HQ_pred], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ_pred], delta_layers=None)
        pred_p_f = pred.float()
        psnrs_p.append(mean_psnr(to_u8(pred_p_f), to_u8(gt)))
        if with_iqa:
            iqa = compute_iqa(pred_p_f, gt, device=device)
            for k_ in iqa_acc: iqa_acc[k_].append(iqa[k_])

        h_b, skips = render.encode(lq, kv_layers=[z_HQ], delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ], delta_layers=None)
        psnrs_o.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))

        if verbose:
            extra = f" {fmt_iqa(iqa)}" if with_iqa else ""
            print(f"    [{name}] Pred={psnrs_p[-1]:.3f}{extra}  oracle={psnrs_o[-1]:.3f}")

    refiner.train(); time_cond.train(); render.train()
    if backbone is not None: backbone.eval()
    mean_iqa = ({k: float(np.mean(v)) for k, v in iqa_acc.items()} if with_iqa else None)
    return float(np.mean(psnrs_p)), float(np.mean(psnrs_o)), mean_iqa


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
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--early_stop_iter", type=int, default=900)
    ap.add_argument("--eval_steps", type=int, default=4)
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training, load ckpt from out_dir, run eval+IQA only")
    main(ap.parse_args())
