"""MeanFlow-LLVE: 1-step generative for V-JEPA latent (Geng, Deng, ..., He, May 2025).

Reference: arXiv 2505.13447 — "Mean Flows for One-step Generative Modeling".
Implements average-velocity field u_θ(z_t, t, r) trained with the JVP-based
MeanFlow identity, enabling 1-step inference (z_HQ = z_LQ + u(z_LQ, 1, 0)).

vs vjlf (rectified flow + 4-step Euler ODE):
  - Same DiT-style backbone, but conditions on BOTH (t, r) instead of only t
  - Loss enforces consistency of average velocity over interval [r, t]
  - Inference: 1 forward pass (vs 4 in vjlf)

Key loss form (from paper):
    v_inst = z_HQ - z_LQ                                 (linear-interp instantaneous velocity)
    z_t = (1-t)·z_LQ + t·z_HQ
    u, du = jvp(fn=lambda z,t: u_θ(z, t, r),
                primals=(z_t, t),
                tangents=(v_inst, 1))                    (one extra fwd pass)
    target = v_inst - (t - r) · du                       (stop-grad)
    L = MSE(u, target)

  When r = t the second term vanishes → matches standard FM.
  When r < t the network learns averaged velocity → enables 1-step gen.
"""
from __future__ import annotations
import argparse, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path: sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models.vjepa_backbone import build_vjepa_backbone
from models.v11.delta_modulated_decoder import DeltaModulatedDecoder
from tools.iqa_metrics import compute_iqa, fmt_iqa
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           charbonnier, build_split_seqs)


# ---------- helpers ---------------------------------------------------
def _sin_emb(t, dim, max_period=10000.0):
    half = dim // 2
    f = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * f[None, :]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _modulate(x, sh, sc):
    return x * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)


class TimePairEmb(nn.Module):
    """Embed both t and r → concatenated MLP."""
    def __init__(self, dim, freq=256):
        super().__init__()
        self.freq = freq
        self.mlp_t = nn.Sequential(nn.Linear(freq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.mlp_r = nn.Sequential(nn.Linear(freq, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t, r):
        return self.mlp_t(_sin_emb(t, self.freq)) + self.mlp_r(_sin_emb(r, self.freq))


class _SelfAttn(nn.Module):
    """Manual attention (math kernel) — required for torch.func.jvp support."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads; self.h_dim = dim // heads
        self.scale = self.h_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.h_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class MeanFlowBlock(nn.Module):
    """Factorized spatial-temporal SA + MLP. AdaLN-Zero by (t, r) embedding."""
    def __init__(self, dim, heads, T, H, W, ffn_mult=4.0, use_ckpt=True):
        super().__init__()
        self.T, self.H, self.W = T, H, W; self.dim = dim
        self.use_ckpt = use_ckpt
        self.adaln = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_s = _SelfAttn(dim, heads)
        self.norm_t = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_t = _SelfAttn(dim, heads)
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * ffn_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))

    def _spatial(self, x, sh, sc, ga):
        B = x.shape[0]; T, H, W, D = self.T, self.H, self.W, self.dim
        h = _modulate(self.norm_s(x), sh, sc)
        h = h.reshape(B, T, H * W, D).reshape(B * T, H * W, D)
        h = self.attn_s(h).reshape(B, T * H * W, D)
        return x + ga.unsqueeze(1) * h

    def _temporal(self, x, sh, sc, ga):
        B = x.shape[0]; T, H, W, D = self.T, self.H, self.W, self.dim
        h = _modulate(self.norm_t(x), sh, sc)
        h = h.reshape(B, T, H * W, D).transpose(1, 2).reshape(B * H * W, T, D)
        h = self.attn_t(h).reshape(B, H * W, T, D).transpose(1, 2).reshape(B, T * H * W, D)
        return x + ga.unsqueeze(1) * h

    def _mlp(self, x, sh, sc, ga):
        h = _modulate(self.norm_f(x), sh, sc)
        return x + ga.unsqueeze(1) * self.mlp(h)

    def forward(self, x, t_emb):
        m = self.adaln(t_emb).chunk(9, dim=1)
        sh_s,sc_s,ga_s, sh_t,sc_t,ga_t, sh_f,sc_f,ga_f = m
        # NOTE: don't grad-checkpoint — JVP needs full forward graph
        x = self._spatial(x, sh_s, sc_s, ga_s)
        x = self._temporal(x, sh_t, sc_t, ga_t)
        x = self._mlp(x, sh_f, sc_f, ga_f)
        return x


class MeanFlowDiT(nn.Module):
    """DiT-style velocity field u_θ(z, t, r) → predicts average velocity."""
    def __init__(self, vjepa_dim=1024, grid=(8, 16, 16), dim=512, depth=12, heads=8, ffn_mult=4.0):
        super().__init__()
        T, H, W = grid
        self.T, self.H, self.W, self.dim = T, H, W, dim
        self.in_proj = nn.Linear(vjepa_dim, dim)
        self.pos = nn.Parameter(torch.zeros(1, T * H * W, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.tr_emb = TimePairEmb(dim)
        self.blocks = nn.ModuleList([
            MeanFlowBlock(dim, heads, T, H, W, ffn_mult) for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out_adaln = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.out_adaln.weight); nn.init.zeros_(self.out_adaln.bias)
        self.out_proj = nn.Linear(dim, vjepa_dim)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, t, r):
        x = self.in_proj(z) + self.pos
        emb = self.tr_emb(t, r)
        for blk in self.blocks: x = blk(x, emb)
        sh, sc = self.out_adaln(emb).chunk(2, dim=1)
        x = _modulate(self.norm_out(x), sh, sc)
        return self.out_proj(x)


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


def mean_flow_loss(u_fn, z_LQ, z_HQ, p_inst=0.25):
    """MeanFlow loss with JVP. p_inst = prob of sampling r=t (instantaneous-only)."""
    B = z_LQ.shape[0]
    device = z_LQ.device
    t = torch.rand(B, device=device)
    # With probability p_inst, r=t (degenerates to FM); else r in [0, t)
    r = torch.where(torch.rand(B, device=device) < p_inst,
                    t, torch.rand(B, device=device) * t)
    tv = t.view(B, 1, 1)
    z_t = (1.0 - tv) * z_LQ + tv * z_HQ
    v_inst = z_HQ - z_LQ                                          # (B, N, D)

    # JVP: compute u and its directional derivative wrt (z_t, t) along (v_inst, 1)
    def fn(z_in, t_in):
        return u_fn(z_in, t_in, r)
    tangent_t = torch.ones_like(t)
    u_pred, du = torch.func.jvp(fn, (z_t, t), (v_inst, tangent_t))
    # du = ∂u/∂z @ v_inst + ∂u/∂t * 1

    # Target: v_inst - (t - r) * du   (stop grad)
    diff = (t - r).view(B, 1, 1)
    target = (v_inst - diff * du).detach()
    return F.mse_loss(u_pred, target)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    mf = MeanFlowDiT(vjepa_dim=1024, grid=(8, 16, 16),
                     dim=args.dit_dim, depth=args.dit_depth, heads=args.dit_heads,
                     ffn_mult=4.0).to(device)
    n_mf = sum(p.numel() for p in mf.parameters() if p.requires_grad)

    render = DeltaModulatedDecoder(
        delta_in_encoder=False,
        vjepa_layer_per_scale=(0, 0, 0, 0, 0),
        attn_at_scales=(3, 4),
    ).to(device)
    n_r = sum(p.numel() for p in render.parameters() if p.requires_grad)

    projector = REPAProjector(in_ch=render.dims[-1], out_dim=render.vjepa_dim).to(device)
    n_p = sum(p.numel() for p in projector.parameters())

    print(f"[MeanFlow-LLVE] V-JEPA encoder 304M frozen")
    print(f"                MeanFlowDiT: {n_mf/1e6:.2f}M trainable (dim={args.dit_dim} depth={args.dit_depth})")
    print(f"                Render: {n_r/1e6:.2f}M, Projector: {n_p/1e6:.2f}M")
    print(f"                Loss: MeanFlow JVP + {args.lambda_pixel}·Charbonnier + {args.lambda_repa}·REPA")
    print(f"                Inference: 1-step (z_HQ_pred = z_LQ + u_θ(z_LQ, t=1, r=0))")

    train_params = list(mf.parameters()) + list(render.parameters()) + list(projector.parameters())
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr * 0.05)

    train_seqs, test_seqs = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    log = []
    if args.eval_only:
        ckpt_d = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
        mf.load_state_dict(ckpt_d["mf"]); render.load_state_dict(ckpt_d["render"])
        if "projector" in ckpt_d: projector.load_state_dict(ckpt_d["projector"])
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

            # MeanFlow loss (no autocast inside JVP for stability)
            L_mf = mean_flow_loss(mf, z_LQ, z_HQ, p_inst=args.p_inst)

            # Render with oracle z_HQ
            kv = [z_HQ]
            h_b, skips = render.encode(lq, kv_layers=kv, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=kv, delta_layers=None)
            L_pixel = charbonnier(pred.float(), gt)

            L_repa = torch.zeros((), device=device)
            if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
                z_stu = projector(h_b, tuple(render.vjepa_grid))
                m = z_HQ.mean(dim=1, keepdim=True); s = z_HQ.std(dim=1, keepdim=True) + 1e-6
                z_tea = (z_HQ - m) / s
                L_repa = safe_cos_loss(z_stu, z_tea)

            L = L_mf + args.lambda_pixel * L_pixel + args.lambda_repa * L_repa
            optim.zero_grad(set_to_none=True); L.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optim.step(); sched.step()

            if (it + 1) % args.log_every == 0:
                print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_mf={L_mf.item():.4f} "
                      f"L_pixel={L_pixel.item():.4f} L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
                log.append({"iter": it+1, "L": L.item(), "L_mf": L_mf.item(),
                            "L_pixel": L_pixel.item(), "L_repa": L_repa.item()})
                t0 = time.time()

        torch.save({"mf": mf.state_dict(), "render": render.state_dict(),
                    "projector": projector.state_dict(),
                    "args": vars(args), "log": log, "stage": "meanflow"},
                   os.path.join(args.out_dir, "ckpt.pt"))
        print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # Eval — 1-step inference
    print(f"\n=== FINAL EVAL: MeanFlow-LLVE (1-step) ===")
    mf.eval(); render.eval()
    psnrs_p, psnrs_o = [], []
    iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_LQ = encode(backbone, lq)[args.ref_layer].float()
            z_HQ = encode(backbone, gt)[args.ref_layer].float()
            B = z_LQ.shape[0]
            t_one = torch.ones(B, device=device); r_zero = torch.zeros(B, device=device)
            u = mf(z_LQ, t_one, r_zero)
            z_HQ_pred = z_LQ + u  # 1-step: t=1, r=0 → integral = u * 1

            h_b, skips = render.encode(lq, kv_layers=[z_HQ_pred], delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = render.decode(h_b, skips, lq, kv_layers=[z_HQ_pred], delta_layers=None)
            pred_f = pred.float()
            p = mean_psnr(to_u8(pred_f), to_u8(gt))
            iqa = compute_iqa(pred_f, gt, device=device)
            psnrs_p.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])

            h_b, skips = render.encode(lq, kv_layers=[z_HQ], delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_o = render.decode(h_b, skips, lq, kv_layers=[z_HQ], delta_layers=None)
            psnrs_o.append(mean_psnr(to_u8(pred_o.float()), to_u8(gt)))
            print(f"    [{name}] Pred={p:.3f} {fmt_iqa(iqa)}  oracle={psnrs_o[-1]:.3f}")

    psnr_p, psnr_o = float(np.mean(psnrs_p)), float(np.mean(psnrs_o))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "MEANFLOW_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# MeanFlow-LLVE: 1-step Kaiming MeanFlow on V-JEPA latent\n\n")
        f.write(f"MeanFlowDiT {n_mf/1e6:.2f}M + Render {n_r/1e6:.2f}M + Proj {n_p/1e6:.2f}M trainable\n")
        f.write(f"V-JEPA encoder 304M frozen; inference = 1 forward pass\n")
        f.write(f"p_inst (prob of r=t in training) = {args.p_inst}\n\n")
        f.write(f"## SDSD held-out (1-step pipeline)\n\n")
        f.write(f"- PSNR    = {psnr_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}  (lower better)\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}  (lower better)\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}  (higher better)\n\n")
        f.write(f"## Oracle-render PSNR = {psnr_o:.3f}\n\n")
        f.write(f"vs vjlf 19.0, vjpf 19.15, mmdit 19.7, resflow 18.93, clip_30k 20.7, nafnet 20.4\n")
    print(f"\nMeanFlow-LLVE PSNR={psnr_p:.3f} {fmt_iqa(mean_iqa)}  oracle={psnr_o:.3f}")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lambda_pixel", type=float, default=1.0)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--early_stop_iter", type=int, default=9000)
    ap.add_argument("--dit_dim", type=int, default=512)
    ap.add_argument("--dit_depth", type=int, default=12)
    ap.add_argument("--dit_heads", type=int, default=8)
    ap.add_argument("--p_inst", type=float, default=0.25,
                    help="Prob of sampling r=t during training (degenerates to FM); 0.25 from paper.")
    ap.add_argument("--eval_only", action="store_true")
    main(ap.parse_args())
