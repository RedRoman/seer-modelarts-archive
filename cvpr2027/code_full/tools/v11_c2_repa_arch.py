"""REPA architectural variants N1-N8.

N1 attn_map     — align self-similarity attention maps (softmax(QK^T/sqrt(d)))
N2 predictive   — decoder mid → MLP head → predict V-JEPA(GT) features (MSE)
N3 cka          — Centered Kernel Alignment (dim/orthogonal-invariant)
N4 rkd_distance — pairwise distance preservation (Park CVPR'19)
N5 bidirectional — two projectors d↔v, cycle consistency
N6 local_window — local 3D window cosine alignment (preserves locality)
N7 self_distill — control: align decoder mid to its OWN late layer (no V-JEPA)
N8 hard_token   — REPA cosine reweighted by per-token error magnitude

All use V2-style decoder, no V-JEPA at inference.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time

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


class ConvProj(nn.Module):
    def __init__(self, in_ch=256, out_dim=1024, k=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_dim, kernel_size=k, padding=k//2)

    def forward(self, h_b, vjepa_grid):
        x = self.conv(h_b)
        x = F.interpolate(x, size=vjepa_grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), -1, x.size(1)).contiguous()
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim, hid=2048, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, out_dim),
        )
    def forward(self, x): return self.net(x)


def _norm_layer(z):
    m = z.mean(dim=1, keepdim=True); s = z.std(dim=1, keepdim=True) + 1e-6
    return (z - m) / s


# --- loss functions ----------------------------------------------------------

def loss_n1_attn_map(z_stu, z_tea, heads=4):
    """N1: align self-similarity attention maps."""
    B, N, D = z_stu.shape
    Dh = D // heads
    qs = z_stu.view(B, N, heads, Dh).transpose(1, 2)   # (B, h, N, Dh)
    ks = qs
    qt = z_tea.view(B, N, heads, Dh).transpose(1, 2)
    kt = qt
    attn_s = (qs @ ks.transpose(-2, -1) / math.sqrt(Dh)).softmax(-1)
    attn_t = (qt @ kt.transpose(-2, -1) / math.sqrt(Dh)).softmax(-1).detach()
    # KL divergence: sum(t * log(t/s))
    eps = 1e-8
    return (attn_t * (torch.log(attn_t + eps) - torch.log(attn_s + eps))).sum(-1).mean()


def loss_n2_predictive(z_stu, z_tea):
    """N2: MSE prediction (z_stu IS the predictor head output)."""
    return F.mse_loss(z_stu, z_tea)


def loss_n3_cka(z_stu, z_tea):
    """N3: 1 - CKA. flatten (B,N,D) → (B*N, D)."""
    X = z_stu.reshape(-1, z_stu.size(-1))
    Y = z_tea.reshape(-1, z_tea.size(-1))
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    sX = X @ X.T   # (BN, BN)
    sY = Y @ Y.T
    hsic = (sX * sY).sum()
    norm = torch.sqrt((sX * sX).sum() * (sY * sY).sum() + 1e-8)
    return 1 - hsic / (norm + 1e-8)


def loss_n4_rkd_distance(z_stu, z_tea, max_pairs=2048):
    """N4: pairwise distance preservation. Sample max_pairs to avoid OOM."""
    B, N, D = z_stu.shape
    Xs = z_stu.reshape(B*N, D)
    Xt = z_tea.reshape(B*N, D)
    if Xs.size(0) > max_pairs:
        idx = torch.randperm(Xs.size(0), device=Xs.device)[:max_pairs]
        Xs, Xt = Xs[idx], Xt[idx]
    pds = torch.cdist(Xs, Xs, p=2)
    pdt = torch.cdist(Xt, Xt, p=2)
    pds = pds / (pds[pds > 0].mean() + 1e-6)
    pdt = pdt / (pdt[pdt > 0].mean() + 1e-6)
    return F.smooth_l1_loss(pds, pdt)


def loss_n5_bidirectional(z_stu, z_tea, projector_v2d, h_b_proj):
    """N5: bidirectional. d→v already computed (z_stu); also v→d.
    h_b_proj is decoder mid feature flattened (the target for v→d)."""
    # forward direction: z_stu align to z_tea (cosine)
    fwd = -(F.normalize(z_stu.float(), dim=-1, eps=1e-3) * F.normalize(z_tea.float(), dim=-1, eps=1e-3).detach()).sum(-1).mean()
    # backward direction: project z_tea back to decoder space
    z_back = projector_v2d(z_tea)        # (B, N, D_dec)
    bwd = -(F.normalize(z_back.float(), dim=-1, eps=1e-3) * F.normalize(h_b_proj.float(), dim=-1, eps=1e-3).detach()).sum(-1).mean()
    return 0.5 * fwd + 0.5 * bwd


def loss_n6_local_window(z_stu, z_tea, vjepa_grid, win=3):
    """N6: align in local 3D windows (T,H,W). Reshape (B,N,D)→(B,T,H,W,D)→unfold."""
    B, N, D = z_stu.shape
    T, H, W = vjepa_grid
    s = z_stu.view(B, T, H, W, D).permute(0, 4, 1, 2, 3)    # (B, D, T, H, W)
    t = z_tea.view(B, T, H, W, D).permute(0, 4, 1, 2, 3)
    # mean-pool in 3x3x3 windows
    s_avg = F.avg_pool3d(s, kernel_size=win, stride=1, padding=win//2)
    t_avg = F.avg_pool3d(t, kernel_size=win, stride=1, padding=win//2)
    s_flat = s_avg.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
    t_flat = t_avg.permute(0, 2, 3, 4, 1).reshape(B, -1, D)
    return -(F.normalize(s_flat.float(), dim=-1, eps=1e-3) * F.normalize(t_flat.float(), dim=-1, eps=1e-3).detach()).sum(-1).mean()


def loss_n8_hard_token(z_stu, z_tea, gamma=2.0):
    """N8: per-token cosine reweighted by error magnitude (focal-style)."""
    cos = (F.normalize(z_stu.float(), dim=-1, eps=1e-3) * F.normalize(z_tea.float(), dim=-1, eps=1e-3).detach()).sum(-1)   # (B, N)
    err = (1 - cos).clamp(0, 2)
    w = err ** gamma
    w = w / (w.mean() + 1e-6)
    return (w * (1 - cos)).mean()


# --- main ---

def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval() if args.variant != "N7_self_distill" else None
    if backbone is not None:
        for p in backbone.parameters(): p.requires_grad_(False)

    decoder = DeltaModulatedDecoder(delta_in_encoder=False).to(device)
    n_d = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    # variant-specific projection heads
    if args.variant == "N2_predictive":
        projector = MLPHead(in_dim=decoder.dims[-1], hid=2048, out_dim=decoder.vjepa_dim).to(device)
        # MLPHead applied per-token, but input is per-token feature; reshape needed
        # We project h_b: first flatten to per-token, then MLP, then reshape
    elif args.variant == "N5_bidirectional":
        projector = ConvProj(decoder.dims[-1], decoder.vjepa_dim).to(device)
        projector_v2d = MLPHead(in_dim=decoder.vjepa_dim, hid=512, out_dim=decoder.vjepa_dim).to(device)
        # h_b → flat → project v2d to same vjepa_dim space (we compare in vjepa_dim)
    elif args.variant == "N7_self_distill":
        # No V-JEPA. Need decoder feature at TWO layers. Use h_b (bottleneck) as student target,
        # use a "self-target" = detached h_b at later iter or simply detach + project
        # Simpler: we project h_b to a target space, then align another copy of h_b to same target
        # Even simpler: contract — random projection target
        # Cleanest: align h_b to detached h_b through deeper Mlp head — i.e. consistency loss.
        # We'll use a Conv-then-Conv self-pred: conv1(h_b) align to conv2(h_b).detach()
        projector = ConvProj(decoder.dims[-1], decoder.dims[-1]).to(device)   # student
        projector_v2d = ConvProj(decoder.dims[-1], decoder.dims[-1]).to(device)  # target
    else:
        projector = ConvProj(decoder.dims[-1], decoder.vjepa_dim).to(device)

    proj_params = list(projector.parameters())
    if args.variant == "N5_bidirectional":
        proj_params += list(projector_v2d.parameters())
    elif args.variant == "N7_self_distill":
        proj_params += list(projector_v2d.parameters())
    n_p = sum(p.numel() for p in proj_params)

    print(f"[{args.variant}] decoder {n_d/1e6:.2f}M + projector(s) {n_p/1e6:.2f}M; "
          f"lambda={args.lambda_repa}, ref_layer={args.ref_layer}, early_stop={args.early_stop_iter}")

    optim = torch.optim.AdamW(list(decoder.parameters()) + proj_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters, eta_min=args.lr*0.05)

    train_seqs, _ = build_split_seqs()
    rng = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()
    for it in range(args.iters):
        idx = int(rng.integers(0, len(train_seqs)))
        name, lq_p, gt_p = train_seqs[idx]
        lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
        lq, gt = lq.to(device), gt.to(device)
        h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
        L_rgb = charbonnier(pred.float(), gt)

        L_repa = torch.zeros((), device=device)
        if args.lambda_repa > 0 and (args.early_stop_iter < 0 or it < args.early_stop_iter):
            vg = tuple(decoder.vjepa_grid)
            if args.variant == "N7_self_distill":
                # Project h_b two ways; align student to detached target
                h_flat_s = projector(h_b, vg)         # student
                h_flat_t = projector_v2d(h_b, vg).detach()  # target (separate projector, same input)
                L_repa = -(F.normalize(h_flat_s.float(), dim=-1, eps=1e-3) * F.normalize(h_flat_t.float(), dim=-1, eps=1e-3).detach()).sum(-1).mean()
            else:
                with torch.no_grad():
                    z_L = encode(backbone, lq)
                    z_tea_raw = z_L[args.ref_layer].float()
                z_tea = _norm_layer(z_tea_raw)
                if args.variant == "N1_attn_map":
                    z_stu = projector(h_b, vg)
                    L_repa = loss_n1_attn_map(z_stu, z_tea)
                elif args.variant == "N2_predictive":
                    # Use GT V-JEPA as target
                    with torch.no_grad():
                        z_H = encode(backbone, gt)
                        z_tea_gt = _norm_layer(z_H[args.ref_layer].float())
                    # MLP per-token: flatten h_b spatially
                    Bh, Ch, Th, Hh, Wh = h_b.shape
                    h_flat = h_b.permute(0, 2, 3, 4, 1).reshape(Bh, -1, Ch)   # (B, N_h, C)
                    pred_v = projector(h_flat)                                 # (B, N_h, vjepa_dim)
                    # interpolate to V-JEPA grid count
                    N_v = z_tea_gt.size(1)
                    if pred_v.size(1) != N_v:
                        # naive resample: take first N_v or repeat
                        pred_v = pred_v.transpose(1, 2)        # (B, D, N_h)
                        pred_v = F.interpolate(pred_v, size=N_v, mode="linear", align_corners=False)
                        pred_v = pred_v.transpose(1, 2)        # (B, N_v, D)
                    L_repa = loss_n2_predictive(pred_v, z_tea_gt)
                elif args.variant == "N3_cka":
                    z_stu = projector(h_b, vg)
                    L_repa = loss_n3_cka(z_stu, z_tea)
                elif args.variant == "N4_rkd_distance":
                    z_stu = projector(h_b, vg)
                    L_repa = loss_n4_rkd_distance(z_stu, z_tea)
                elif args.variant == "N5_bidirectional":
                    z_stu = projector(h_b, vg)
                    # v→d back projection — must be in same dim. Use vjepa_dim throughout.
                    # h_b_target is z_stu detached
                    L_repa = loss_n5_bidirectional(z_stu, z_tea, projector_v2d, z_stu.detach())
                elif args.variant == "N6_local_window":
                    z_stu = projector(h_b, vg)
                    L_repa = loss_n6_local_window(z_stu, z_tea, vg)
                elif args.variant == "N8_hard_token":
                    z_stu = projector(h_b, vg)
                    L_repa = loss_n8_hard_token(z_stu, z_tea)
                else:
                    raise ValueError(args.variant)

        L = L_rgb + args.lambda_repa * L_repa
        optim.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + proj_params, 1.0)
        optim.step(); sched.step()

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} L={L.item():.4f} L_rgb={L_rgb.item():.4f} "
                  f"L_repa={L_repa.item():.4f} dt={time.time()-t0:.1f}s")
            log.append({"iter": it+1, "L": L.item(), "L_rgb": L_rgb.item(), "L_repa": L_repa.item()})
            t0 = time.time()

    save_dict = {"decoder": decoder.state_dict(), "args": vars(args), "log": log,
                 "stage": f"repa_{args.variant}"}
    torch.save(save_dict, os.path.join(args.out_dir, "ckpt.pt"))

    # Eval no_prior
    decoder.eval()
    test_seqs = build_split_seqs()[1]
    psnrs = []
    with torch.no_grad():
        for name, lq_p, gt_p in test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            h_b, skips = decoder.encode(lq, kv_layers=None, delta_layers=None)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = decoder.decode(h_b, skips, lq, kv_layers=None, delta_layers=None)
            psnrs.append(mean_psnr(to_u8(pred.float()), to_u8(gt)))
    val = float(np.mean(psnrs))
    md = os.path.join(args.out_dir, "REPA_ARCH_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# REPA Arch Variant {args.variant}\n\n")
        f.write(f"variant: {args.variant}\n")
        f.write(f"lambda_repa: {args.lambda_repa}, early_stop_iter: {args.early_stop_iter}\n")
        f.write(f"## no_prior PSNR = {val:.3f}\n\n")
        f.write(f"vs Stage 1 17.962 (∆ = {val-17.962:+.3f})\n")
        f.write(f"vs E2 REPA 19.433 (∆ = {val-19.433:+.3f})\n")
        f.write(f"vs E2b REPA+HASTE 19.556 (∆ = {val-19.556:+.3f})\n")
    print(f"\n[{args.variant}] PSNR = {val:.3f}  (E2b best = 19.556)")
    print(f"wrote {md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["N1_attn_map","N2_predictive","N3_cka","N4_rkd_distance",
                             "N5_bidirectional","N6_local_window","N7_self_distill","N8_hard_token"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda_repa", type=float, default=0.5)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--early_stop_iter", type=int, default=900, help="HASTE-style; 900/1500=60%")
    main(ap.parse_args())
