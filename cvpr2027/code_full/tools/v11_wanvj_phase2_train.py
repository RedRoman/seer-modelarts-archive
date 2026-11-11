"""WanVJ-LLVE Phase 2 — comprehensive recipe upgrade for visual-quality win.

Continues from wanvj_25k Phase 1 ckpt. 11 modifications:
  1. L_l1 (RGB direct anchor)
  2. L_chroma (YCbCr CbCr anchor) — main fix for chroma drift
  3. L_lpips (perceptual, every 4th iter to amortize cost)
  4. L_repa (V-JEPA mid-DiT-block alignment)
  5. Data mix: BVI + SDSD train fold (1:1 sampling)
  6. Continue from wanvj_25k ckpt (LoRA + projector + LQ cond)
  7. lr 5e-5 (½ of phase1 1e-4)
  8. EMA 0.999 (NAFNet++ recipe)
  9. 1k-iter linear ramp on all NEW losses
 10. HASTE: drop L_repa at iter 7000/8000 (alignment converged)
 11. + chroma AdaIN post-processing at inference time

Memory: ~50-55GB peak (LoRA + Wan VAE decode-in-loop + LPIPS + REPA proj)
Iters: 8000  (~9 hr at ~4 sec/iter with decode + LPIPS)
"""
from __future__ import annotations
import argparse, os, copy, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Reuse Phase 1 trainer infrastructure
sys.path.insert(0, "/home/ma-user/workspace/zihaohe/vjepa_llve")
from tools.v11_wanvj_train import (
    _RMSNorm,  # already monkey-patched on import
    LoRALinear, inject_lora_on_dit,
    VJepaToText, LQConditioner,
    wan_encode, wan_decode, get_wan_latents_norm,
    rgb_to_ycbcr, ycbcr_to_rgb, chroma_adain,
    WAN_PATH, BVI_RLV_PATH,
)
from tools.iqa_metrics import compute_iqa, fmt_iqa, save_compare_video
from tools.v11_oracle_delta_probe import (load_clip, to_u8, mean_psnr, encode,
                                           build_split_seqs)
from data.bvi_rlv_dataset import BVIRLVDataset
from models.vjepa_backbone import build_vjepa_backbone

from diffusers import AutoencoderKLWan, WanTransformer3DModel


def chroma_loss_ycbcr(pred_rgb, gt_rgb):
    """L1 on Cb/Cr channels in [0,1]-scaled YCbCr."""
    p = rgb_to_ycbcr(pred_rgb)
    g = rgb_to_ycbcr(gt_rgb)
    return F.l1_loss(p[:, 1:], g[:, 1:])


def build_lpips(device):
    """Lazy-load LPIPS (VGG)."""
    import pyiqa
    m = pyiqa.create_metric("lpips", as_loss=True, device=device)
    for p in m.parameters(): p.requires_grad_(False)
    return m


def lpips_video(lpips_metric, pred, gt, n_frames=4):
    """Compute LPIPS on n_frames evenly-sampled frames; returns scalar."""
    T = pred.shape[2]
    idx = torch.linspace(0, T - 1, n_frames).long()
    p = pred[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 3, pred.shape[3], pred.shape[4])
    g = gt[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 3, gt.shape[3], gt.shape[4])
    return lpips_metric(p, g).mean()


@torch.no_grad()
def update_ema(ema_dict, src_params_dict, decay=0.999):
    for k, p in src_params_dict.items():
        ema_dict[k].mul_(decay).add_(p.data.to(ema_dict[k].dtype), alpha=1 - decay)


class MixedDataset:
    """Sample alternately from BVI-RLV and SDSD train fold."""
    def __init__(self, bvi, sdsd_train_seqs, clip_len=16, crop=256, rng=None):
        self.bvi = bvi
        self.sdsd = sdsd_train_seqs  # list of (name, lq_path, gt_path)
        self.clip_len = clip_len
        self.crop = crop
        self.rng = rng or np.random.default_rng(0)

    def sample(self, source: str):
        if source == "bvi":
            s = self.bvi[int(self.rng.integers(0, len(self.bvi)))]
            return s["lq"], s["hq"]
        else:
            name, lq_p, gt_p = self.sdsd[int(self.rng.integers(0, len(self.sdsd)))]
            lq, gt = load_clip(lq_p, gt_p, crop_size=self.crop, clip_len=self.clip_len)
            return lq.squeeze(0), gt.squeeze(0)


def main(args):
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Phase2] loading frozen backbones ...")
    vae = AutoencoderKLWan.from_pretrained(WAN_PATH, subfolder="vae",
                                           torch_dtype=torch.float32).to(device).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    dit = WanTransformer3DModel.from_pretrained(WAN_PATH, subfolder="transformer",
                                                torch_dtype=torch.bfloat16).to(device).eval()
    for p in dit.parameters(): p.requires_grad_(False)

    with open(args.cfg) as f: cfg = yaml.safe_load(f).get("model", {})
    backbone = build_vjepa_backbone(cfg["backbone"]).to(device).eval()
    for p in backbone.parameters(): p.requires_grad_(False)

    n_lora = inject_lora_on_dit(dit, rank=args.lora_rank)
    dit.to(device)
    print(f"[Phase2] injected LoRA on {n_lora} attn linears")

    vj2text = VJepaToText(vj_dim=1024, text_dim=4096).to(device)
    lq_cond = LQConditioner(ch=16).to(device)

    # ---- Continue from Phase 1 ckpt ----
    if args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=device)
        dit.load_state_dict({k: v.to(device) for k, v in sd["dit_lora"].items()},
                            strict=False)
        vj2text.load_state_dict(sd["vj2text"])
        lq_cond.load_state_dict(sd["lq_cond"])
        print(f"[Phase2] continue from {args.init_ckpt}")

    train_params = ([p for p in dit.parameters() if p.requires_grad]
                    + list(vj2text.parameters())
                    + list(lq_cond.parameters()))
    n_train = sum(p.numel() for p in train_params)
    print(f"[Phase2] trainable: {n_train/1e6:.2f}M; lr={args.lr}; iters={args.iters}")

    # ---- REPA projector (DiT bottleneck → V-JEPA dim) ----
    # DiT internal hidden = 1536 (head_dim=128 × 12 heads); V-JEPA = 1024
    repa_proj = nn.Linear(1536, 1024).to(device)
    train_params += list(repa_proj.parameters())

    # Hook to grab DiT bottleneck (after block 14, mid of 30 blocks)
    _bottleneck = {}
    def _hook(m, i, o):
        # o is tuple or tensor, hook on transformer block output
        h = o[0] if isinstance(o, tuple) else o
        _bottleneck["h"] = h
    dit.blocks[14].register_forward_hook(_hook)

    # ---- Loss modules ----
    print("[Phase2] loading LPIPS (VGG) ...")
    lpips_m = build_lpips(device)

    # ---- Optim + scheduler ----
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.iters,
                                                       eta_min=args.lr * 0.05)

    # ---- EMA on trainable params (state_dict copy) ----
    ema_state = {}
    for p_grp in [dit, vj2text, lq_cond]:
        for k, p in p_grp.state_dict().items():
            if p.dtype.is_floating_point:
                ema_state[id(p_grp), k] = p.detach().clone().float()

    # ---- Data ----
    bvi = BVIRLVDataset(BVI_RLV_PATH, split="train", clip_len=16,
                        crop_size=args.crop, is_train=True)
    sdsd_train_seqs, sdsd_test_seqs = build_split_seqs()
    print(f"[data] BVI: {len(bvi)}; SDSD train: {len(sdsd_train_seqs)}; "
          f"SDSD held-out: {len(sdsd_test_seqs)}")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    mean, std = get_wan_latents_norm(vae, device)

    # ---- Training loop ----
    log = []
    for it in range(args.iters):
        # 1:1 BVI:SDSD alternation
        source = "bvi" if (it % 2 == 0) else "sdsd"
        if source == "bvi":
            s = bvi[int(rng.integers(0, len(bvi)))]
            lq = s["lq"].unsqueeze(0).to(device)
            gt = s["hq"].unsqueeze(0).to(device)
        else:
            name, lq_p, gt_p = sdsd_train_seqs[int(rng.integers(0, len(sdsd_train_seqs)))]
            try:
                lq_t, gt_t = load_clip(lq_p, gt_p, crop_size=args.crop)
                lq = lq_t.to(device); gt = gt_t.to(device)
            except Exception as e:
                # Skip SDSD seq if load fails (some seqs too short)
                source = "bvi"
                s = bvi[int(rng.integers(0, len(bvi)))]
                lq = s["lq"].unsqueeze(0).to(device)
                gt = s["hq"].unsqueeze(0).to(device)

        with torch.no_grad():
            z_lq = wan_encode(vae, lq, mean, std)
            z_hq = wan_encode(vae, gt, mean, std)
            z_vj = encode(backbone, lq)[args.ref_layer].float()
            # V-JEPA on GT for REPA target
            z_vj_gt = encode(backbone, gt)[args.ref_layer].float()  # (B, 2048, 1024)

        B = z_lq.shape[0]
        t = torch.rand(B, device=device)
        tv = t.view(B, 1, 1, 1, 1)
        z_t = (1.0 - tv) * z_lq + tv * z_hq
        v_target = z_hq - z_lq
        z_t_cond = z_t + lq_cond(z_lq)
        text_emb = vj2text(z_vj)
        timestep = (t * 1000).long()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = dit(z_t_cond.to(torch.bfloat16),
                         timestep,
                         encoder_hidden_states=text_emb.to(torch.bfloat16),
                         return_dict=False)[0].float()

        # --- L_flow (latent flow) ---
        L_flow = F.mse_loss(v_pred, v_target)

        # --- Losses requiring decode (RGB pred at t=0 prediction) ---
        # Predict z_HQ via 1-NFE: z_HQ_pred = z_lq + v_pred_at_t=0
        # During training we use the v_pred at sampled t — but for pixel/lpips losses
        # we want the CONSISTENT 1-NFE prediction. Re-forward DiT at t=0 (cheap, 1 extra fwd).
        # OPTIMIZATION: use v_pred from current step as approximation (v_pred ≈ v_pred_t=0
        # under rectified flow, which targets straight-line trajectories)
        z_hq_pred = (z_lq + v_pred)
        # Decode WITH gradient (needed for pixel losses to reach DiT/LoRA)
        pred_rgb = wan_decode(vae, z_hq_pred, mean, std).clamp(0, 1)
        T_out = pred_rgb.shape[2]
        T_drop = (gt.shape[2] - T_out) // 2
        gt_trim = gt[:, :, T_drop:T_drop + T_out]

        L_l1     = F.l1_loss(pred_rgb, gt_trim)
        L_chroma = chroma_loss_ycbcr(pred_rgb, gt_trim)

        # LPIPS every Nth iter (memory amortization)
        L_lpips = torch.tensor(0.0, device=device)
        if (it + 1) % args.lpips_every == 0:
            L_lpips = lpips_video(lpips_m, pred_rgb, gt_trim, n_frames=4)

        # REPA on DiT mid-bottleneck
        L_repa = torch.tensor(0.0, device=device)
        haste_active = it < args.haste_iter
        if haste_active and "h" in _bottleneck:
            h_mid = _bottleneck["h"].float()  # (B, N, 1536)
            h_proj = repa_proj(h_mid)  # (B, N, 1024)
            # z_vj_gt is (B, 2048, 1024); align mean over tokens to be shape-agnostic
            # Use cosine on mean-pooled features for stability
            h_p = F.normalize(h_proj.mean(dim=1), dim=-1)
            v_p = F.normalize(z_vj_gt.mean(dim=1), dim=-1)
            L_repa = -(h_p * v_p).sum(dim=-1).mean()

        # Ramp NEW losses from 0 -> full over args.ramp_iters
        ramp = min(1.0, (it + 1) / max(1, args.ramp_iters))
        L = (L_flow
             + ramp * args.lambda_l1     * L_l1
             + ramp * args.lambda_chroma * L_chroma
             + ramp * args.lambda_lpips  * L_lpips
             + ramp * args.lambda_repa   * L_repa)

        optim.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optim.step(); sched.step()

        # EMA
        if (it + 1) % 10 == 0:
            with torch.no_grad():
                for p_grp in [dit, vj2text, lq_cond]:
                    for k, p in p_grp.state_dict().items():
                        key = (id(p_grp), k)
                        if key in ema_state:
                            ema_state[key].mul_(args.ema).add_(
                                p.detach().to(ema_state[key].dtype), alpha=1 - args.ema)

        if (it + 1) % args.log_every == 0:
            print(f"  it {it+1}/{args.iters} src={source} ramp={ramp:.2f} "
                  f"L={L.item():.4f} flow={L_flow.item():.4f} "
                  f"l1={L_l1.item():.4f} ch={L_chroma.item():.4f} "
                  f"lp={L_lpips.item():.4f} rp={L_repa.item():.4f} "
                  f"haste={int(haste_active)}")
            log.append({"iter": it+1, "L": L.item(), "L_flow": L_flow.item(),
                        "L_l1": L_l1.item(), "L_chroma": L_chroma.item(),
                        "L_lpips": L_lpips.item(), "L_repa": L_repa.item()})

    # ---- Save ckpt (raw + EMA) ----
    torch.save({"dit_lora": {k: v.cpu() for k, v in dit.state_dict().items()
                              if "A" in k or "B" in k},
                "vj2text": vj2text.state_dict(),
                "lq_cond": lq_cond.state_dict(),
                "repa_proj": repa_proj.state_dict(),
                "ema_state": {f"{a}|{b}": v.cpu() for (a, b), v in ema_state.items()},
                "args": vars(args), "log": log, "stage": "wanvj_phase2"},
               os.path.join(args.out_dir, "ckpt.pt"))
    print(f"[ckpt] saved → {args.out_dir}/ckpt.pt")

    # ---- Eval (1-NFE + chroma AdaIN post-processing) ----
    print("\n=== EVAL: WanVJ Phase 2 (1-NFE + AdaIN-lq) ===")
    dit.eval(); vj2text.eval(); lq_cond.eval()
    psnrs = []; iqa_acc = {"ssim": [], "lpips": [], "dists": [], "clipiqa": []}
    viz_root = args.viz_dir or os.path.join(args.out_dir, "viz")
    os.makedirs(viz_root, exist_ok=True)
    with torch.no_grad():
        for name, lq_p, gt_p in sdsd_test_seqs:
            lq, gt = load_clip(lq_p, gt_p, crop_size=args.crop)
            lq, gt = lq.to(device), gt.to(device)
            z_lq = wan_encode(vae, lq, mean, std)
            z_vj = encode(backbone, lq)[args.ref_layer].float()
            text_emb = vj2text(z_vj)
            B = z_lq.shape[0]
            t_zero = torch.zeros(B, device=device).long()
            z_t_cond = z_lq + lq_cond(z_lq)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_pred = dit(z_t_cond.to(torch.bfloat16), t_zero,
                             encoder_hidden_states=text_emb.to(torch.bfloat16),
                             return_dict=False)[0].float()
            pred = wan_decode(vae, z_lq + v_pred, mean, std).float()
            T_out = pred.shape[2]
            T_drop = (gt.shape[2] - T_out) // 2
            gt_trim = gt[:, :, T_drop:T_drop + T_out]
            lq_trim = lq[:, :, T_drop:T_drop + T_out]
            # +AdaIN post-process (lq mode for deployable, no GT at infer)
            pred_adain = chroma_adain(pred, lq_trim).clamp(0, 1)
            p = mean_psnr(to_u8(pred_adain), to_u8(gt_trim))
            iqa = compute_iqa(pred_adain, gt_trim, device=device)
            psnrs.append(p)
            for k in iqa_acc: iqa_acc[k].append(iqa[k])
            print(f"    [{name}] PSNR={p:.3f} {fmt_iqa(iqa)}")
            save_compare_video(os.path.join(viz_root, f"{name}.mp4"),
                               lq_trim, pred_adain, gt_trim, fps=10)

    mean_p = float(np.mean(psnrs))
    mean_iqa = {k: float(np.mean(v)) for k, v in iqa_acc.items()}
    md = os.path.join(args.out_dir, "WANVJ_PHASE2_EVAL.md")
    with open(md, "w") as f:
        f.write(f"# WanVJ Phase 2 (BVI+SDSD mix, full loss recipe, EMA, +AdaIN)\n\n")
        f.write(f"Init from Phase 1 ckpt: {args.init_ckpt}\n")
        f.write(f"Iters: {args.iters}; LR: {args.lr}; HASTE: {args.haste_iter}; ramp: {args.ramp_iters}\n")
        f.write(f"Loss weights: l1={args.lambda_l1} chroma={args.lambda_chroma} "
                f"lpips={args.lambda_lpips} repa={args.lambda_repa}\n\n")
        f.write(f"## SDSD held-out (1-NFE + AdaIN-lq)\n\n")
        f.write(f"- PSNR    = {mean_p:.3f}\n")
        f.write(f"- SSIM    = {mean_iqa['ssim']:.4f}\n")
        f.write(f"- LPIPS   = {mean_iqa['lpips']:.4f}\n")
        f.write(f"- DISTS   = {mean_iqa['dists']:.4f}\n")
        f.write(f"- CLIPIQA = {mean_iqa['clipiqa']:.4f}\n")
    print(f"\n[Phase2] PSNR={mean_p:.3f} {fmt_iqa(mean_iqa)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="/cache/zihaohe/checkpoint/runs/v9_0b/config.yaml")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_ckpt", default="/cache/zihaohe/checkpoint/v11_probes/wanvj_25k/ckpt.pt")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--ramp_iters", type=int, default=1000)
    ap.add_argument("--haste_iter", type=int, default=7000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--lpips_every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ref_layer", type=int, default=3)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lambda_l1", type=float, default=0.5)
    ap.add_argument("--lambda_chroma", type=float, default=0.5)
    ap.add_argument("--lambda_lpips", type=float, default=0.1)
    ap.add_argument("--lambda_repa", type=float, default=0.3)
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--viz_dir", default=None)
    main(ap.parse_args())
