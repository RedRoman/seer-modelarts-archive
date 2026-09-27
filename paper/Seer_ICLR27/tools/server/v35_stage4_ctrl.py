"""tools/v35_stage4_ctrl.py -- controlled V-JEPA vs no-V-JEPA Stage-4 fine-tune (ICLR rebuttal-style control).

Copy of v35_stage4_train.py with three additions: --no_vjepa (replace all V-JEPA
tokens by zeros, drop backbone/refiner/M2), --save_at (extra checkpoints), and
final-only checkpointing is what downstream eval uses (no test-set selection).

Original docstring follows.

tools/v35_stage4_train.py -- reconstructed v35 Seer Stage-4 fine-tune.

Rebuilt from the surviving train_20260620-115443.log and the model classes
(`pixel_branch_v10.py`, `predictor_lora.py`, `vjepa_backbone.py`). The
original tool was deleted; only checkpoints + log remained. This tool
reproduces Stage-4 exactly, then adds two CLI knobs so we can retrain the
two variants the reviewer critique requires:

    M1-active   : --gate_init 0.5 --router_c 4.0
    M3-active   : --gate_init 0.0 --router_c 1.0
    (default)   : --gate_init 0.0 --router_c 4.0   (matches archived run)

Pipeline per iter (batch=1 * accum=8 -> effective 8, crop=256, T=16):

    z_L_layers  = VJEPA(LQ)             frozen, 4 layer outputs (5/11/17/23)
    z_H_ref     = VJEPA(HQ)[3]          frozen, target for M2
    refined_ref = refiner(z_L_layers[3])  in bf16, back to fp32
    z_pixel     = [z_L_layers[0..2], refined_ref]   feeds pixel_branch
    out         = pixel_branch(LQ, z_pixel)          returns pred + R + L

    L = 1.5 * pix + 0.15 * retinex + 0.02 * smooth + 0.05 * jepa
      pix     = charbonnier(pred, HQ)
      retinex = charbonnier(R * L, HQ)
      smooth  = TV_spatial(L)
      jepa    = L1(refined_ref, z_H_ref.detach())

Audit every 1000 iters on SDSD-indoor test (15 seqs), 512x512 center crop,
compute per-frame RGB-PSNR and Y-PSNR, aggregate as frame-mean. Save
best_rgb512.pt / best_y512.pt (matching archived layout).

Note: the archived Stage-4 log did not print `gate=`; Stage-3 log did.
This tool restores the gate print (task spec asks for it) so the M1-active
retrain can be diagnosed at each log line.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data.common import read_image_01  # noqa: E402
from data.sdsd_dataset import SDSDDataset  # noqa: E402
from losses.v6 import charbonnier_loss  # noqa: E402
from losses.v7 import (  # noqa: E402
    illumination_smoothness_loss,
    retinex_compose_loss,
)
from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
from data.bvi_rlv_dataset import BVIRLVDataset  # noqa: E402
from seer_variants import (  # noqa: E402
    RANDOM_VJEPA_SEED, apply_pixel_branch_variant, param_fingerprint,
)


# ---------------------------------------------------------------------------
# Hard-coded paths (match memory/reference_remote_paths + v11 tools)
# ---------------------------------------------------------------------------
VJEPA_REPO = "/home/ma-user/workspace/zihaohe/modelzoo/vjepa2-main"
VJEPA_CKPT = "/home/ma-user/workspace/zihaohe/checkpoint/vjepa2_1_vitl_dist_vitG_384.pt"
SDSD_INDOOR = "/cache/zihaohe/datasets/SDSD_indoor"
SDSD_OUTDOOR = "/cache/zihaohe/datasets/SDSD_outdoor"
REF_LAYER = 3                       # deepest layer index (V-JEPA layer 23)
NUM_VJEPA_LAYERS = 4
N_TOKENS = 8 * 16 * 16              # (T/2) x (256/16) x (256/16) V-JEPA token grid
VJEPA_DIM = 1024


def zero_tokens(batch: int, device) -> List[torch.Tensor]:
    """no-V-JEPA control: every cross-attn K/V source is an all-zero token map."""
    z = torch.zeros(batch, N_TOKENS, VJEPA_DIM, device=device)
    return [z for _ in range(NUM_VJEPA_LAYERS)]

# Loss weights (from v10.yaml, preserved through v35)
W_PIX = 1.5
W_RETINEX = 0.15
W_SMOOTH = 0.02
W_JEPA = 0.05                       # lambda_M M2


# ---------------------------------------------------------------------------
# RGB / Y PSNR helpers (borrowed from audit_protocol_v9.py, kept in-file)
# ---------------------------------------------------------------------------
def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr_per_frame(pred: torch.Tensor, gt: torch.Tensor,
                   max_val: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return 20 * math.log10(max_val) - 10 * torch.log10(mse)


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    if H < size or W < size:
        return x
    h0 = (H - size) // 2
    w0 = (W - size) // 2
    return x[..., h0:h0 + size, w0:w0 + size]


# ---------------------------------------------------------------------------
# Audit (single fixed protocol matching Stage-4 log RGB-512 / Y-512 columns)
# ---------------------------------------------------------------------------
@torch.no_grad()
def audit_indoor_512(backbone, refiner, pixel_branch, seqs, device,
                     clip_len: int = 16) -> Tuple[float, float]:
    """RGB-PSNR and Y-PSNR at 512-center-crop, frame-mean over all sequences."""
    was_pb = pixel_branch.training
    was_rf = refiner.training if refiner is not None else False
    pixel_branch.eval()
    if refiner is not None:
        refiner.eval()
    rgb_all: List[float] = []
    y_all: List[float] = []
    for _name, lq_paths, gt_paths in seqs:
        n_total = len(lq_paths)
        T = min(clip_len, n_total)
        start = (n_total - T) // 2
        lq_np = np.stack([read_image_01(lq_paths[start + i]) for i in range(T)])
        gt_np = np.stack([read_image_01(gt_paths[start + i]) for i in range(T)])
        lq = torch.from_numpy(lq_np).permute(3, 0, 1, 2).unsqueeze(0).float()
        gt = torch.from_numpy(gt_np).permute(3, 0, 1, 2).unsqueeze(0).float()
        H, W = lq.shape[-2:]
        if min(H, W) < 512:
            lq = center_crop(lq, 256)
            gt = center_crop(gt, 256)
        else:
            lq = center_crop(lq, 512)
            gt = center_crop(gt, 512)
        lq, gt = lq.to(device), gt.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if backbone is None:
                z_pixel = zero_tokens(lq.shape[0], device)
            else:
                z_layers = backbone(lq)["feats"]
                refined = refiner(z_layers[REF_LAYER].bfloat16())
                z_pixel = list(z_layers[:REF_LAYER]) + [refined]
            out = pixel_branch(lq, z_pixel)
        pred = out["pred_pixels"].clamp(0, 1).float()
        p = pred[0].permute(1, 0, 2, 3)
        g = gt[0].permute(1, 0, 2, 3)
        rgb_all.extend(psnr_per_frame(p, g).cpu().tolist())
        y_all.extend(psnr_per_frame(rgb_to_y(p), rgb_to_y(g)).cpu().tolist())
    pixel_branch.train(was_pb)
    if refiner is not None:
        refiner.train(was_rf)
    return float(np.mean(rgb_all)), float(np.mean(y_all))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_train_loader(crop: int, clip_len: int, num_workers: int) -> DataLoader:
    ds_in = SDSDDataset(
        root=SDSD_INDOOR, subset="direct", split="train",
        clip_len=clip_len, crop_size=crop, is_train=True,
    )
    ds_out = SDSDDataset(
        root=SDSD_OUTDOOR, subset="direct", split="train",
        clip_len=clip_len, crop_size=crop, is_train=True,
    )
    ds = ConcatDataset([ds_in, ds_out])
    return DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )


BVI_ROOT = "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos"


def build_bvi_fewshot_loader(scenes, crop: int, clip_len: int, num_workers: int) -> DataLoader:
    """Few-shot target-domain adaptation data: the given BVI-RLV *training-split* scenes
    (both fps variants). The BVI-RLV test scenes used for evaluation are never touched."""
    ds = BVIRLVDataset(root=BVI_ROOT, split="train", clip_len=clip_len, crop_size=crop, is_train=True)
    keep = set(scenes)
    ds.seqs = [s for s in ds.seqs if s[0].split("__")[0] in keep]
    assert len({s[0].split("__")[0] for s in ds.seqs}) == len(keep), (keep, [s[0] for s in ds.seqs])
    test = BVIRLVDataset(root=BVI_ROOT, split="test", clip_len=clip_len, crop_size=crop, is_train=False)
    assert not keep & {s[0].split("__")[0] for s in test.seqs}, "few-shot scenes overlap the test split"
    print(f"[fewshot] {len(ds.seqs)} BVI train sequences from scenes {sorted(keep)}")
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True, persistent_workers=(num_workers > 0))


def build_audit_seqs():
    ds = SDSDDataset(
        root=SDSD_INDOOR, subset="direct", split="test",
        clip_len=16, crop_size=256, is_train=False,
    )
    return list(ds.seqs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", required=True,
                    help="Stage-3 ckpt with pixel_branch + cf_refiner blobs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="v35_S4")
    ap.add_argument("--gate_init", type=float, default=0.0,
                    help="Override CounterfactualRefiner.gate init")
    ap.add_argument("--router_c", type=float, default=4.0,
                    help="Override VJepaPixelBranch router_logits identity mass")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--accum_steps", type=int, default=8,
                    help="Effective batch = batch_size * accum_steps = 1 * 8")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--audit_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--no_vjepa", action="store_true",
                    help="Control arm: zero V-JEPA tokens, no backbone/refiner/M2")
    ap.add_argument("--save_at", type=int, nargs="*", default=[],
                    help="Extra iterations at which to save step_<it>.pt")
    ap.add_argument("--random_vjepa", action="store_true",
                    help="Control arm: frozen V-JEPA ViT-L with random (untrained) weights")
    ap.add_argument("--disable_igab", action="store_true")
    ap.add_argument("--disable_lic", action="store_true")
    ap.add_argument("--disable_bottleneck", action="store_true",
                    help="Drop temporal mixer + CFIM at the bottleneck")
    ap.add_argument("--keep_gate_router", action="store_true",
                    help="Keep the gate / router values stored in init_ckpt (adaptation runs)")
    ap.add_argument("--train_data", default="sdsd", choices=["sdsd", "bvi_fewshot"])
    ap.add_argument("--bvi_scenes", default="S01_colour_chart,S09_kitchen,S17_faces2,S25_figures2")
    ap.add_argument("--w_pix", type=float, default=W_PIX)
    ap.add_argument("--w_retinex", type=float, default=W_RETINEX)
    ap.add_argument("--w_smooth", type=float, default=W_SMOOTH)
    ap.add_argument("--w_jepa", type=float, default=W_JEPA)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.out_dir, f"train_{stamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger()
    log.info(f"=== {args.tag}: v35 Stage 4 (retrainer) "
             f"gate_init={args.gate_init} router_c={args.router_c} ===")
    log.info(f"init_ckpt: {args.init_ckpt}")
    log.info(f"out_dir:   {args.out_dir}")
    log.info(f"log:       {log_path}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")

    backbone = None
    refiner = None
    if not args.no_vjepa:
      if args.random_vjepa:
          torch.manual_seed(RANDOM_VJEPA_SEED)
      backbone = VJepaBackboneLocal21(
        local_repo=VJEPA_REPO, local_ckpt=("" if args.random_vjepa else VJEPA_CKPT),
        feature_layers=(5, 11, 17, 23),
        image_size=256, frames_per_clip=args.clip_len,
        patch_size=16, tubelet_size=2, freeze=True,
      ).to(device)
      backbone.eval()
      for p in backbone.parameters():
          p.requires_grad_(False)
      if args.random_vjepa:
          args.backbone_fp = param_fingerprint(backbone.model)
          torch.manual_seed(args.seed)
          log.info(f"  random V-JEPA backbone, fingerprint={args.backbone_fp:.6f}")

      refiner = CounterfactualRefiner(
        local_repo=VJEPA_REPO, ckpt_path=VJEPA_CKPT,
        encoder_embed_dim=1024,
        image_size=256, num_frames=args.clip_len,
        patch_size=16, tubelet_size=2,
        predictor_embed_dim=384, depth=12, num_heads=12,
        lora_rank=args.lora_rank, freeze_body=True,
      ).to(device)

    pixel_branch = VJepaPixelBranch(
        in_channels=3,
        vjepa_dim=1024,
        vjepa_grid=(args.clip_len // 2, 16, 16),
        num_vjepa_layers=NUM_VJEPA_LAYERS,
        cross_attn_at_scales=(1, 2, 3, 4),
        use_cfim_bottleneck=True,
        use_illumination_cues=True,
        use_gradient_checkpoint=True,
        router_init_c=args.router_c,
    ).to(device)

    state = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "pixel_branch" not in state or "cf_refiner" not in state:
        raise RuntimeError(
            f"init_ckpt {args.init_ckpt} missing pixel_branch/cf_refiner keys; "
            f"got {list(state.keys()) if isinstance(state, dict) else type(state)}"
        )
    miss_pb, unex_pb = pixel_branch.load_state_dict(state["pixel_branch"], strict=False)
    if refiner is not None:
        miss_rf, unex_rf = refiner.load_state_dict(state["cf_refiner"], strict=False)
    else:
        miss_rf, unex_rf = [], []
    log.info(f"  pixel_branch load: missing={len(miss_pb)} unexpected={len(unex_pb)}")
    log.info(f"  cf_refiner   load: missing={len(miss_rf)} unexpected={len(unex_rf)}")
    if miss_pb:
        log.info(f"    pb-missing (first 5): {list(miss_pb)[:5]}")
    if miss_rf:
        log.info(f"    rf-missing (first 5): {list(miss_rf)[:5]}")

    with torch.no_grad():
      if not args.keep_gate_router:
        if refiner is not None:
            refiner.gate.fill_(args.gate_init)
        router_init = torch.zeros(5, NUM_VJEPA_LAYERS)
        idx_per_scale = [0, 0, 1, 2, 3]
        for s, li in enumerate(idx_per_scale):
            router_init[s, li] = float(args.router_c)
        pixel_branch.router_logits.copy_(router_init.to(pixel_branch.router_logits))
    apply_pixel_branch_variant(pixel_branch, args.disable_igab, args.disable_lic,
                               args.disable_bottleneck)
    log.info(f"  variant: random_vjepa={args.random_vjepa} disable_igab={args.disable_igab} "
             f"disable_lic={args.disable_lic} disable_bottleneck={args.disable_bottleneck} "
             f"w=({args.w_pix},{args.w_retinex},{args.w_smooth},{args.w_jepa})")
    gate_now = float(refiner.gate.detach().item()) if refiner is not None else float("nan")
    router_softmax0 = F.softmax(pixel_branch.router_logits, dim=1)[0].cpu().tolist()
    log.info(f"  gate (post-override)   = {gate_now:+.4f}")
    log.info(f"  router[0] softmax post = {['%.4f' % v for v in router_softmax0]}")

    trainable = ([p for p in refiner.parameters() if p.requires_grad] if refiner is not None else []) + \
                [p for p in pixel_branch.parameters() if p.requires_grad]
    log.info(f"  no_vjepa={args.no_vjepa}  save_at={args.save_at}  seed={args.seed}")
    n_train = sum(p.numel() for p in trainable) / 1e6
    log.info(f"  trainable: {n_train:.2f}M "
             f"(pixel_branch + cf_refiner LoRA/gate/new_proj)")

    optim = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
    )

    if args.train_data == "bvi_fewshot":
        loader = build_bvi_fewshot_loader(args.bvi_scenes.split(","), args.crop, args.clip_len,
                                          args.num_workers)
    else:
        loader = build_train_loader(args.crop, args.clip_len, args.num_workers)
    audit_seqs = build_audit_seqs()
    log.info(f"  audit sequences (SDSD-indoor test): {len(audit_seqs)}")

    t0 = time.time()
    rgb0, y0 = audit_indoor_512(backbone, refiner, pixel_branch, audit_seqs, device,
                                clip_len=args.clip_len)
    log.info(f"Zero-iter audit ({time.time() - t0:.1f}s): "
             f"RGB-512={rgb0:.3f}  Y-512={y0:.3f}")
    best_rgb = rgb0
    best_y = y0

    pixel_branch.train()
    if refiner is not None:
        refiner.train()
    optim.zero_grad(set_to_none=True)

    it = 0
    accum = 0
    running_total = 0.0
    running_pix = 0.0
    running_jepa = 0.0
    step_start = time.time()

    data_iter = iter(loader)
    while it < args.iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["hq"].to(device, non_blocking=True)

        if args.no_vjepa:
            z_pixel = zero_tokens(lq.shape[0], device)
        else:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_L_layers = backbone(lq)["feats"]
                    z_H_ref = backbone(gt)["feats"][REF_LAYER]
                z_L_layers = [z.float() for z in z_L_layers]
                z_H_ref = z_H_ref.float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                refined_ref = refiner(z_L_layers[REF_LAYER].bfloat16())
            refined_ref_f = refined_ref.float()
            z_pixel = list(z_L_layers[:REF_LAYER]) + [refined_ref_f]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = pixel_branch(lq, z_pixel)
        pred = out["pred_pixels"].float()
        R_hat = out["R_hat"].float()
        L_hat = out["L_hat"].float()

        l_pix = charbonnier_loss(pred, gt)
        l_ret = retinex_compose_loss(R_hat, L_hat, gt)
        l_smo = illumination_smoothness_loss(L_hat)
        l_jep = (pred.new_zeros(()) if args.no_vjepa
                 else F.l1_loss(refined_ref_f, z_H_ref))
        l_total = (args.w_pix * l_pix + args.w_retinex * l_ret
                   + args.w_smooth * l_smo + args.w_jepa * l_jep)

        (l_total / args.accum_steps).backward()
        accum += 1
        running_total += float(l_total.item())
        running_pix += float(l_pix.item())
        running_jepa += float(l_jep.item())

        if accum >= args.accum_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            accum = 0
            it += 1

            if it % args.log_every == 0:
                dt = time.time() - step_start
                step_start = time.time()
                gate_val = float(refiner.gate.detach().item()) if refiner is not None else float("nan")
                r_soft = F.softmax(pixel_branch.router_logits.detach(), dim=1)
                max_per_scale = r_soft.max(dim=1).values.cpu().tolist()
                log.info(
                    f"it {it:5d}/{args.iters} "
                    f"total={running_total / args.log_every / args.accum_steps:.4f} "
                    f"pix={running_pix / args.log_every / args.accum_steps:.4f} "
                    f"jepa={running_jepa / args.log_every / args.accum_steps:.4f} "
                    f"gate={gate_val:+.4f} "
                    f"router_max_per_scale={['%.6f' % v for v in max_per_scale]} "
                    f"step={dt / args.log_every:.2f}s"
                )
                running_total = 0.0
                running_pix = 0.0
                running_jepa = 0.0

            if it in args.save_at:
                torch.save(
                    {"step": it, "pixel_branch": pixel_branch.state_dict(),
                     "cf_refiner": refiner.state_dict() if refiner is not None else None,
                     "args": vars(args)},
                    os.path.join(args.out_dir, f"step_{it}.pt"),
                )
                log.info(f"  -> saved step_{it}.pt")

            if it % args.audit_every == 0:
                t0 = time.time()
                rgb, y = audit_indoor_512(
                    backbone, refiner, pixel_branch, audit_seqs, device,
                    clip_len=args.clip_len,
                )
                log.info(f"[audit @ {it}] ({time.time() - t0:.1f}s) "
                         f"RGB-512={rgb:.3f}  Y-512={y:.3f}")
                if y > best_y:
                    best_y = y
                    torch.save(
                        {"step": it, "pixel_branch": pixel_branch.state_dict(),
                         "cf_refiner": refiner.state_dict() if refiner is not None else None,
                         "best_y512": best_y, "best_rgb512": best_rgb},
                        os.path.join(args.out_dir, "best_y512.pt"),
                    )
                    log.info(f"  -> best Y-512 = {best_y:.3f} saved")
                if rgb > best_rgb:
                    best_rgb = rgb
                    torch.save(
                        {"step": it, "pixel_branch": pixel_branch.state_dict(),
                         "cf_refiner": refiner.state_dict() if refiner is not None else None,
                         "best_y512": best_y, "best_rgb512": best_rgb},
                        os.path.join(args.out_dir, "best_rgb512.pt"),
                    )
                    log.info(f"  -> best RGB-512 = {best_rgb:.3f} saved")

    torch.save(
        {"step": it, "pixel_branch": pixel_branch.state_dict(),
         "cf_refiner": refiner.state_dict() if refiner is not None else None,
         "best_y512": best_y, "best_rgb512": best_rgb,
         "args": vars(args)},
        os.path.join(args.out_dir, "final.pt"),
    )
    log.info(f"=== {args.tag} done: best Y-512 = {best_y:.3f}  "
             f"best RGB-512 = {best_rgb:.3f} ===")


if __name__ == "__main__":
    main()
