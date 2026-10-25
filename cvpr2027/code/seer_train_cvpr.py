"""E1 training entry point for the CVPR round: from-scratch or warm-started Seer.

Differences from tools/v35_stage4_ctrl.py (the ICLR-round script):
  * --init_ckpt is optional; without it every trainable module starts from its
    own initialisation (pixel branch random, M1 gate 0, router at c), which is
    what hypothesis H2 of notes/CVPR_PREREG.md asks for.
  * train/val sequence lists come from datasets/splits_cvpr.json, so no sequence
    of the test or val split can enter training.
  * validation (not test) is scored every --val_every steps and logged; the final
    checkpoint is the one evaluated, never a validation-selected one.
  * device is npu or cuda, chosen by --device.

Arms (H1/H2):  --vjepa pretrained | none | random
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "tools"))

from data.common import (clip_to_tensor, list_images, make_eval_rng, make_train_rng,  # noqa: E402
                         paired_center_crop, paired_hflip, paired_random_crop,
                         paired_vflip, read_image_01, sample_clip_indices)
from losses.v6 import charbonnier_loss  # noqa: E402
from losses.v7 import illumination_smoothness_loss, retinex_compose_loss  # noqa: E402
from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.pixel_branch_v2 import SeerV2PixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
from models.exposure_head import (ExposureHead, apply_gain, target_log_gain)  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/temp/zihaohe/seer_cvpr")
SPLITS = f"{ROOT}/datasets/splits_cvpr.json"
VJEPA_REPO = f"{ROOT}/modelzoo/vjepa2-main"
VJEPA_CKPT = f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt"
REF_LAYER, N_LAYERS, N_TOKENS, VJEPA_DIM = 3, 4, 2048, 1024
W_PIX, W_RET, W_SMO, W_JEP = 1.5, 0.15, 0.02, 0.05
RANDOM_VJEPA_SEED = 1234


class ListedSDSD(Dataset):
    """SDSD sequences given explicitly by name, so splits are never re-derived."""

    def __init__(self, root, names, clip_len, crop, is_train):
        self.items = []
        for n in names:
            lq = list_images(os.path.join(root, "input", n))
            gt = list_images(os.path.join(root, "GT", n))
            k = min(len(lq), len(gt))
            if k >= 1:
                self.items.append((n, lq[:k], gt[:k]))
        self.clip_len, self.crop, self.is_train = clip_len, crop, is_train
        if not self.items:
            raise RuntimeError(f"no sequences under {root} for {len(names)} names")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, lq_p, gt_p = self.items[i]
        rng = make_train_rng() if self.is_train else make_eval_rng(i)
        idx = sample_clip_indices(len(lq_p), self.clip_len, rng=rng if self.is_train else None)
        lq = np.stack([read_image_01(lq_p[j]) for j in idx])
        gt = np.stack([read_image_01(gt_p[j]) for j in idx])
        if self.is_train:
            lq, gt = paired_random_crop(lq, gt, self.crop, rng)
            lq, gt = paired_hflip(lq, gt, 0.5, rng)
            lq, gt = paired_vflip(lq, gt, 0.5, rng)
        else:
            lq, gt = paired_center_crop(lq, gt, self.crop)
        return {"lq": clip_to_tensor(lq), "hq": clip_to_tensor(gt), "name": name}


def rgb_to_y(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def psnr(pred, gt, eps=1e-10):
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    return -10 * torch.log10(mse)


def zero_tokens(b, dev):
    z = torch.zeros(b, N_TOKENS, VJEPA_DIM, device=dev)
    return [z] * N_LAYERS


@torch.no_grad()
def validate(backbone, refiner, pb, loader, dev, amp_dtype, exp_head=None, use_vjepa=True):
    was = pb.training
    pb.eval()
    if refiner is not None:
        refiner.eval()
    ps = []
    for b in loader:
        lq, gt = b["lq"].to(dev), b["hq"].to(dev)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            if backbone is None:
                z = zero_tokens(lq.shape[0], dev)
            elif refiner is None:
                z = list(backbone(lq)["feats"])
            else:
                f = backbone(lq)["feats"]
                z = list(f[:REF_LAYER]) + [refiner(f[REF_LAYER].to(amp_dtype))]
            out = pb(lq, z)
        pr = out["pred_pixels"].float()
        if exp_head is not None:
            tok = z[REF_LAYER] if use_vjepa else None
            pr = apply_gain(pr, exp_head(lq, tok), lq)
        p = pr.clamp(0, 1)[0].permute(1, 0, 2, 3)
        g = gt[0].permute(1, 0, 2, 3)
        ps.append(psnr(rgb_to_y(p), rgb_to_y(g)).mean().item())
    pb.train(was)
    if refiner is not None:
        refiner.train(was)
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vjepa", choices=["pretrained", "none", "random"], default="pretrained")
    ap.add_argument("--edc", choices=["on", "off"], default="off",
                    help="exposure-decoupled conditioning: predict the global gain separately")
    ap.add_argument("--edc_vjepa", choices=["on", "off"], default="on",
                    help="let the gain head see the pooled V-JEPA token (scene prior)")
    ap.add_argument("--edc_weight", type=float, default=1.0, help="weight of the log-gain loss")
    ap.add_argument("--no_edc_detach", dest="edc_detach", action="store_false",
                    help="let the gain gradient reach the pixel branch; the "
                         "default keeps them decoupled, which is the point of EDC")
    ap.add_argument("--edc_detach", action="store_true", default=True,
                    help="keep the gain gradient out of the pixel branch")
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2",
                    help="v2 drops M1/M2/M3 and IGAB, which the ablations showed do not help")
    ap.add_argument("--init_ckpt", default="", help="optional warm start; empty = from scratch")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--accum_steps", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=1500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--router_c", type=float, default=4.0)
    ap.add_argument("--gate_init", type=float, default=0.0)
    ap.add_argument("--taps", default="5,11,17,23",
                    help="V-JEPA blocks read as conditioning (E2/H3). The default "
                         "spans shallow to deep; 23,23,23,23 gives every scale "
                         "the deepest block only, 17,17,23,23 the top two.")
    ap.add_argument("--kv", choices=["per_tubelet", "full", "hybrid"],
                    default="per_tubelet",
                    help="how much of the clip each query frame sees (E15/H12); "
                         "per_tubelet is the current behaviour")
    ap.add_argument("--xattn", choices=["global", "spatial"], default="global",
                    help="spatial gives queries and tokens a shared positional "
                         "basis so a pixel can prefer its own location (E11/H10)")
    ap.add_argument("--branch", choices=["full", "small"], default="full",
                    help="small is the ~22M pixel branch for E3/H4; full is 361M")
    ap.add_argument("--device", default="npu", choices=["npu", "cuda"])
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(f"{args.device}:0")
    amp_dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(args.out_dir, f"train_{time.strftime('%Y%m%d-%H%M%S')}.log"),
                                      mode="w"), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    log.info(f"=== {args.tag} | vjepa={args.vjepa} init={args.init_ckpt or 'SCRATCH'} "
             f"seed={args.seed} iters={args.iters} device={args.device} ===")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    sp = json.load(open(SPLITS))
    # the splits file carries absolute roots from the host that built it
    for _k, _v in sp.items():
        if isinstance(_v, dict) and isinstance(_v.get('root'), str):
            _v['root'] = _v['root'].replace('/temp/zihaohe/seer_cvpr', ROOT)

    backbone = refiner = None
    if args.vjepa != "none":
        if args.vjepa == "random":
            torch.manual_seed(RANDOM_VJEPA_SEED)
        backbone = VJepaBackboneLocal21(
            local_repo=VJEPA_REPO, local_ckpt=("" if args.vjepa == "random" else VJEPA_CKPT),
            feature_layers=tuple(int(t) for t in args.taps.split(",")),
            image_size=256, frames_per_clip=args.clip_len,
            patch_size=16, tubelet_size=2, freeze=True).to(dev).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        if args.vjepa == "random":
            fp = float(sum(p.double().sum() for p in list(backbone.model.parameters())[:8]))
            log.info(f"  random V-JEPA fingerprint {fp:.6f}")
            torch.manual_seed(args.seed)
        if args.arch == "v1":
            refiner = CounterfactualRefiner(
            local_repo=VJEPA_REPO, ckpt_path=VJEPA_CKPT, encoder_embed_dim=VJEPA_DIM,
            image_size=256, num_frames=args.clip_len, patch_size=16, tubelet_size=2,
            predictor_embed_dim=384, depth=12, num_heads=12,
                lora_rank=args.lora_rank, freeze_body=True).to(dev)

    if args.arch == "v2":
        # E3/H4 asks whether a 361M branch memorises the source domain through
        # the tokens; the small variant keeps every component and only narrows
        # and shortens the stages, so the arms differ in capacity alone.
        shape = (dict(dims=(20, 40, 80, 160, 320), num_blocks=(2, 4, 4, 6, 8),
                      heads=(2, 4, 4, 8, 8)) if args.branch == "small" else {})
        pb = SeerV2PixelBranch(
            in_channels=3, vjepa_dim=VJEPA_DIM, vjepa_grid=(args.clip_len // 2, 16, 16),
            num_vjepa_layers=N_LAYERS, cross_attn_at_scales=(1, 2, 3, 4),
            use_cfim_bottleneck=True, use_illumination_cues=True,
            use_gradient_checkpoint=True, xattn=args.xattn,
            kv_mode=args.kv, **shape).to(dev)
    else:
        pb = VJepaPixelBranch(
        in_channels=3, vjepa_dim=VJEPA_DIM, vjepa_grid=(args.clip_len // 2, 16, 16),
        num_vjepa_layers=N_LAYERS, cross_attn_at_scales=(1, 2, 3, 4),
        use_cfim_bottleneck=True, use_illumination_cues=True,
            use_gradient_checkpoint=True, router_init_c=args.router_c).to(dev)

    if args.init_ckpt:
        st = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        m, u = pb.load_state_dict(st["pixel_branch"], strict=False)
        log.info(f"  warm start pixel_branch missing={len(m)} unexpected={len(u)}")
        if refiner is not None and st.get("cf_refiner"):
            m, u = refiner.load_state_dict(st["cf_refiner"], strict=False)
            log.info(f"  warm start cf_refiner missing={len(m)} unexpected={len(u)}")
    else:
        log.info("  FROM SCRATCH: no checkpoint loaded")
    with torch.no_grad():
        if refiner is not None:
            refiner.gate.fill_(args.gate_init)
        if hasattr(pb, "router_logits"):
            ri = torch.zeros(5, N_LAYERS)
            for s_, li in enumerate([0, 0, 1, 2, 3]):
                ri[s_, li] = float(args.router_c)
            pb.router_logits.copy_(ri.to(pb.router_logits))

    exp_head = None
    if args.edc == "on":
        exp_head = ExposureHead(vjepa_dim=VJEPA_DIM,
                                use_vjepa=(args.edc_vjepa == "on" and args.vjepa != "none")).to(dev)
        log.info(f"  EDC on (vjepa prior: {args.edc_vjepa}, weight {args.edc_weight})")

    trainable = ([p for p in refiner.parameters() if p.requires_grad] if refiner else []) + \
                ([p for p in exp_head.parameters()] if exp_head is not None else []) + \
                [p for p in pb.parameters() if p.requires_grad]
    log.info(f"  trainable {sum(p.numel() for p in trainable)/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))

    mk = lambda key, sub, tr: ListedSDSD(sp[key]["root"], sp[key][sub], args.clip_len, args.crop, tr)
    train_ds = ConcatDataset([mk("SDSD_indoor", "train", True), mk("SDSD_outdoor", "train", True)])
    val_ds = ConcatDataset([mk("SDSD_indoor", "val", False), mk("SDSD_outdoor", "val", False)])
    log.info(f"  train seqs {len(train_ds)}  val seqs {len(val_ds)}")
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    pb.train()
    if refiner is not None:
        refiner.train()
    opt.zero_grad(set_to_none=True)
    it = accum = 0
    run = {"total": 0.0, "pix": 0.0, "jepa": 0.0, "exp": 0.0}
    t0 = time.time()
    di = iter(loader)
    while it < args.iters:
        try:
            b = next(di)
        except StopIteration:
            di = iter(loader); b = next(di)
        lq, gt = b["lq"].to(dev, non_blocking=True), b["hq"].to(dev, non_blocking=True)
        if args.vjepa == "none":
            zp = zero_tokens(lq.shape[0], dev); l_jep = lq.new_zeros(())
        elif refiner is None:
            # v2: frozen tokens go straight to the pixel branch; no M1, no M2
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dtype):
                zl = backbone(lq)["feats"]
            zp = [z.float() for z in zl]; l_jep = lq.new_zeros(())
        else:
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dtype):
                zl = backbone(lq)["feats"]; zh = backbone(gt)["feats"][REF_LAYER]
            zl = [z.float() for z in zl]; zh = zh.float()
            with torch.autocast(device_type=dev.type, dtype=amp_dtype):
                ref = refiner(zl[REF_LAYER].to(amp_dtype))
            ref_f = ref.float(); zp = list(zl[:REF_LAYER]) + [ref_f]
            l_jep = F.l1_loss(ref_f, zh)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            out = pb(lq, zp)
        pred, R, L = out["pred_pixels"].float(), out["R_hat"].float(), out["L_hat"].float()
        l_exp = pred.new_zeros(())
        if exp_head is not None:
            tok = zp[REF_LAYER] if args.vjepa != "none" else None
            g_hat = exp_head(lq, tok.detach() if tok is not None else None)
            l_exp = F.l1_loss(g_hat, target_log_gain(lq, gt))
            src = pred.detach() if args.edc_detach else pred
            pred_exp = apply_gain(src, g_hat.detach() if args.edc_detach else g_hat, lq)
            if not args.edc_detach:
                pred = pred_exp
        l_pix = charbonnier_loss(pred, gt)
        loss = (W_PIX * l_pix + W_RET * retinex_compose_loss(R, L, gt)
                + W_SMO * illumination_smoothness_loss(L) + W_JEP * l_jep
                + args.edc_weight * l_exp)
        (loss / args.accum_steps).backward()
        accum += 1
        run["total"] += float(loss); run["pix"] += float(l_pix); run["jepa"] += float(l_jep)
        run["exp"] += float(l_exp)
        if accum >= args.accum_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            accum = 0; it += 1
            if it % args.log_every == 0:
                n = args.log_every * args.accum_steps
                g = float(refiner.gate) if refiner is not None else float("nan")
                log.info(f"it {it:5d}/{args.iters} total={run['total']/n:.4f} pix={run['pix']/n:.4f} "
                         f"jepa={run['jepa']/n:.4f} exp={run['exp']/n:.4f} gate={g:+.5f} "
                         f"{(time.time()-t0)/args.log_every:.2f}s/it")
                run = {k: 0.0 for k in run}; t0 = time.time()
            if it % args.val_every == 0 or it == args.iters:
                v = validate(backbone, refiner, pb, val_loader, dev, amp_dtype,
                             exp_head, args.vjepa != "none")
                log.info(f"[val @ {it}] Y-PSNR {v:.3f} (validation split, not used for selection)")
            if it % args.save_every == 0 and it < args.iters:
                torch.save({"step": it, "pixel_branch": pb.state_dict(),
                            "cf_refiner": refiner.state_dict() if refiner else None,
                            "exposure_head": exp_head.state_dict() if exp_head is not None else None,
                            "args": vars(args)}, os.path.join(args.out_dir, f"step_{it}.pt"))
    torch.save({"step": it, "pixel_branch": pb.state_dict(),
                "cf_refiner": refiner.state_dict() if refiner else None,
                "exposure_head": exp_head.state_dict() if exp_head is not None else None,
                "args": vars(args)}, os.path.join(args.out_dir, "final.pt"))
    log.info(f"=== {args.tag} done ===")


if __name__ == "__main__":
    main()
