"""E4 training entry point: leave-one-domain-out over {SDSD, BVI, LOL}.

This is the experiment hypothesis H5 of notes/CVPR_PREREG.md turns on, and the
one that decides whether "Cross-Domain Prior" survives in the title: train on two
of the three domains, hold the third out entirely, with and without the V-JEPA
prior, and ask whether the prior buys anything on a domain never seen.

The training loop is the same as tools/seer_train_cvpr.py (E1); it is duplicated
rather than imported because E1 runs are in flight against that file and must
keep the code version they started with. The differences are all in the data:

  * --heldout {sdsd,bvi,lol} picks the held-out domain; training uses the train
    splits of the other two and validation their val splits.
  * BVI clips come from the frozen lists in splits_cvpr.json, not from the 80/20
    rule BVIRLVDataset derives internally.
  * LOL is still images. Each pair is replicated to a static clip of clip_len
    frames, so the temporal modules see a degenerate but valid clip. This is
    stated in the paper rather than hidden: a static clip is what a single image
    is, for a video model.

Arms: --vjepa pretrained | none (H5 compares these two on the held-out domain).
"""
from __future__ import annotations

import argparse
import json
import logging
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
                         paired_vflip, read_image_01, repeat_single_to_clip,
                         sample_clip_indices)
from losses.v6 import charbonnier_loss  # noqa: E402
from losses.v7 import illumination_smoothness_loss, retinex_compose_loss  # noqa: E402
from models.exposure_head import ExposureHead, apply_gain, target_log_gain  # noqa: E402
from models.pixel_branch_v2 import SeerV2PixelBranch  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
from seer_train_cvpr import (ListedSDSD, psnr, rgb_to_y, validate,  # noqa: E402
                             zero_tokens)

ROOT = os.environ.get("SEER_ROOT", "/temp/zihaohe/seer_cvpr")
SPLITS = f"{ROOT}/datasets/splits_cvpr.json"
SPLITS_LOL = f"{ROOT}/datasets/splits_lol.json"
VJEPA_REPO = f"{ROOT}/modelzoo/vjepa2-main"
VJEPA_CKPT = f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt"
REF_LAYER, N_LAYERS, N_TOKENS, VJEPA_DIM = 3, 4, 2048, 1024
W_PIX, W_RET, W_SMO = 1.5, 0.15, 0.02
RANDOM_VJEPA_SEED = 1234
DOMAINS = ("sdsd", "bvi", "lol")


class ListedBVI(Dataset):
    """BVI-RLV clips named `SCENE__FPSfps`, taken from the frozen split lists."""

    def __init__(self, root, names, clip_len, crop, is_train):
        self.items = []
        for n in names:
            base, fps = n.split("__")
            fps = fps.replace("fps", "")
            lq = list_images(os.path.join(root, base, f"low_light_{fps}"))
            gt = list_images(os.path.join(root, base, f"normal_light_{fps}"))
            k = min(len(lq), len(gt))
            if k >= 1:
                self.items.append((n, lq[:k], gt[:k]))
        self.clip_len, self.crop, self.is_train = clip_len, crop, is_train
        if not self.items:
            raise RuntimeError(f"no BVI sequences under {root} for {len(names)} names")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, lq_p, gt_p = self.items[i]
        rng = make_train_rng() if self.is_train else make_eval_rng(i)
        idx = sample_clip_indices(len(lq_p), self.clip_len,
                                  rng=rng if self.is_train else None)
        lq = np.stack([read_image_01(lq_p[j]) for j in idx])
        gt = np.stack([read_image_01(gt_p[j]) for j in idx])
        if self.is_train:
            lq, gt = paired_random_crop(lq, gt, self.crop, rng)
            lq, gt = paired_hflip(lq, gt, 0.5, rng)
            lq, gt = paired_vflip(lq, gt, 0.5, rng)
        else:
            lq, gt = paired_center_crop(lq, gt, self.crop)
        return {"lq": clip_to_tensor(lq), "hq": clip_to_tensor(gt), "name": name}


class LOLPairs(Dataset):
    """LOL image pairs presented as static clips of clip_len frames."""

    def __init__(self, pairs, clip_len, crop, is_train):
        self.items = list(pairs)          # [(name, lq_path, gt_path), ...]
        self.clip_len, self.crop, self.is_train = clip_len, crop, is_train
        if not self.items:
            raise RuntimeError("LOLPairs got an empty list")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, lq_p, gt_p = self.items[i]
        rng = make_train_rng() if self.is_train else make_eval_rng(i)
        lq = repeat_single_to_clip(read_image_01(lq_p), self.clip_len)
        gt = repeat_single_to_clip(read_image_01(gt_p), self.clip_len)
        if self.is_train:
            lq, gt = paired_random_crop(lq, gt, self.crop, rng)
            lq, gt = paired_hflip(lq, gt, 0.5, rng)
            lq, gt = paired_vflip(lq, gt, 0.5, rng)
        else:
            lq, gt = paired_center_crop(lq, gt, self.crop)
        return {"lq": clip_to_tensor(lq), "hq": clip_to_tensor(gt), "name": name}


def remap(entry):
    e = dict(entry)
    if isinstance(e.get("root"), str):
        e["root"] = e["root"].replace("/temp/zihaohe/seer_cvpr", ROOT)
    return e


def build(domain, sub, clip_len, crop, is_train):
    """Datasets for one domain and one split ('train' or 'val')."""
    sp = json.load(open(SPLITS))
    if domain == "sdsd":
        out = []
        for key in ("SDSD_indoor", "SDSD_outdoor"):
            e = remap(sp[key])
            out.append(ListedSDSD(e["root"], e[sub], clip_len, crop, is_train))
        return out
    if domain == "bvi":
        e = remap(sp["BVI_RLV"])
        return [ListedBVI(e["root"], e[sub], clip_len, crop, is_train)]
    if domain == "lol":
        spl = json.load(open(SPLITS_LOL))
        out = []
        for key, e in spl.items():
            if not isinstance(e, dict) or "root" not in e:
                continue
            e = remap(e)
            s = e["splits"][sub]
            pairs = [(f"{key}/{n}", os.path.join(e["root"], s["lq"], n),
                      os.path.join(e["root"], s["gt"], n)) for n in s["names"]]
            if pairs:
                out.append(LOLPairs(pairs, clip_len, crop, is_train))
        return out
    raise ValueError(f"unknown domain {domain!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--heldout", required=True, choices=list(DOMAINS))
    ap.add_argument("--branch", choices=["full", "small"], default="full",
                    help="small is the 22.4M branch; at ~6 s/it it is the only "
                         "configuration reaching 20000 iterations in time")
    ap.add_argument("--only", default="", choices=["", *DOMAINS],
                    help="train on this single domain instead of the two "
                         "that are not held out (E12: in-domain on a truly "
                         "dynamic dataset, where --only bvi --heldout sdsd "
                         "trains and validates on BVI alone)")
    ap.add_argument("--vjepa", choices=["pretrained", "none", "random"], default="pretrained")
    ap.add_argument("--edc", choices=["on", "off"], default="off")
    ap.add_argument("--edc_vjepa", choices=["on", "off"], default="on")
    ap.add_argument("--edc_weight", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--accum_steps", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--grad_ckpt", choices=["on", "off"], default="on",
                    help="gradient checkpointing; off is mathematically identical "
                         "and faster when memory allows")
    ap.add_argument("--balance_domains", choices=["on", "off"], default="on",
                    help="sample each training domain with equal probability")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["npu", "cuda"])
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(f"{args.device}:0")
    amp_dtype = torch.bfloat16
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(
            os.path.join(args.out_dir, f"train_{time.strftime('%Y%m%d-%H%M%S')}.log"),
            mode="w"), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    train_domains = ([args.only] if args.only
                     else [d for d in DOMAINS if d != args.heldout])
    log.info(f"=== {args.tag} | heldout={args.heldout} train_on={train_domains} "
             f"vjepa={args.vjepa} seed={args.seed} iters={args.iters} "
             f"device={args.device} ===")

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    backbone = None
    if args.vjepa != "none":
        if args.vjepa == "random":
            torch.manual_seed(RANDOM_VJEPA_SEED)
        backbone = VJepaBackboneLocal21(
            local_repo=VJEPA_REPO,
            local_ckpt=("" if args.vjepa == "random" else VJEPA_CKPT),
            feature_layers=(5, 11, 17, 23), image_size=256,
            frames_per_clip=args.clip_len, patch_size=16, tubelet_size=2,
            freeze=True).to(dev).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        if args.vjepa == "random":
            torch.manual_seed(args.seed)

    _shape = (dict(dims=(20, 40, 80, 160, 320), num_blocks=(2, 4, 4, 6, 8),
                   heads=(2, 4, 4, 8, 8)) if args.branch == "small" else {})
    pb = SeerV2PixelBranch(
        in_channels=3, vjepa_dim=VJEPA_DIM, vjepa_grid=(args.clip_len // 2, 16, 16),
        num_vjepa_layers=N_LAYERS, cross_attn_at_scales=(1, 2, 3, 4),
        use_cfim_bottleneck=True, use_illumination_cues=True,
        use_gradient_checkpoint=(args.grad_ckpt == "on"), **_shape).to(dev)

    exp_head = None
    if args.edc == "on":
        exp_head = ExposureHead(
            vjepa_dim=VJEPA_DIM,
            use_vjepa=(args.edc_vjepa == "on" and args.vjepa != "none")).to(dev)

    trainable = ([p for p in exp_head.parameters()] if exp_head is not None else []) + \
                [p for p in pb.parameters() if p.requires_grad]
    log.info(f"  trainable {sum(p.numel() for p in trainable) / 1e6:.2f}M")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.99))

    tr_sets, va_sets, per_domain = [], [], []
    for d in train_domains:
        s = build(d, "train", args.clip_len, args.crop, True)
        per_domain.append((d, sum(len(x) for x in s)))
        tr_sets += s
        va_sets += build(d, "val", args.clip_len, args.crop, False)
    train_ds, val_ds = ConcatDataset(tr_sets), ConcatDataset(va_sets)
    log.info(f"  train seqs {len(train_ds)} from {dict(per_domain)}; val seqs {len(val_ds)}")

    # LOL contributes ~1.9k image pairs against ~0.14k video sequences, so an
    # unweighted sampler would make "hold out SDSD" and "hold out BVI" train
    # almost entirely on static single-image clips while "hold out LOL" trains on
    # video. That would confound H5 with a change of training modality, so each
    # domain is sampled with equal probability instead.
    sampler = None
    if args.balance_domains == "on" and len(per_domain) > 1:
        w, k = [], len(per_domain)
        for _, n in per_domain:
            w += [1.0 / (k * n)] * n
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(w, dtype=torch.double), num_samples=len(train_ds),
            replacement=True)
        log.info("  domain-balanced sampling: "
                 + ", ".join(f"{d} p={1.0 / k:.3f} over {n} seqs"
                             for (d, n) in per_domain))
    loader = DataLoader(train_ds, batch_size=1, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    pb.train()
    opt.zero_grad(set_to_none=True)
    it = accum = 0
    run = {"total": 0.0, "pix": 0.0, "exp": 0.0}
    t0 = time.time()
    di = iter(loader)
    while it < args.iters:
        try:
            b = next(di)
        except StopIteration:
            di = iter(loader); b = next(di)
        lq, gt = b["lq"].to(dev, non_blocking=True), b["hq"].to(dev, non_blocking=True)
        if args.vjepa == "none":
            zp = zero_tokens(lq.shape[0], dev)
        else:
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=amp_dtype):
                zl = backbone(lq)["feats"]
            zp = [z.float() for z in zl]
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            out = pb(lq, zp)
        pred, R, L = out["pred_pixels"].float(), out["R_hat"].float(), out["L_hat"].float()
        l_exp = pred.new_zeros(())
        if exp_head is not None:
            tok = zp[REF_LAYER] if args.vjepa != "none" else None
            g_hat = exp_head(lq, tok.detach() if tok is not None else None)
            l_exp = F.l1_loss(g_hat, target_log_gain(lq, gt))
        l_pix = charbonnier_loss(pred, gt)
        loss = (W_PIX * l_pix + W_RET * retinex_compose_loss(R, L, gt)
                + W_SMO * illumination_smoothness_loss(L)
                + args.edc_weight * l_exp)
        (loss / args.accum_steps).backward()
        accum += 1
        run["total"] += float(loss.detach()); run["pix"] += float(l_pix.detach())
        run["exp"] += float(l_exp.detach())
        if accum >= args.accum_steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            accum = 0; it += 1
            if it % args.log_every == 0:
                n = args.log_every * args.accum_steps
                log.info(f"it {it:5d}/{args.iters} total={run['total'] / n:.4f} "
                         f"pix={run['pix'] / n:.4f} exp={run['exp'] / n:.4f} "
                         f"{(time.time() - t0) / args.log_every:.2f}s/it")
                run = {k: 0.0 for k in run}; t0 = time.time()
            if it % args.val_every == 0 or it == args.iters:
                v = validate(backbone, None, pb, val_loader, dev, amp_dtype,
                             exp_head, args.vjepa != "none")
                log.info(f"[val @ {it}] Y-PSNR {v:.3f} "
                         f"(train-domain val split, not used for selection)")
            if it % args.save_every == 0 and it < args.iters:
                torch.save({"step": it, "pixel_branch": pb.state_dict(),
                            "exposure_head": exp_head.state_dict() if exp_head else None,
                            "args": vars(args)},
                           os.path.join(args.out_dir, f"step_{it}.pt"))
    torch.save({"step": it, "pixel_branch": pb.state_dict(),
                "exposure_head": exp_head.state_dict() if exp_head else None,
                "args": vars(args)}, os.path.join(args.out_dir, "final.pt"))
    log.info(f"=== {args.tag} done ===")


if __name__ == "__main__":
    main()
