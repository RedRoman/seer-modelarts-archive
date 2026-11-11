"""Matched validation-only pilots for temporal detail fusion.

Each arm loads the same full-token-trained pixel weights and consumes the same
explicit sample plan. This is method development, not a SOTA benchmark result.
No test images are opened. Model selection may use the development validation
split; a later benchmark must separately specify its final protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "tools"))

from data.common import (clip_to_tensor, list_images, paired_hflip,
                         paired_random_crop, paired_vflip, read_image_01,
                         sample_clip_indices)
from losses.v6 import charbonnier_loss
from losses.v7 import illumination_smoothness_loss, retinex_compose_loss
from eval_e4 import (build_model, held_out_sequences, lpips_of,
                     psnr_y_uint8, rgb_psnr_uint8)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(a.shape), str(a.dtype))).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def make_plan(n_sequences, n_samples, seed):
    """Independent from model construction, worker count and device RNGs."""
    rng = np.random.default_rng(seed)
    plan = []
    while len(plan) < n_samples:
        for index in rng.permutation(n_sequences):
            if len(plan) == n_samples:
                break
            plan.append({"sequence_index": int(index),
                         "augmentation_seed": int(rng.integers(0, 2**63 - 1))})
    return plan


class PlannedSDSD(Dataset):
    def __init__(self, split_path, root, samples, seed, crop, clip_len):
        sp = json.loads(Path(split_path).read_text())
        self.items = []
        for key in ("SDSD_indoor", "SDSD_outdoor"):
            entry = sp[key]
            assert not set(entry["train"]) & (set(entry["val"]) | set(entry["test"]))
            base = entry["root"].replace("/temp/zihaohe/seer_cvpr", str(root))
            for name in entry["train"]:
                low = list_images(os.path.join(base, "input", name))
                high = list_images(os.path.join(base, "GT", name))
                k = min(len(low), len(high))
                if k == 0:
                    raise RuntimeError(f"Empty training sequence: {key}/{name}")
                self.items.append((f"{key}/{name}", low[:k], high[:k]))
        if len(self.items) != 136:
            raise ValueError(f"Expected frozen train136, found {len(self.items)}")
        self.plan = make_plan(len(self.items), samples, seed)
        self.crop, self.clip_len = crop, clip_len

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        spec = self.plan[index]
        name, low, high = self.items[spec["sequence_index"]]
        rng = np.random.default_rng(spec["augmentation_seed"])
        frames = sample_clip_indices(len(low), self.clip_len, rng)
        lq = np.stack([read_image_01(low[j]) for j in frames])
        gt = np.stack([read_image_01(high[j]) for j in frames])
        lq, gt = paired_random_crop(lq, gt, self.crop, rng)
        lq, gt = paired_hflip(lq, gt, .5, rng)
        lq, gt = paired_vflip(lq, gt, .5, rng)
        return {"lq": clip_to_tensor(lq), "gt": clip_to_tensor(gt),
                "name": name, "sample_index": index}


@torch.no_grad()
def evaluate(model, backbone, clips, dev, step, variant):
    model.eval()
    rows = []
    for name, lq_u8, gt_u8, kind in clips:
        assert kind == "video" and len(lq_u8) == 16
        x = torch.from_numpy(lq_u8.copy()).float().div_(255).permute(3, 0, 1, 2)[None].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            z = [a.float() for a in backbone(x)["feats"]]
            output = model(x, z)
        pred = output["pred_pixels"].float().clamp(0, 1)[0].permute(1, 2, 3, 0)
        pred_u8 = (pred.cpu().numpy() * 255).round().astype(np.uint8)
        p = torch.from_numpy(pred_u8.copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
        g = torch.from_numpy(gt_u8.copy()).float().permute(0, 3, 1, 2).div_(255).to(dev)
        tlp, gt_tlp = lpips_of(p[1:], p[:-1], dev), lpips_of(g[1:], g[:-1], dev)
        row = {"name": name,
               "psnr_y": float(np.mean([psnr_y_uint8(a, b) for a, b in zip(pred_u8, gt_u8)])),
               "psnr_rgb": float(np.mean([rgb_psnr_uint8(a, b) for a, b in zip(pred_u8, gt_u8)])),
               "lpips": lpips_of(p, g, dev), "tlp": tlp, "gt_tlp": gt_tlp,
               "tlp_gap": abs(tlp - gt_tlp)}
        if not all(np.isfinite(v) for k, v in row.items() if k != "name"):
            raise ValueError(f"Nonfinite validation metrics: {row}")
        rows.append(row)
    if len(rows) != 16 or len({r["name"] for r in rows}) != 16:
        raise ValueError("Validation must contain exactly 16 distinct frozen sequences")
    scores = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "name"}
    model.train()
    return {"variant": variant, "step": step, "split": "val", "n": len(rows),
            "metrics": scores, "per_sequence": rows,
            "protocol": "SDSD frozen val16, centre16 frames, centre256 crop, rounded uint8; development only"}


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "local", "prior"], required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--fusion_lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=20261111)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", choices=["npu", "cuda"], default="npu")
    ap.add_argument("--base_git", default="04313bba457fd53decbfdccde13168a001ec671f")
    args = ap.parse_args()
    if min(args.steps, args.accum, args.eval_every) < 1:
        raise ValueError("steps, accum and eval_every must be positive")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(outdir / "train.log")])
    log = logging.getLogger("method_pilot")
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    dev = torch.device(args.device + ":0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    root = Path(os.environ.get("SEER_ROOT", "/temp/zihaohe/seer_cvpr"))
    splits = root / "datasets/splits_cvpr.json"
    dataset = PlannedSDSD(splits, root, args.steps * args.accum, args.data_seed, 256, 16)
    plan = {"data_seed": args.data_seed, "sequence_names": [i[0] for i in dataset.items],
            "samples": dataset.plan}
    write_json(outdir / "data_plan.json", plan)
    base, backbone, refiner, exposure, arm, clip_len, saved = build_model(args.init, dev)
    if refiner is not None or exposure is not None or arm != "pretrained" or clip_len != 16:
        raise ValueError("Pilot requires the default pretrained v2 checkpoint")
    if saved.get("branch") != "small" or saved.get("arch") != "v2":
        raise ValueError("Pilot requires small v2")
    baseline_sha = state_digest(base.state_dict())
    baseline_config = base.eval_reconstruction
    fusion_init_sha = None
    if args.variant == "baseline":
        model = base
    else:
        from models.pixel_branch_fusion import SeerFusionPixelBranch
        # A separate RNG context keeps base construction and data RNG independent.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 9000)
            model = SeerFusionPixelBranch(
                dims=(20, 40, 80, 160, 320), num_blocks=(2, 4, 4, 6, 8),
                heads=(2, 4, 4, 8, 8), vjepa_grid=(8, 16, 16),
                use_gradient_checkpoint=True, fusion_mode=args.variant,
                fusion_scale=3, fusion_radius=2, fusion_offset=2, fusion_tap=2)
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        if unexpected or any(not k.startswith("temporal_fusion.") for k in missing):
            raise ValueError((missing, unexpected))
        fusion_init_sha = state_digest(model.temporal_fusion.state_dict())
        model = model.to(dev)
        del base
    model.use_checkpoint = True
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    old = [p for n, p in params if not n.startswith("temporal_fusion.")]
    new = [p for n, p in params if n.startswith("temporal_fusion.")]
    groups = [{"params": old, "lr": args.lr}]
    if new:
        groups.append({"params": new, "lr": args.fusion_lr})
    optimizer = torch.optim.AdamW(groups, betas=(.9, .99), weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        generator=torch.Generator().manual_seed(args.data_seed),
                        persistent_workers=args.workers > 0, pin_memory=False)
    sources = [Path(__file__), CODE / "models/pixel_branch_v2.py",
               CODE / "models/vjepa_backbone.py", CODE / "models/pixel_branch_fusion.py",
               CODE / "models/temporal_detail_fusion.py", CODE / "data/common.py"]
    metadata = {"args": vars(args), "init_sha256": digest(args.init),
                "base_pixel_state_sha256": baseline_sha,
                "base_reconstruction": baseline_config,
                "fusion_initial_state_sha256": fusion_init_sha,
                "fusion_config": model.get_fusion_config() if hasattr(model, "get_fusion_config") else None,
                "split_sha256": digest(splits), "data_plan_sha256": digest(outdir / "data_plan.json"),
                "source_sha256": {str(p.relative_to(CODE)): digest(p) for p in sources if p.exists()},
                "torch_version": torch.__version__,
                "trainable_base": sum(p.numel() for p in old),
                "trainable_fusion": sum(p.numel() for p in new),
                "selection": "exploratory validation-only method development, no test evaluation"}
    write_json(outdir / "provenance.json", metadata)
    log.info("START variant=%s steps=%d accum=%d params=%d+%d base=%s plan=%s",
             args.variant, args.steps, args.accum, metadata["trainable_base"],
             metadata["trainable_fusion"], baseline_sha, metadata["data_plan_sha256"])
    clips = list(held_out_sequences("sdsd", "val", 16, 256))
    initial = evaluate(model, backbone, clips, dev, 0, args.variant)
    write_json(outdir / "val_step_000000.json", initial)
    log.info("VAL step=0 %s", initial["metrics"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    it = 0
    recent, first_batch_sha = [], None
    started = time.time()
    for micro, batch in enumerate(loader):
        if first_batch_sha is None:
            first_batch_sha = state_digest({"lq": batch["lq"], "gt": batch["gt"]})
            metadata["first_batch_sha256"] = first_batch_sha
            metadata["first_batch_name"] = batch["name"][0]
            write_json(outdir / "provenance.json", metadata)
        low, gt = batch["lq"].to(dev), batch["gt"].to(dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16):
            tokens = [z.float() for z in backbone(low)["feats"]]
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            result = model(low, tokens)
        pred, reflectance, illumination = [result[k].float() for k in ("pred_pixels", "R_hat", "L_hat")]
        pixel = charbonnier_loss(pred, gt)
        loss = (1.5 * pixel + .15 * retinex_compose_loss(reflectance, illumination, gt)
                + .02 * illumination_smoothness_loss(illumination))
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"Nonfinite training loss at microbatch {micro}")
        (loss / args.accum).backward()
        recent.append(float(loss.detach()))
        if (micro + 1) % args.accum:
            continue
        torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        it += 1
        if it % 20 == 0:
            aux = result.get("fusion_aux", {})
            stats = {k: float(v.detach().mean()) for k, v in aux.items()
                     if torch.is_tensor(v) and v.numel() == 1}
            log.info("TRAIN step=%d loss=%.6f seconds/step=%.3f fusion=%s", it,
                     float(np.mean(recent)), (time.time() - started) / it, stats)
            recent = []
        if it % args.eval_every == 0 or it == args.steps:
            report = evaluate(model, backbone, clips, dev, it, args.variant)
            write_json(outdir / f"val_step_{it:06d}.json", report)
            checkpoint = {"step": it, "args": vars(args), "pixel_branch": cpu_tree(model.state_dict()),
                          "optimizer": cpu_tree(optimizer.state_dict()),
                          "torch_rng_state": torch.get_rng_state(), "provenance": metadata}
            path = outdir / ("final.pt" if it == args.steps else f"step_{it:06d}.pt")
            torch.save(checkpoint, path)
            write_json(path.with_suffix(".sha256.json"), {"sha256": digest(path)})
            log.info("VAL step=%d %s", it, report["metrics"])
    log.info("COMPLETE variant=%s optimizer_steps=%d duration_seconds=%.3f", args.variant, it, time.time() - started)


if __name__ == "__main__":
    main()
