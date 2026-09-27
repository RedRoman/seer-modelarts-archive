"""Unified six-split evaluation for Seer-family checkpoints (ICLR re-run).

Reuses the baselines' data iterators and metric functions (baselines/common/
eval_lib.py + eval_lib_v2.py) so Seer and the baselines share one protocol:
video = center T=16 frames, 512x512 center crop; images = full resolution,
tiled to T=16 for V-JEPA and scored on the middle output frame. PSNR on uint8
RGB, LPIPS-alex, DISTS and CLIPIQA via pyiqa.

L0 is scored on Seer's own SDSD-indoor test split (15 seqs, SDSDDataset 80/20),
and the JSON also stores which of those sequences belong to STCD-12, so the
6 sequences held out for both Seer and STCD can be aggregated separately.

Outputs raw, +CP with the archived alpha = (0.50, 0.50, 0.68) ("HMP+CP"), and +CP
with alpha refit in closed form (Eq. alpha*) on a fixed set of 15 clips from
Seer's SDSD *training* split ("HMP+CPfit"; no test data involved), and +CP with
alpha cross-fitted on the target split itself ("HMP+CPxfit": scenes sorted by
name are split into two folds by index parity -- the 10/20 fps copies of a BVI
scene and the _2 pass of an SDSD scene stay in the same fold -- and each fold
uses the alpha fitted on the other fold, so no clip is scored with an alpha
fitted on its own scene).
Fitted alphas are clipped to [0, 1] so the fusion stays convex; raw values are stored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

VJ = "/temp/zihaohe/workspace/vjepa_llve"
BASE = "/temp/zihaohe/workspace/baselines"
CP_DIR = "/temp/zihaohe/workspace/quick_ablations/cp_transfer"
for p in (VJ, os.path.join(VJ, "tools"), BASE, CP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.eval_lib import (  # noqa: E402
    DATA_ROOTS, STCD_INDOOR_12, _load_img_any, center_crop,
    iter_sdsd_outdoor, iter_bvi, iter_lol_v1, iter_lol_v2,
    rgb_psnr_uint8, lpips_batch,
)
from common.eval_lib_v2 import dists_batch, clipiqa_batch  # noqa: E402
from cp_module import cp_surrogate, fuse_per_channel  # noqa: E402
from data.sdsd_dataset import SDSDDataset  # noqa: E402
from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
import v35_stage4_train as T4  # noqa: E402
from seer_variants import (  # noqa: E402
    RANDOM_VJEPA_SEED, apply_pixel_branch_variant, param_fingerprint,
)
from temporal_metrics import safe_gt_flows, temporal_metrics  # noqa: E402

ALPHA = torch.tensor([0.50, 0.50, 0.68])
CLIP_LEN = 16


def iter_sdsd_train_fitset(n_frames: int = 16, crop: int = 512, n_in: int = 8, n_out: int = 7):
    """Fixed alpha-fitting pool: evenly spaced clips from Seer's SDSD train split."""
    for root, n in ((DATA_ROOTS["SDSD_indoor"], n_in), (DATA_ROOTS["SDSD_outdoor"], n_out)):
        ds = SDSDDataset(root=root, subset="direct", split="train",
                         clip_len=n_frames, crop_size=256, is_train=False)
        seqs = sorted(ds.seqs, key=lambda s: s[0])
        pick = [seqs[int(round(i * (len(seqs) - 1) / max(1, n - 1)))] for i in range(n)]
        for seq_name, lq_paths, gt_paths in pick:
            Tn = min(len(lq_paths), len(gt_paths))
            start = max(0, (Tn - n_frames) // 2)
            end = min(Tn, start + n_frames)
            lqs = np.stack([center_crop(_load_img_any(p), crop) for p in lq_paths[start:end]])
            gts = np.stack([center_crop(_load_img_any(p), crop) for p in gt_paths[start:end]])
            yield seq_name, lqs, gts


def scene_key(name: str) -> str:
    """Group sequences of the same scene into one fold: BVI-RLV stores each scene at
    10 and 20 fps, SDSD stores a second pass of a scene as <name>_2."""
    for suf in ("__10fps", "__20fps"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name[:-2] if name.endswith("_2") else name


def moments(pred, cp, gt):
    """Per-channel sums of e1*e1, e1*e2, e2*e2 (e1 = cp - gt, e2 = pred - gt)."""
    e1 = (cp - gt).double().transpose(0, 1).reshape(3, -1)
    e2 = (pred - gt).double().transpose(0, 1).reshape(3, -1)
    return torch.stack([(e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)]).cpu()


def alpha_from(m):
    s11, s12, s22 = m[0], m[1], m[2]
    den = s11 - 2 * s12 + s22
    raw = torch.where(den.abs() > 1e-12, (s22 - s12) / den, torch.full_like(den, 0.5))
    return raw.clamp(0, 1).float(), raw.tolist()


@torch.no_grad()
def fit_alpha(model, dev):
    """Closed-form per-channel alpha* = (m22 - m12) / (m11 - 2 m12 + m22), pooled over pixels."""
    s11 = torch.zeros(3, dtype=torch.float64, device=dev)
    s12 = torch.zeros_like(s11)
    s22 = torch.zeros_like(s11)
    names = []
    for name, lq_u8, gt_u8 in iter_sdsd_train_fitset():
        lq = torch.from_numpy(lq_u8).float().div(255).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
        gt = torch.from_numpy(gt_u8).float().div(255).permute(0, 3, 1, 2).to(dev)
        lq_p, H, W = pad16(lq)
        pred = model(lq_p)[..., :H, :W][0].permute(1, 0, 2, 3)
        lq_t = lq[0].permute(1, 0, 2, 3)
        m = moments(pred, cp_surrogate(pred, lq_t), gt)
        s11 += m[0].to(dev); s12 += m[1].to(dev); s22 += m[2].to(dev)
        names.append(name)
    a, raw = alpha_from(torch.stack([s11, s12, s22]).cpu())
    return a, raw, names


def iter_sdsd_indoor_seer(n_frames: int = 16, crop: int = 512):
    """Seer's SDSD-indoor test split (15 seqs), same slicing as eval_lib."""
    ds = SDSDDataset(root=DATA_ROOTS["SDSD_indoor"], subset="direct", split="test",
                     clip_len=n_frames, crop_size=256, is_train=False)
    for seq_name, lq_paths, gt_paths in ds.seqs:
        Tn = min(len(lq_paths), len(gt_paths))
        start = max(0, (Tn - n_frames) // 2)
        end = min(Tn, start + n_frames)
        lqs = np.stack([center_crop(_load_img_any(p), crop) for p in lq_paths[start:end]])
        gts = np.stack([center_crop(_load_img_any(p), crop) for p in gt_paths[start:end]])
        yield seq_name, lqs, gts


SPLITS = {
    "L0_SDSD-indoor": ("video", lambda: iter_sdsd_indoor_seer()),
    "L1_SDSD-outdoor": ("video", lambda: iter_sdsd_outdoor(n_frames=16, crop=512)),
    "L2_BVI-RLV": ("video", lambda: iter_bvi(n_frames=16, crop=512)),
    "L3_LOL-v1": ("image", iter_lol_v1),
    "L4_LOLv2-Syn": ("image", lambda: iter_lol_v2("Syn")),
    "L5_LOLv2-Real": ("image", lambda: iter_lol_v2("Real")),
}


class Seer(torch.nn.Module):
    def __init__(self, ckpt: str, no_vjepa: bool, device):
        super().__init__()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        targs = state.get("args") or {}
        self.variant = {k: bool(targs.get(k, False)) for k in
                        ("no_vjepa", "random_vjepa", "disable_igab", "disable_lic", "disable_bottleneck")}
        self.variant["no_vjepa"] = self.variant["no_vjepa"] or no_vjepa
        print(f"[Seer] variant from ckpt args: {self.variant}")
        no_vjepa = self.variant["no_vjepa"]
        self.no_vjepa = no_vjepa
        self.device = device
        self.pb = VJepaPixelBranch(
            in_channels=3, vjepa_dim=1024, vjepa_grid=(CLIP_LEN // 2, 16, 16),
            num_vjepa_layers=4, cross_attn_at_scales=(1, 2, 3, 4),
            use_cfim_bottleneck=True, use_illumination_cues=True,
            use_gradient_checkpoint=False, router_init_c=4.0,
        )
        apply_pixel_branch_variant(self.pb, self.variant["disable_igab"],
                                   self.variant["disable_lic"], self.variant["disable_bottleneck"])
        miss, unex = self.pb.load_state_dict(state["pixel_branch"], strict=False)
        print(f"[Seer] pixel_branch missing={len(miss)} unexpected={len(unex)}")
        self.pb = self.pb.to(device).eval()
        self.bb = self.rf = None
        if not no_vjepa:
            rnd = self.variant["random_vjepa"]
            if rnd:
                torch.manual_seed(RANDOM_VJEPA_SEED)
            self.bb = VJepaBackboneLocal21(
                local_repo=T4.VJEPA_REPO, local_ckpt=("" if rnd else T4.VJEPA_CKPT),
                feature_layers=(5, 11, 17, 23), image_size=256,
                frames_per_clip=CLIP_LEN, patch_size=16, tubelet_size=2, freeze=True,
            )
            if rnd:
                fp = param_fingerprint(self.bb.model)
                assert abs(fp - targs["backbone_fp"]) < 1e-6 * max(1.0, abs(fp)), \
                    f"random backbone mismatch: {fp} vs {targs['backbone_fp']}"
                print(f"[Seer] random V-JEPA backbone rebuilt, fingerprint {fp:.6f} matches")
            self.bb = self.bb.to(device).eval()
            self.rf = CounterfactualRefiner(
                local_repo=T4.VJEPA_REPO, ckpt_path=T4.VJEPA_CKPT,
                encoder_embed_dim=1024, image_size=256, num_frames=CLIP_LEN,
                patch_size=16, tubelet_size=2, predictor_embed_dim=384,
                depth=12, num_heads=12, lora_rank=16, freeze_body=True,
            )
            miss, unex = self.rf.load_state_dict(state["cf_refiner"], strict=False)
            print(f"[Seer] cf_refiner missing={len(miss)} unexpected={len(unex)} "
                  f"gate={float(self.rf.gate):+.6f}")
            self.rf = self.rf.to(device).eval()

    @torch.no_grad()
    def forward(self, clip):  # (1,3,T,H,W) in [0,1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.no_vjepa:
                z_pixel = T4_zero_tokens(clip.shape[0], self.device)
            else:
                z = self.bb(clip)["feats"]
                z_pixel = list(z[:3]) + [self.rf(z[3].bfloat16())]
            out = self.pb(clip, z_pixel)["pred_pixels"]
        return out.float().clamp(0, 1)


def T4_zero_tokens(batch, device):
    z = torch.zeros(batch, (CLIP_LEN // 2) * 16 * 16, 1024, device=device)
    return [z] * 4


def pad16(x):
    H, W = x.shape[-2:]
    ph, pw = (-H) % 16, (-W) % 16
    if ph or pw:
        B, C, T, _, _ = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        y = F.pad(y, (0, pw, 0, ph), mode="reflect")
        x = y.reshape(B, T, C, H + ph, W + pw).permute(0, 2, 1, 3, 4)
    return x, H, W


def score_frames(pred, gt):
    """pred, gt: (T,3,H,W) float in [0,1] on GPU."""
    ps, lp, di, cq = [], [], [], []
    for t in range(pred.shape[0]):
        p = pred[t:t + 1]
        g = gt[t:t + 1]
        pu = (p[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gu = (g[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        ps.append(rgb_psnr_uint8(pu, gu))
        lp.append(lpips_batch(p, g))
        di.append(dists_batch(p, g))
        cq.append(clipiqa_batch(p))
    return dict(psnr_rgb=float(np.mean(ps)), lpips=float(np.mean(lp)),
                dists=float(np.mean(di)), clipiqa=float(np.mean(cq)), n_frames=len(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_vjepa", action="store_true")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = Seer(args.ckpt, args.no_vjepa, dev)
    alpha_fit, alpha_raw, fit_names = fit_alpha(model, dev)
    print(f"[CP] closed-form alpha on {len(fit_names)} SDSD train clips: raw={alpha_raw} "
          f"clipped={alpha_fit.tolist()}", flush=True)
    per_seq = {}
    summary = {}
    for split in args.splits:
        kind, it_fn = SPLITS[split]
        per_seq[split] = {"HMP-raw": {}, "HMP+CP": {}, "HMP+CPfit": {}, "HMP+CPxfit": {}}
        stash = {}   # name -> (pred u8, lq, gt, moments) kept on CPU for the cross-fit pass
        t0 = time.time()
        for i, (name, lq_u8, gt_u8) in enumerate(it_fn()):
            if args.limit is not None and i >= args.limit:
                break
            lq = torch.from_numpy(lq_u8).float().div(255).permute(3, 0, 1, 2).unsqueeze(0).to(dev)
            gt = torch.from_numpy(gt_u8).float().div(255).permute(0, 3, 1, 2).to(dev)
            if kind == "image":
                lq = lq.repeat(1, 1, CLIP_LEN, 1, 1)
            lq_p, H, W = pad16(lq)
            pred = model(lq_p)[..., :H, :W]                      # (1,3,T,H,W)
            if kind == "image":
                pred = pred[:, :, CLIP_LEN // 2:CLIP_LEN // 2 + 1]
                lq = lq[:, :, :1]
            pred_t = pred[0].permute(1, 0, 2, 3).contiguous()   # (T,3,H,W)
            lq_t = lq[0].permute(1, 0, 2, 3).contiguous()
            surr = cp_surrogate(pred_t, lq_t)
            cp = fuse_per_channel(pred_t, surr, ALPHA)
            cpf = fuse_per_channel(pred_t, surr, alpha_fit)
            per_seq[split]["HMP-raw"][name] = score_frames(pred_t, gt)
            per_seq[split]["HMP+CP"][name] = score_frames(cp, gt)
            per_seq[split]["HMP+CPfit"][name] = score_frames(cpf, gt)
            if kind == "video":
                flows = safe_gt_flows(gt)
                per_seq[split]["HMP-raw"][name].update(temporal_metrics(pred_t, lpips_batch, flows))
                per_seq[split]["HMP+CP"][name].update(temporal_metrics(cp, lpips_batch, flows))
                per_seq[split]["HMP+CPfit"][name].update(temporal_metrics(cpf, lpips_batch, flows))
                per_seq[split].setdefault("GT-temporal", {})[name] = temporal_metrics(gt, lpips_batch, flows)
            stash[name] = (pred_t.half().cpu(), lq_t.half().cpu(), gt.half().cpu(),
                           moments(pred_t, surr, gt))
        # ---- cross-fitted target-domain alpha (2 folds by sorted-name parity) ----
        names = sorted(stash)
        scenes = sorted({scene_key(n) for n in names})
        fold = {n: scenes.index(scene_key(n)) % 2 for n in names}
        xfit_alpha = {}
        for f in (0, 1):
            other = [n for n in names if fold[n] != f]
            if not other:
                continue
            a, raw = alpha_from(sum(stash[n][3] for n in other))
            xfit_alpha[f] = {"clipped": a.tolist(), "raw": raw, "fit_on": other}
            for n in names:
                if fold[n] != f:
                    continue
                pred_t, lq_t, gt = (x.float().to(dev) for x in stash[n][:3])
                cpx = fuse_per_channel(pred_t, cp_surrogate(pred_t, lq_t), a)
                per_seq[split]["HMP+CPxfit"][n] = score_frames(cpx, gt)
                if kind == "video":
                    per_seq[split]["HMP+CPxfit"][n].update(
                        temporal_metrics(cpx, lpips_batch, safe_gt_flows(gt)))
        per_seq[split]["_xfit_alpha"] = xfit_alpha
        del stash
        summary[split] = {}
        for var, d in per_seq[split].items():
            if not d or var.startswith("_"):
                continue
            keys = [k for k in ("psnr_rgb", "lpips", "dists", "clipiqa", "tlp", "mabd", "flicker", "warp_err")
                    if all(k in v for v in d.values())]
            summary[split][var] = {k: float(np.mean([v[k] for v in d.values()])) for k in keys}
            if var == "GT-temporal":
                summary[split][var]["n"] = len(d)
                continue
            summary[split][var]["n"] = len(d)
            if split.startswith("L0"):
                clean = [s for s in d if s in STCD_INDOOR_12]
                summary[split][var]["clean6_psnr_rgb"] = float(np.mean([d[s]["psnr_rgb"] for s in clean]))
                summary[split][var]["clean6_lpips"] = float(np.mean([d[s]["lpips"] for s in clean]))
                summary[split][var]["clean6_seqs"] = clean
            s = summary[split][var]
            tm = " ".join(f"{k}={s[k]:.4f}" for k in ("tlp", "mabd", "flicker", "warp_err") if k in s)
            print(f"{split:16s} {var:8s} PSNR={s['psnr_rgb']:.3f} LPIPS={s['lpips']:.4f} "
                  f"DISTS={s['dists']:.4f} CLIPIQA={s['clipiqa']:.4f} n={s['n']} {tm} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt": args.ckpt, "no_vjepa": model.no_vjepa, "variant": model.variant,
                       "cp_alpha_archived": ALPHA.tolist(),
                       "cp_alpha_fit": {"clipped": alpha_fit.tolist(), "raw": alpha_raw,
                                        "fit_clips": fit_names},
                       "protocol": "unified: video center T=16 512 crop; images full-res tiled T=16, "
                                   "middle frame; uint8 RGB PSNR; LPIPS/DISTS/CLIPIQA pyiqa",
                       "per_seq": per_seq, "summary": summary}, f, indent=1)
    print(f"[SAVE] {args.out}")


if __name__ == "__main__":
    main()
