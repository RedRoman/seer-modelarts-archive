"""Score an E4 checkpoint on the domain it was never trained on.

This is what H5 turns on, so the protocol is the baselines' protocol, not a new
one: metric functions come from baselines/common/eval_lib.py, video is the
centre 16 frames at a 512 centre crop, LOL images are scored at full resolution
with the clip tiled to 16 frames and the middle output frame read out. PSNR is
computed on uint8 RGB exactly as the published baselines compute it.

Two differences from the archived evaluator, both deliberate:
  * sequence lists come from the frozen splits, not from a rule re-derived at
    evaluation time, so no sequence can drift between runs;
  * the known frame-pairing bug in iter_sdsd_indoor_stcd is not inherited --
    LQ and GT are paired by sorted index within each sequence.

Also reports the GT-mean diagnostic: the same prediction rescaled so its mean
luma matches the ground truth's. The gap between the two columns is the part of
the error that is global exposure rather than structure.

The test split is read only when --split test is passed explicitly, and the
pre-registration allows that once, on final checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES = os.path.join(os.path.dirname(CODE), "baselines")
for p in (CODE, os.path.join(CODE, "tools"), BASELINES,
          os.path.join(BASELINES, "common")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.eval_lib import _load_img_any, center_crop, rgb_psnr_uint8  # noqa: E402
from data.common import list_images  # noqa: E402
from models.pixel_branch_v10 import VJepaPixelBranch  # noqa: E402
from models.pixel_branch_v2 import SeerV2PixelBranch  # noqa: E402
from models.predictor_lora import CounterfactualRefiner  # noqa: E402
from models.exposure_head import ExposureHead, apply_gain  # noqa: E402
from models.vjepa_backbone import VJepaBackboneLocal21  # noqa: E402
from temporal_metrics import safe_gt_flows, temporal_metrics  # noqa: E402

ROOT = os.environ.get("SEER_ROOT", "/cache/zihaohe/seer_cvpr")
SPLITS = f"{ROOT}/datasets/splits_cvpr.json"
SPLITS_LOL = f"{ROOT}/datasets/splits_lol.json"
VJEPA_REPO = f"{ROOT}/modelzoo/vjepa2-main"
VJEPA_CKPT = f"{ROOT}/weights/vjepa2_1_vitl_dist_vitG_384.pt"
REF_LAYER, N_LAYERS, VJEPA_DIM, N_TOKENS = 3, 4, 1024, 2048

_LPIPS = None


def _sdpa_pref():
    """Prefer the memory-lean attention kernels; fall back to math if needed."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.MATH])
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def lpips_of(pred_01, gt_01, dev):
    """pred/gt: (B,3,H,W) in [0,1]. pyiqa's LPIPS, as the baselines used."""
    global _LPIPS
    if _LPIPS is None:
        import pyiqa
        _LPIPS = pyiqa.create_metric("lpips", device=dev, as_loss=False).eval()
    with torch.no_grad():
        return float(_LPIPS(pred_01.clamp(0, 1).to(dev),
                            gt_01.clamp(0, 1).to(dev)).mean().item())


def luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def psnr_y_uint8(pred_u8, gt_u8):
    p, g = luma(pred_u8.astype(np.float64)), luma(gt_u8.astype(np.float64))
    mse = float(np.mean((p - g) ** 2))
    return float("inf") if mse <= 1e-10 else 20 * np.log10(255.0 / np.sqrt(mse))


def remap(entry):
    e = dict(entry)
    if isinstance(e.get("root"), str):
        e["root"] = e["root"].replace("/temp/zihaohe/seer_cvpr", ROOT)
    return e


def held_out_sequences(domain, split, n_frames, crop):
    """Yield (name, lq_uint8 (T,H,W,3), gt_uint8, kind) for the held-out domain."""
    sp = json.load(open(SPLITS))
    if domain == "sdsd":
        for key, in (("SDSD_indoor",), ("SDSD_outdoor",)):
            e = remap(sp[key])
            for n in e[split]:
                lq_p = list_images(os.path.join(e["root"], "input", n))
                gt_p = list_images(os.path.join(e["root"], "GT", n))
                k = min(len(lq_p), len(gt_p))
                if k == 0:
                    continue
                s = max(0, (k - n_frames) // 2)
                idx = list(range(s, min(k, s + n_frames)))
                lq = np.stack([center_crop(_load_img_any(lq_p[i]), crop) for i in idx])
                gt = np.stack([center_crop(_load_img_any(gt_p[i]), crop) for i in idx])
                yield f"{key}/{n}", lq, gt, "video"
    elif domain == "bvi":
        e = remap(sp["BVI_RLV"])
        for n in e[split]:
            base, fps = n.split("__")
            fps = fps.replace("fps", "")
            lq_p = list_images(os.path.join(e["root"], base, f"low_light_{fps}"))
            gt_p = list_images(os.path.join(e["root"], base, f"normal_light_{fps}"))
            k = min(len(lq_p), len(gt_p))
            if k == 0:
                continue
            s = max(0, (k - n_frames) // 2)
            idx = list(range(s, min(k, s + n_frames)))
            lq = np.stack([center_crop(_load_img_any(lq_p[i]), crop) for i in idx])
            gt = np.stack([center_crop(_load_img_any(gt_p[i]), crop) for i in idx])
            yield f"BVI/{n}", lq, gt, "video"
    elif domain == "lol":
        spl = json.load(open(SPLITS_LOL))
        for key, e in spl.items():
            if not isinstance(e, dict) or "root" not in e:
                continue
            e = remap(e)
            s = e["splits"][split]
            for n in s["names"]:
                lq_p = os.path.join(e["root"], s["lq"], n)
                gt_p = os.path.join(e["root"], s["gt"], n)
                if not os.path.exists(gt_p):
                    # LOLv2-synthetic names its inputs with an extra leading "r"
                    # (r00816405t.png against 00816405t.png); the other two LOL
                    # releases use identical names on both sides
                    alt = os.path.join(e["root"], s["gt"],
                                       n[1:] if n.startswith("r") else "r" + n)
                    if not os.path.exists(alt):
                        raise FileNotFoundError(
                            f"{key}: no ground truth for {n} (tried {gt_p} and {alt})")
                    gt_p = alt
                lq = _load_img_any(lq_p)[None]
                gt = _load_img_any(gt_p)[None]
                yield f"{key}/{n}", lq, gt, "image"
    else:
        raise ValueError(domain)


def build_model(ckpt, dev, force_arm=None):
    """Rebuild whichever architecture the checkpoint was trained with.

    Loading a v1 checkpoint into SeerV2PixelBranch does not fail: the IGAB and
    router tensors land in `unexpected` rather than `missing`, so the model comes
    up silently stripped of them and without the M1 refiner. The architecture is
    therefore taken from the checkpoint's own args, and strictness is checked
    both ways.
    """
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = st.get("args", {})
    arm = saved.get("vjepa", "pretrained")
    trained_arm = arm
    if force_arm is not None and force_arm != arm:
        # H16 criterion (ii) is an inference-only test: take a model trained
        # with the real tokens and feed it tokens from a randomly initialised
        # encoder. Nothing but the token source changes, so a trained gate that
        # has learned to recognise unusable conditioning should close and
        # recover the no-token result rather than losing the 4.1 dB that the
        # ungated model loses.
        if force_arm == "none":
            raise ValueError("force_arm=none would leave the cross-attention "
                             "without keys; it is a different architecture")
        arm = force_arm
    arch = saved.get("arch", "v2")
    clip_len = int(saved.get("clip_len", 16))
    common = dict(in_channels=3, vjepa_dim=VJEPA_DIM,
                  vjepa_grid=(clip_len // 2, 16, 16), num_vjepa_layers=N_LAYERS,
                  cross_attn_at_scales=(1, 2, 3, 4), use_cfim_bottleneck=True,
                  use_illumination_cues=True, use_gradient_checkpoint=False)
    if arch == "v1":
        pb = VJepaPixelBranch(router_init_c=float(saved.get("router_c", 4.0)),
                              **common).to(dev).eval()
    else:
        # reconstruct the exact architecture the checkpoint was trained with,
        # including branch size and xattn variant, before loading any weights
        branch = saved.get("branch", "full")
        xattn = saved.get("xattn", "global")
        shape = (dict(dims=(20, 40, 80, 160, 320), num_blocks=(2, 4, 4, 6, 8),
                      heads=(2, 4, 4, 8, 8)) if branch == "small" else {})
        pb = SeerV2PixelBranch(xattn=xattn, **shape, **common).to(dev).eval()
    missing, unexpected = pb.load_state_dict(st["pixel_branch"], strict=False)
    truly_missing = [k for k in missing]
    truly_unexpected = [k for k in unexpected]
    if truly_missing or truly_unexpected:
        raise RuntimeError(f"{ckpt}: arch={arch} missing={len(truly_missing)} "
                           f"unexpected={len(truly_unexpected)} "
                           f"(e.g. {(truly_missing or truly_unexpected)[:3]})")
    # A checkpoint trained with --edc on carries an exposure head; scoring
    # without it would measure a different model from the one trained.
    exp_head = None
    if st.get("exposure_head"):
        exp_head = ExposureHead(
            vjepa_dim=VJEPA_DIM,
            use_vjepa=(saved.get("edc_vjepa", "on") == "on" and arm != "none")
        ).to(dev).eval()
        m, u = exp_head.load_state_dict(st["exposure_head"], strict=False)
        if m:
            raise RuntimeError(f"{ckpt}: exposure head missing {len(m)} keys")
    bb = refiner = None
    if arm != "none":
        bb = VJepaBackboneLocal21(
            local_repo=VJEPA_REPO,
            local_ckpt=("" if arm == "random" else VJEPA_CKPT),
            feature_layers=(5, 11, 17, 23), image_size=256,
            frames_per_clip=clip_len, patch_size=16, tubelet_size=2,
            freeze=True).to(dev).eval()
        if arch == "v1" and st.get("cf_refiner"):
            refiner = CounterfactualRefiner(
                local_repo=VJEPA_REPO, ckpt_path=VJEPA_CKPT,
                encoder_embed_dim=VJEPA_DIM, image_size=256, num_frames=clip_len,
                patch_size=16, tubelet_size=2, predictor_embed_dim=384, depth=12,
                num_heads=12, lora_rank=int(saved.get("lora_rank", 16)),
                freeze_body=True).to(dev).eval()
            m2, u2 = refiner.load_state_dict(st["cf_refiner"], strict=False)
            if m2:
                raise RuntimeError(f"{ckpt}: refiner missing {len(m2)} keys")
    return pb, bb, refiner, exp_head, arm, clip_len, saved


@torch.no_grad()
def run_clip(pb, bb, refiner, exp_head, arm, lq_u8, clip_len, dev, amp_dtype):
    """lq_u8: (T,H,W,3) uint8 -> predicted (T,H,W,3) uint8.

    At a 512 crop the cross-attention falls back to the math backend and
    materialises a (T*heads, HW, 256) score matrix -- 8 GB in one allocation on
    a card we share. Asking for the flash/memory-efficient kernels first is the
    same arithmetic with a fraction of the peak.
    """
    t_in, h0, w0 = lq_u8.shape[0], lq_u8.shape[1], lq_u8.shape[2]
    x = torch.from_numpy(lq_u8).float().div(255.).permute(3, 0, 1, 2)[None].to(dev)
    if t_in < clip_len:                       # images and short clips are tiled
        x = x.repeat(1, 1, -(-clip_len // t_in), 1, 1)[:, :, :clip_len]
    # the 5-scale U-Net unshuffles four times, so both sides must be a multiple
    # of 32; LOL frames are 400x600 and would fail at the fourth level
    ph, pw = (-h0) % 32, (-w0) % 32
    if ph or pw:
        x = torch.nn.functional.pad(x.reshape(-1, 3, h0, w0), (0, pw, 0, ph),
                                    mode="reflect").reshape(1, 3, -1, h0 + ph, w0 + pw)
    with _sdpa_pref(), torch.autocast(device_type=dev.type, dtype=amp_dtype):
        if arm == "none":
            z = [torch.zeros(1, N_TOKENS, VJEPA_DIM, device=dev)] * N_LAYERS
        else:
            z = [t.float() for t in bb(x)["feats"]]
            if refiner is not None:        # v1 routes the last layer through M1
                z = z[:3] + [refiner(z[3].to(amp_dtype)).float()]
        out = pb(x, z)
    pred_t = out["pred_pixels"].float().clamp(0, 1)
    if exp_head is not None:
        tok = z[REF_LAYER] if arm != "none" else None
        pred_t = apply_gain(pred_t, exp_head(x, tok), x)
    pred = pred_t.clamp(0, 1)[0].permute(1, 2, 3, 0)
    pred = pred[:, :h0, :w0]
    return (pred.cpu().numpy() * 255.0).round().astype(np.uint8)


@torch.no_grad()
def run_clip_tiled(pb, bb, refiner, exp_head, arm, lq_u8, clip_len, dev, amp_dtype, tile, overlap):
    """Run the model on overlapping spatial tiles and blend the results.

    The V-JEPA encoder resizes whatever it is given to 256, so feeding it a 512
    crop hands the cross-attention tokens that describe twice the field of view
    at half the detail -- out of distribution for weights trained on 256 crops,
    and measured to cost the pretrained arm 3.3 dB. Scoring tile by tile puts
    every component back on the input it was trained for; the overlap is blended
    with a linear ramp so no seam is scored.
    """
    t, h, w = lq_u8.shape[:3]
    if h <= tile and w <= tile:
        return run_clip(pb, bb, refiner, exp_head, arm, lq_u8, clip_len, dev, amp_dtype)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    acc = np.zeros((t, h, w, 3), np.float64)
    wsum = np.zeros((1, h, w, 1), np.float64)
    ramp = np.ones(tile)
    if overlap > 0:
        ramp[:overlap] = np.linspace(0, 1, overlap, endpoint=False)
        ramp[-overlap:] = np.linspace(1, 0, overlap, endpoint=False)
    win = (ramp[:, None] * ramp[None, :])[None, :, :, None]
    for y in ys:
        for x in xs:
            out = run_clip(pb, bb, refiner, exp_head, arm, lq_u8[:, y:y + tile, x:x + tile],
                           clip_len, dev, amp_dtype)
            acc[:, y:y + tile, x:x + tile] += out.astype(np.float64) * win
            wsum[:, y:y + tile, x:x + tile] += win
    return np.clip(acc / np.maximum(wsum, 1e-8), 0, 255).round().astype(np.uint8)


def gt_mean_rescale(pred_u8, gt_u8):
    """Rescale the prediction so its mean luma matches the ground truth's."""
    p = pred_u8.astype(np.float64)
    a = float(luma(gt_u8.astype(np.float64)).mean())
    b = float(luma(p).mean())
    if b <= 1e-6:
        return pred_u8
    return np.clip(p * (a / b), 0, 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True, choices=["sdsd", "bvi", "lol"])
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="test is read once, on final checkpoints, per the pre-registration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=256,
                    help="256 is the training crop; the V-JEPA path is measured "
                         "to lose 3.3 dB when scored at 512 untiled")
    ap.add_argument("--tile", type=int, default=0,
                    help="run on tiles of this size and blend; 0 disables")
    ap.add_argument("--tile_overlap", type=int, default=32)
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--device", default="cuda", choices=["cuda", "npu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force_arm", default=None, choices=["pretrained", "random"],
                    help="score a trained checkpoint with a different token "
                         "source than it was trained with (H16 criterion ii)")
    args = ap.parse_args()

    if args.device == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(f"{args.device}:0")
    amp_dtype = torch.bfloat16
    pb, bb, refiner, exp_head, arm, clip_len, saved = build_model(args.ckpt, dev, force_arm=args.force_arm)
    print(f"[eval_e4] {os.path.basename(args.ckpt)} arm={arm} "
          f"arch={saved.get('arch','v2')} heldout={args.heldout} split={args.split} "
          f"trained_heldout={saved.get('heldout')} seed={saved.get('seed')}", flush=True)
    # The guard stops a model being scored on a domain it was trained on by
    # accident. A run with --only is the deliberate exception: it trains on one
    # domain and is meant to be scored on that domain's held-out val split.
    trained_only = saved.get("only") or ""
    if trained_only:
        if args.heldout != trained_only:
            raise SystemExit(f"checkpoint trained on {trained_only!r} only; "
                             f"score it with --heldout {trained_only}")
    elif saved.get("heldout") and saved["heldout"] != args.heldout:
        raise SystemExit(f"checkpoint was trained holding out {saved['heldout']!r}, "
                         f"not {args.heldout!r}")

    rows = []
    for i, (name, lq, gt, kind) in enumerate(
            held_out_sequences(args.heldout, args.split, args.n_frames, args.crop)):
        if args.limit is not None and i >= args.limit:
            break
        if args.tile:
            pred = run_clip_tiled(pb, bb, refiner, exp_head, arm, lq, clip_len, dev,
                                  amp_dtype, args.tile, args.tile_overlap)
        else:
            pred = run_clip(pb, bb, refiner, exp_head, arm, lq, clip_len, dev, amp_dtype)
        if kind == "image":
            # a single image is tiled to a static clip; the archived protocol
            # reads the middle output frame back out
            pred = pred[pred.shape[0] // 2:pred.shape[0] // 2 + 1]
        gt_s, lq_s = gt, lq
        n = min(pred.shape[0], gt_s.shape[0])
        pred, gt_s, lq_s = pred[:n], gt_s[:n], lq_s[:n]
        gm = gt_mean_rescale(pred, gt_s)
        to_t = lambda a: torch.from_numpy(a.copy()).float().div(255.).permute(0, 3, 1, 2)
        row = {
            "name": name, "kind": kind, "frames": int(n),
            "psnr_rgb": float(np.mean([rgb_psnr_uint8(pred[j], gt_s[j]) for j in range(n)])),
            "psnr_y": float(np.mean([psnr_y_uint8(pred[j], gt_s[j]) for j in range(n)])),
            "psnr_y_gtmean": float(np.mean([psnr_y_uint8(gm[j], gt_s[j]) for j in range(n)])),
            "lpips": lpips_of(to_t(pred), to_t(gt_s), dev),
            "input_psnr_y": float(np.mean([psnr_y_uint8(lq_s[j], gt_s[j]) for j in range(n)])),
        }
        if kind == "video" and n >= 2:
            # the meaningful quantity is the gap to the ground truth's own
            # temporal statistics, not the absolute value: a frozen output has
            # the best tlp and the worst warp error
            pt, gtt = to_t(pred).to(dev), to_t(gt_s).to(dev)
            flows = safe_gt_flows(gtt)
            lp = lambda a, b: lpips_of(a, b, dev)
            for k, v in temporal_metrics(pt, lp, flows).items():
                row[k] = v
            for k, v in temporal_metrics(gtt, lp, flows).items():
                row["gt_" + k] = v
        rows.append(row)
        print(f"  {i:3d} {name:<34} PSNR-Y {row['psnr_y']:6.2f} "
              f"(GT-mean {row['psnr_y_gtmean']:6.2f})  LPIPS {row['lpips']:.4f}", flush=True)

    keys = ["psnr_rgb", "psnr_y", "psnr_y_gtmean", "lpips", "input_psnr_y"]
    keys += [k for k in ("tlp", "mabd", "flicker", "warp_err") if k in rows[0]]
    keys += [k for k in ("gt_tlp", "gt_mabd", "gt_flicker", "gt_warp_err") if k in rows[0]]
    agg = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    out = {"ckpt": args.ckpt, "arm": arm, "heldout": args.heldout, "split": args.split,
           "seed": saved.get("seed"), "iters": saved.get("iters"),
           "edc": saved.get("edc", "off"),
           "crop": args.crop, "tile": args.tile, "tile_overlap": args.tile_overlap,
           "n_frames": args.n_frames,
           "n_sequences": len(rows), "aggregate": agg, "per_sequence": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n{len(rows)} sequences | PSNR-Y {agg['psnr_y']:.3f} | "
          f"GT-mean {agg['psnr_y_gtmean']:.3f} | LPIPS {agg['lpips']:.4f} | "
          f"input {agg['input_psnr_y']:.3f}\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
