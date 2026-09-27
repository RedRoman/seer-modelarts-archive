"""Visualize trained model outputs on validation set.

Usage:
    python tools/eval_visual.py --version v7 \
        --ckpt /cache/zihaohe/checkpoint/runs/v7/ckpts/best.pt \
        --out  /cache/zihaohe/visualizations/v7_best

For each val sequence:
    1. Run model.sample(LQ) → pred
    2. Compute PSNR/SSIM
    3. Save:
       - <name>_pred.mp4  (just prediction)
       - <name>_compare.mp4 (LQ | pred | HQ side-by-side, with text annotations)
       - <name>_frames/   (per-frame PNG of compare layout, every 4 frames)
    4. Write summary.md with per-sequence metrics + global mean
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from data import build_dataset
from utils import dump_config, ensure_dir, load_config, psnr, ssim


# ----------------------------------------------------------------------
# Model factory: load v6 / v7 / v8
# ----------------------------------------------------------------------
def build_model_for_version(version: str, cfg: Dict, device):
    if version == "v6":
        from models.vjepa_llve_v6 import VJepaLLVE_v6 as M
    elif version == "v6d":
        from models.vjepa_llve_v6d import VJepaLLVE_v6D as M
    elif version == "v6e":
        from models.vjepa_llve_v6e import VJepaLLVE_v6E as M
    elif version == "v7":
        from models.vjepa_llve_v7 import VJepaLLVE_v7 as M
    elif version == "v8":
        from models.vjepa_llve_v8 import VJepaLLVE_v8 as M
    else:
        raise ValueError(f"unknown version {version}")
    model = M(cfg["model"]).to(device)
    model.freeze_backbone()
    model.eval()
    return model


def load_state_for_version(version: str, ckpt_path: str, model):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_loaded = 0
    if "latent_dit" in state and hasattr(model, "latent_dit"):
        model.latent_dit.load_state_dict(state["latent_dit"], strict=False); n_loaded += 1
    if "pixel_decoder" in state and hasattr(model, "pixel_decoder"):
        model.pixel_decoder.load_state_dict(state["pixel_decoder"], strict=False); n_loaded += 1
    if "pixel_branch" in state and hasattr(model, "pixel_branch"):
        model.pixel_branch.load_state_dict(state["pixel_branch"], strict=False); n_loaded += 1
    if "predictor_refiner" in state and hasattr(model, "predictor_refiner"):
        model.predictor_refiner.load_state_dict(state["predictor_refiner"], strict=False); n_loaded += 1
    if "illumination_router" in state and hasattr(model, "illumination_router"):
        model.illumination_router.load_state_dict(state["illumination_router"], strict=False); n_loaded += 1
    print(f"[load] {ckpt_path} → loaded {n_loaded} components | step={state.get('step', '?')} "
          f"best={state.get('best_metric', '?')} best_ema={state.get('best_ema_metric', '?')}")


# ----------------------------------------------------------------------
# Tensor → uint8 frame helpers
# ----------------------------------------------------------------------
def _tensor_to_uint8_frames(x: torch.Tensor) -> np.ndarray:
    """(B=1, 3, T, H, W) [0,1] → (T, H, W, 3) uint8 RGB."""
    assert x.shape[0] == 1
    x = x[0].clamp(0, 1).cpu().numpy()        # (3, T, H, W)
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 3, 0))         # (T, H, W, 3)
    return x


def _annotate_frame(img_rgb: np.ndarray, label: str) -> np.ndarray:
    """Add a text label to top-left of frame."""
    img = img_rgb.copy()
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 1, cv2.LINE_AA)
    return img


def write_mp4(path: str, frames_rgb: np.ndarray, fps: int = 8) -> None:
    """frames_rgb: (T, H, W, 3) uint8 → write H.264 mp4 via ffmpeg pipe.

    Uses ffmpeg subprocess for H.264 yuv420p encoding (browser/QuickTime
    compatible). cv2's mp4v fourcc produces files that fail to load in many
    web/native players.
    """
    import subprocess
    T, H, W, _ = frames_rgb.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-crf", "18",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for t in range(T):
        proc.stdin.write(frames_rgb[t].tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="ignore")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {err}")


def stack_horizontal(*frame_lists: List[np.ndarray]) -> np.ndarray:
    """Take 1+ (T, H, W, 3) arrays → (T, H, sum_W, 3) horizontal concat."""
    return np.concatenate(frame_lists, axis=2)


# ----------------------------------------------------------------------
# Per-sequence eval
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_one(model, batch, device, fps: int = 8) -> Dict:
    lq = batch["lq"].to(device, non_blocking=True)
    hq = batch["hq"].to(device, non_blocking=True)
    out = model.sample(lq)
    pred = out["pred_pixels"].clamp(0, 1)
    p = psnr(pred, hq).item()
    s = ssim(pred, hq).item()

    # Convert to uint8 frames
    lq_frames = _tensor_to_uint8_frames(lq)
    pred_frames = _tensor_to_uint8_frames(pred)
    hq_frames = _tensor_to_uint8_frames(hq)

    # Annotate
    lq_anno = np.stack([_annotate_frame(lq_frames[t], "LQ") for t in range(lq_frames.shape[0])])
    pred_anno = np.stack([_annotate_frame(pred_frames[t], f"Pred {p:.2f}dB") for t in range(pred_frames.shape[0])])
    hq_anno = np.stack([_annotate_frame(hq_frames[t], "HQ") for t in range(hq_frames.shape[0])])

    compare = stack_horizontal(lq_anno, pred_anno, hq_anno)

    return {
        "psnr": p, "ssim": s,
        "lq_frames": lq_frames, "pred_frames": pred_frames, "hq_frames": hq_frames,
        "compare_frames": compare,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v6", "v6d", "v6e", "v7", "v8"])
    ap.add_argument("--config", default=None,
                    help="Path to model config (yaml). If omitted, uses configs/train_<version>.yaml")
    ap.add_argument("--ckpt", required=True, help="Path to ckpt (.pt) to load")
    ap.add_argument("--out", required=True, help="Output dir for videos and report")
    ap.add_argument("--max_sequences", type=int, default=14, help="Max val sequences to process")
    ap.add_argument("--fps", type=int, default=8, help="Output mp4 fps")
    ap.add_argument("--save_per_seq_dirs", action="store_true",
                    help="Also dump per-sequence frame PNGs (slow)")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(_REPO_ROOT, f"configs/train_{args.version}.yaml")
    cfg = load_config(cfg_path, overrides=[])

    out_dir = ensure_dir(args.out)
    print(f"[eval_visual] version={args.version} ckpt={args.ckpt} → {out_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[eval_visual] device={device}")

    # Build val dataset (same as training)
    tcfg = cfg["train"]
    val_set = build_dataset(
        cfg["data"]["val"],
        clip_len=tcfg.get("clip_len", 16),
        crop_size=tcfg.get("crop_size", 256), is_train=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    print(f"[eval_visual] val: {len(val_set)} sequences")

    # Build + load model
    model = build_model_for_version(args.version, cfg, device)
    load_state_for_version(args.version, args.ckpt, model)
    model.eval()

    # Eval loop
    results = []
    for i, batch in enumerate(val_loader):
        if i >= args.max_sequences:
            break
        name = batch.get("name", [f"seq{i:03d}"])[0] if isinstance(batch.get("name"), list) else f"seq{i:03d}"
        # safe name
        sname = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:80]
        print(f"  [{i+1}/{min(len(val_set), args.max_sequences)}] {sname}", flush=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            res = eval_one(model, batch, device, fps=args.fps)
        psnr_v, ssim_v = res["psnr"], res["ssim"]
        print(f"      PSNR {psnr_v:.3f}  SSIM {ssim_v:.4f}")

        # Write videos
        write_mp4(os.path.join(out_dir, f"{i:02d}_{sname}_pred.mp4"), res["pred_frames"], fps=args.fps)
        write_mp4(os.path.join(out_dir, f"{i:02d}_{sname}_compare.mp4"), res["compare_frames"], fps=args.fps)

        # Save first frame as PNG (compare)
        first = res["compare_frames"][0]
        cv2.imwrite(
            os.path.join(out_dir, f"{i:02d}_{sname}_frame00.png"),
            cv2.cvtColor(first, cv2.COLOR_RGB2BGR),
        )

        if args.save_per_seq_dirs:
            sub = ensure_dir(os.path.join(out_dir, f"{i:02d}_{sname}_frames"))
            T = res["compare_frames"].shape[0]
            for t in range(0, T, 4):
                cv2.imwrite(
                    os.path.join(sub, f"frame_{t:03d}.png"),
                    cv2.cvtColor(res["compare_frames"][t], cv2.COLOR_RGB2BGR),
                )

        results.append({"idx": i, "name": sname, "psnr": psnr_v, "ssim": ssim_v})

    # Summary
    if results:
        mean_p = sum(r["psnr"] for r in results) / len(results)
        mean_s = sum(r["ssim"] for r in results) / len(results)
        with open(os.path.join(out_dir, "summary.md"), "w") as f:
            f.write(f"# Eval Visual Report — {args.version}\n\n")
            f.write(f"- ckpt: `{args.ckpt}`\n")
            f.write(f"- val sequences: {len(results)}\n")
            f.write(f"- **mean PSNR: {mean_p:.3f} dB**\n")
            f.write(f"- **mean SSIM: {mean_s:.4f}**\n\n")
            f.write(f"| # | name | PSNR | SSIM |\n")
            f.write(f"|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['idx']} | {r['name']} | {r['psnr']:.3f} | {r['ssim']:.4f} |\n")
        print(f"[eval_visual] mean PSNR={mean_p:.3f} SSIM={mean_s:.4f}")
        print(f"[eval_visual] summary written to {out_dir}/summary.md")


if __name__ == "__main__":
    main()
