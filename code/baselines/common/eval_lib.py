"""Shared eval harness for baselines.

Protocol (matches Seer as specified by task brief):
  - Videos (SDSD-indoor, SDSD-outdoor, BVI-RLV): first 16 frames per seq,
    center-crop 512x512, per-frame PSNR/LPIPS, mean over T then seqs.
  - Images (LOL-v1, LOLv2-Real, LOLv2-Syn): full image, per-image PSNR/LPIPS,
    mean over images.
  - PSNR: computed on uint8 [0,255] RGB (float MSE, dB = 20 log10(255/sqrt(mse))).
    This matches LLVE-STCD / most LLIE papers.
  - LPIPS: pyiqa 'lpips' (default alex, ImageNet-normalized), inputs in [0,1].
Datasets:
  - SDSD-indoor 12-seq STCD test split.
  - SDSD-outdoor 25-seq (from SDSDDataset(subset='direct', split='test')).
  - BVI-RLV 16-seq (from BVIRLVDataset(split='test'), 8 scenes x 2 fps).
  - LOL-v1 15 images (/cache/zihaohe/datasets/LOL/LOLv1_test/test/{low,high}).
  - LOLv2-Real 100 images (okhater_lolv2-real/Test/{Input,GT}).
  - LOLv2-Syn 100 images (okhater_lolv2-synthetic/Test/{Input,GT}).
"""
from __future__ import annotations
import os, sys, json, time
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
import torch
import imageio.v2 as imageio

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_VJ_REPO = "/home/ma-user/workspace/zihaohe/vjepa_llve"
if _VJ_REPO not in sys.path:
    sys.path.insert(0, _VJ_REPO)


# ---------------- constants ----------------
STCD_INDOOR_12 = [
    "pair13", "pair15", "pair21", "pair23", "pair31", "pair33",
    "pair50", "pair52", "pair58", "pair60", "pair68", "pair70",
]

DATA_ROOTS = {
    "SDSD_indoor":  "/cache/zihaohe/datasets/SDSD_indoor",
    "SDSD_outdoor": "/cache/zihaohe/datasets/SDSD_outdoor",
    "BVI_RLV":      "/cache/zihaohe/datasets/BVI_RLV/extracted/BVI-Lowlight-videos",
    "LOL_v1":       "/cache/zihaohe/datasets/LOL/LOLv1_test/test",
    "LOLv2_Real":   "/cache/zihaohe/datasets/LOL/okhater_lolv2-real/Test",
    "LOLv2_Syn":    "/cache/zihaohe/datasets/LOL/okhater_lolv2-synthetic/Test",
}


# ---------------- metrics ----------------
_LPIPS_MODEL = None


def get_lpips_model(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import pyiqa
        _LPIPS_MODEL = pyiqa.create_metric("lpips", device=device, as_loss=False).eval()
    return _LPIPS_MODEL


def rgb_psnr_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    """(H, W, 3) uint8 → PSNR dB. Matches LLVE-STCD."""
    p = pred_u8.astype(np.float64); g = gt_u8.astype(np.float64)
    mse = np.mean((p - g) ** 2)
    if mse <= 1e-10:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


@torch.no_grad()
def lpips_batch(pred_01: torch.Tensor, gt_01: torch.Tensor) -> float:
    """pred/gt: (B, 3, H, W) float [0,1] on any device. Returns mean LPIPS."""
    dev = pred_01.device
    if not pred_01.is_cuda:
        dev = torch.device("cuda")
        pred_01 = pred_01.to(dev); gt_01 = gt_01.to(dev)
    m = get_lpips_model(dev)
    v = m(pred_01.clamp(0, 1), gt_01.clamp(0, 1))
    return float(v.mean().item())


# ---------------- IO ----------------
def _load_img_any(path: str) -> np.ndarray:
    """Return HWC uint8 [0,255] RGB."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr
    im = imageio.imread(path)
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    if im.shape[-1] == 4:
        im = im[..., :3]
    return im


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """(H, W, 3) → (size, size, 3) center crop. If image is smaller, pad-reflect."""
    H, W = arr.shape[:2]
    if H < size or W < size:
        pad_h = max(0, size - H); pad_w = max(0, size - W)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        H, W = arr.shape[:2]
    y0 = (H - size) // 2; x0 = (W - size) // 2
    return arr[y0:y0+size, x0:x0+size]


def _list_sorted(dp: str) -> List[str]:
    return sorted(os.path.join(dp, f) for f in os.listdir(dp)
                  if not f.startswith("."))


# ---------------- Dataset iterators ----------------
def iter_sdsd_indoor_stcd(n_frames: int = 16, crop: int = 512):
    """Yields (seq_name, lq_uint8[T,H,W,3], gt_uint8[T,H,W,3]). Center-cropped."""
    root = DATA_ROOTS["SDSD_indoor"]
    lq_root = os.path.join(root, "indoor_static_np", "input")
    gt_root = os.path.join(root, "indoor_static_np", "GT")
    for seq in STCD_INDOOR_12:
        lq_dir = os.path.join(lq_root, seq); gt_dir = os.path.join(gt_root, seq)
        if not (os.path.isdir(lq_dir) and os.path.isdir(gt_dir)):
            print(f"[SKIP] {seq}: dir missing"); continue
        lq_files = _list_sorted(lq_dir); gt_files = _list_sorted(gt_dir)
        by_lq = {os.path.basename(p): p for p in lq_files}
        by_gt = {os.path.basename(p): p for p in gt_files}
        common = sorted(set(by_lq) & set(by_gt))
        # center T=n_frames temporal slice
        Tn = len(common)
        if Tn == 0: continue
        start = max(0, (Tn - n_frames) // 2); end = start + n_frames
        pick = common[start:end][:n_frames]
        lqs = np.stack([_load_img_any(by_lq[n]) for n in pick], axis=0)
        gts = np.stack([_load_img_any(by_gt[n]) for n in pick], axis=0)
        lqs = np.stack([center_crop(f, crop) for f in lqs], axis=0)
        gts = np.stack([center_crop(f, crop) for f in gts], axis=0)
        yield seq, lqs, gts


def iter_sdsd_outdoor(n_frames: int = 16, crop: int = 512):
    """Uses V-JEPA-LLVE 80/20 split (25 seqs). SDSDDataset pairs LQ/GT by
    sorted index (not basename), because SDSD-outdoor "_2" variants have
    intentionally offset frame indices between LQ and GT."""
    from data.sdsd_dataset import SDSDDataset
    ds = SDSDDataset(root=DATA_ROOTS["SDSD_outdoor"], subset="direct",
                     split="test", clip_len=n_frames, crop_size=256, is_train=False)
    for seq_name, lq_paths, gt_paths in ds.seqs:
        Tn = min(len(lq_paths), len(gt_paths))
        if Tn == 0: continue
        start = max(0, (Tn - n_frames) // 2); end = min(Tn, start + n_frames)
        lq_slice = lq_paths[start:end]; gt_slice = gt_paths[start:end]
        lqs = np.stack([_load_img_any(p) for p in lq_slice], axis=0)
        gts = np.stack([_load_img_any(p) for p in gt_slice], axis=0)
        lqs = np.stack([center_crop(f, crop) for f in lqs], axis=0)
        gts = np.stack([center_crop(f, crop) for f in gts], axis=0)
        yield seq_name, lqs, gts


def iter_bvi(n_frames: int = 16, crop: int = 512):
    from data.bvi_rlv_dataset import BVIRLVDataset
    ds = BVIRLVDataset(root=DATA_ROOTS["BVI_RLV"], split="test",
                       clip_len=n_frames, crop_size=256, is_train=False)
    for seq_name, lq_paths, gt_paths in ds.seqs:
        # BVI has many frames; take center 16
        Tn = len(lq_paths)
        start = max(0, (Tn - n_frames) // 2); end = start + n_frames
        lqs = np.stack([_load_img_any(lq_paths[i]) for i in range(start, min(end, Tn))], axis=0)
        gts = np.stack([_load_img_any(gt_paths[i]) for i in range(start, min(end, Tn))], axis=0)
        lqs = np.stack([center_crop(f, crop) for f in lqs], axis=0)
        gts = np.stack([center_crop(f, crop) for f in gts], axis=0)
        yield seq_name, lqs, gts


def iter_lol_v1():
    root = DATA_ROOTS["LOL_v1"]
    lq_dir = os.path.join(root, "low"); gt_dir = os.path.join(root, "high")
    for fn in sorted(os.listdir(lq_dir)):
        if fn.startswith("."): continue
        lq_p = os.path.join(lq_dir, fn); gt_p = os.path.join(gt_dir, fn)
        if not (os.path.isfile(lq_p) and os.path.isfile(gt_p)): continue
        yield os.path.splitext(fn)[0], _load_img_any(lq_p)[None], _load_img_any(gt_p)[None]


def iter_lol_v2(kind="Real"):
    key = "LOLv2_Real" if kind.lower().startswith("real") else "LOLv2_Syn"
    root = DATA_ROOTS[key]
    lq_dir = os.path.join(root, "Input"); gt_dir = os.path.join(root, "GT")
    # LOL-v2 file names may differ between Input and GT — pair by sorted order
    lq_files = sorted(os.listdir(lq_dir)); gt_files = sorted(os.listdir(gt_dir))
    for lf, gf in zip(lq_files, gt_files):
        if lf.startswith("."): continue
        yield os.path.splitext(lf)[0], _load_img_any(os.path.join(lq_dir, lf))[None], _load_img_any(os.path.join(gt_dir, gf))[None]


DATASET_ITERS = {
    "SDSD_indoor":  ("video", iter_sdsd_indoor_stcd),
    "SDSD_outdoor": ("video", iter_sdsd_outdoor),
    "BVI_RLV":      ("video", iter_bvi),
    "LOL_v1":       ("image", iter_lol_v1),
    "LOLv2_Syn":    ("image", lambda: iter_lol_v2("Syn")),
    "LOLv2_Real":   ("image", lambda: iter_lol_v2("Real")),
}


# ---------------- eval loop ----------------
def evaluate_method(
    method_name: str,
    infer_fn: Callable[[torch.Tensor], torch.Tensor],
    datasets: Optional[List[str]] = None,
    out_path: str = "/cache/zihaohe/baseline_ckpts/results.json",
    notes: str = "",
    device=None,
    n_frames: int = 16,
    crop: int = 512,
    limit_seqs: Optional[int] = None,
    dtype=torch.float32,
):
    """
    infer_fn: takes (B=1, 3, H, W) float [0,1] on device, returns same shape [0,1].
      Called once per frame (video) or once per image (image data).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if datasets is None:
        datasets = list(DATASET_ITERS.keys())

    per_seq: Dict[str, Dict[str, float]] = {}
    summary: Dict[str, Dict[str, float]] = {}

    for ds_name in datasets:
        kind, iter_fn = DATASET_ITERS[ds_name]
        it_kwargs = {}
        if kind == "video":
            iter_ = iter_fn(n_frames=n_frames, crop=crop)
        else:
            iter_ = iter_fn()
        psnrs_ds: List[float] = []; lpips_ds: List[float] = []
        print(f"\n=== {method_name} / {ds_name} ({kind}) ===", flush=True)
        t0 = time.time()
        for i, (seq_name, lq_arr, gt_arr) in enumerate(iter_):
            if limit_seqs is not None and i >= limit_seqs: break
            # (T, H, W, 3) uint8 → (T, 3, H, W) [0,1]
            T = lq_arr.shape[0]
            preds_u8: List[np.ndarray] = []
            gts_u8: List[np.ndarray] = []
            lp_seq: List[float] = []; ps_seq: List[float] = []
            for t in range(T):
                lq_t = torch.from_numpy(lq_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                gt_t = torch.from_numpy(gt_arr[t]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                if dtype != torch.float32:
                    lq_t = lq_t.to(dtype)
                out = infer_fn(lq_t)
                out = out.clamp(0, 1).float()
                out_u8 = (out[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                gt_u8 = (gt_t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                ps_seq.append(rgb_psnr_uint8(out_u8, gt_u8))
                lp_seq.append(lpips_batch(out.float(), gt_t.float()))
                preds_u8.append(out_u8); gts_u8.append(gt_u8)
            mp = float(np.mean(ps_seq)); ml = float(np.mean(lp_seq))
            key = f"{ds_name}__{seq_name}"
            per_seq[key] = {"psnr_rgb": mp, "lpips": ml, "n_frames": T}
            psnrs_ds.append(mp); lpips_ds.append(ml)
            print(f"  [{i+1:3d}] {seq_name}: PSNR={mp:.3f} LPIPS={ml:.4f}  ({T} frame{'s' if T>1 else ''})", flush=True)
        if psnrs_ds:
            summary[ds_name] = {
                "psnr_rgb": float(np.mean(psnrs_ds)),
                "lpips":    float(np.mean(lpips_ds)),
                "n_seqs":   len(psnrs_ds),
            }
        else:
            summary[ds_name] = {"psnr_rgb": float("nan"), "lpips": float("nan"), "n_seqs": 0}
        dt = time.time() - t0
        print(f"  MEAN {ds_name}: PSNR={summary[ds_name]['psnr_rgb']:.3f} LPIPS={summary[ds_name]['lpips']:.4f}  ({dt:.1f}s, {summary[ds_name]['n_seqs']} seqs)", flush=True)

    out = {
        "method": method_name,
        "notes": notes,
        "protocol": f"video: center T={n_frames}, {crop}x{crop} center-crop; images: full-image. PSNR uint8 RGB, LPIPS pyiqa alex.",
        "per_seq": per_seq,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVE] {out_path}", flush=True)
    return out
