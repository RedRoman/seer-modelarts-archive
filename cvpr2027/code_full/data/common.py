"""Shared data utilities: read images/.npy, paired crops/flips, RNG."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NUMPY_EXTS = (".npy",)
FRAME_EXTS = IMAGE_EXTS + NUMPY_EXTS


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_image_01(path: str) -> np.ndarray:
    """Read a single image or .npy frame to (H, W, 3) float32 in [0, 1].

    Handles:
      - PIL formats (PNG/JPG/...): converted to RGB, /255 to [0,1]
      - .npy: auto-detects (H,W,3)/(3,H,W)/(H,W) and uint8/uint16/float32 ranges
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in NUMPY_EXTS:
        arr = np.load(path)
        # Channels-last detection
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # Range normalization
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        else:
            arr = arr.astype(np.float32)
            if arr.max() > 1.5:  # likely 0-255 stored as float
                arr = arr / 255.0
        return np.ascontiguousarray(arr)
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def list_images(root: str) -> List[str]:
    """Frame files (image or .npy) sorted by filename."""
    out = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isfile(full) and name.lower().endswith(FRAME_EXTS):
            out.append(full)
    out.sort()
    return out


def list_sequences(root: str) -> List[str]:
    return sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )


# ---------------------------------------------------------------------------
# Paired aug
# ---------------------------------------------------------------------------
def _pad_if_small(lq, hq, crop):
    H, W = lq.shape[-3:-1]
    if H >= crop and W >= crop:
        return lq, hq
    pad_h = max(0, crop - H)
    pad_w = max(0, crop - W)
    pad = ((0, 0),) * (lq.ndim - 3) + ((0, pad_h), (0, pad_w), (0, 0))
    return np.pad(lq, pad, mode="edge"), np.pad(hq, pad, mode="edge")


def paired_random_crop(lq, hq, crop, rng: np.random.Generator):
    lq, hq = _pad_if_small(lq, hq, crop)
    H, W = lq.shape[-3:-1]
    y = int(rng.integers(0, H - crop + 1))
    x = int(rng.integers(0, W - crop + 1))
    return (lq[..., y:y + crop, x:x + crop, :],
            hq[..., y:y + crop, x:x + crop, :])


def paired_center_crop(lq, hq, crop):
    lq, hq = _pad_if_small(lq, hq, crop)
    H, W = lq.shape[-3:-1]
    y = (H - crop) // 2
    x = (W - crop) // 2
    return (lq[..., y:y + crop, x:x + crop, :],
            hq[..., y:y + crop, x:x + crop, :])


def paired_hflip(lq, hq, p, rng):
    if rng.random() < p:
        lq = lq[..., :, ::-1, :].copy()
        hq = hq[..., :, ::-1, :].copy()
    return lq, hq


def paired_vflip(lq, hq, p, rng):
    if rng.random() < p:
        lq = lq[..., ::-1, :, :].copy()
        hq = hq[..., ::-1, :, :].copy()
    return lq, hq


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------
def clip_to_tensor(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 3:                       # (H, W, 3) -> (3, H, W)
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    if arr.ndim == 4:                       # (T, H, W, 3) -> (3, T, H, W)
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(3, 0, 1, 2)))
    raise ValueError(f"expected 3 or 4 dim array, got {arr.shape}")


def repeat_single_to_clip(img: np.ndarray, T: int) -> np.ndarray:
    return np.broadcast_to(img[None, ...], (T,) + img.shape).copy()


# ---------------------------------------------------------------------------
# Clip sampling
# ---------------------------------------------------------------------------
def sample_clip_indices(num_frames: int, clip_len: int,
                        rng: Optional[np.random.Generator] = None) -> List[int]:
    if num_frames >= clip_len:
        if rng is None:
            start = 0
        else:
            start = int(rng.integers(0, num_frames - clip_len + 1))
        return list(range(start, start + clip_len))
    idx = list(range(num_frames))
    while len(idx) < clip_len:
        idx.append(num_frames - 1)
    return idx[:clip_len]


# ---------------------------------------------------------------------------
# Per-sample RNG
# ---------------------------------------------------------------------------
def make_train_rng() -> np.random.Generator:
    return np.random.default_rng()


def make_eval_rng(idx: int) -> np.random.Generator:
    """Deterministic generator for val: same sample → same crop."""
    return np.random.default_rng(12345 + idx)
