"""BVI-RLV (BVI-Lowlight-videos) dataset loader for V-JEPA-LLVE.

Layout (after unzip):
  <root>/BVI-Lowlight-videos/
    S01_colour_chart/
      low_light_10/   00001.png ... 00248.png  (LQ at 10 fps)
      low_light_20/   00001.png ... 00248.png  (LQ at 20 fps)
      normal_light_10/ ...                     (HQ at 10 fps)
      normal_light_20/ ...                     (HQ at 20 fps)
    S02_animals1/...
    ...
    S40_Books5/

40 scenes × 2 fps variants = 80 LQ/HQ sequence pairs total.
Frame counts vary per scene; first scene = 248 each.
Resolution: 1920x1080 RGB.

Pair convention:
  LQ = low_light_{fps}/<frame>.png
  HQ = normal_light_{fps}/<frame>.png  (same fps, same frame number)

Spec: VJEPA_LLVE_CODE_REVIEW.md §CK2 (BVI download).
"""
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    clip_to_tensor, list_images, list_sequences,
    make_eval_rng, make_train_rng,
    paired_center_crop, paired_random_crop,
    read_image_01, sample_clip_indices,
)


def _list_bvi_pairs(root: str, fps_variants: Tuple[str, ...] = ("10", "20")
                    ) -> List[Tuple[str, str, List[str], List[str]]]:
    """Return list of (scene_name, fps, lq_paths, gt_paths)."""
    out = []
    if not os.path.isdir(root):
        return out
    for scene in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fps in fps_variants:
            lq_dir = os.path.join(scene_dir, f"low_light_{fps}")
            gt_dir = os.path.join(scene_dir, f"normal_light_{fps}")
            if not (os.path.isdir(lq_dir) and os.path.isdir(gt_dir)):
                continue
            lq_frames = list_images(lq_dir)
            gt_frames = list_images(gt_dir)
            if not lq_frames or not gt_frames:
                continue
            # Pair by basename intersection (defensive against frame drops)
            lq_by = {os.path.basename(p): p for p in lq_frames}
            gt_by = {os.path.basename(p): p for p in gt_frames}
            common = sorted(set(lq_by) & set(gt_by))
            if len(common) < 16:  # skip too-short pairs
                continue
            lq_paths = [lq_by[n] for n in common]
            gt_paths = [gt_by[n] for n in common]
            out.append((scene, fps, lq_paths, gt_paths))
    return out


class BVIRLVDataset(Dataset):
    """BVI-RLV LQ/HQ pair dataset.

    Args:
        root: path to BVI-Lowlight-videos/ directory.
        split: 'train' | 'test' | 'all' — deterministic 80/20 by sorted scene name.
        clip_len: number of frames per sample.
        crop_size: spatial crop size for is_train; full resolution for eval.
        is_train: random crop + temporal sampling if True; center otherwise.
        fps_variants: which fps variants to include (default both).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 16,
        crop_size: int = 256,
        is_train: bool = True,
        fps_variants: Tuple[str, ...] = ("10", "20"),
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split.lower()
        self.clip_len = int(clip_len)
        self.crop_size = int(crop_size)
        self.is_train = bool(is_train)

        all_pairs = _list_bvi_pairs(root, fps_variants)
        # Dedup by scene for the split decision (train/test split is by SCENE not pair)
        scene_names = sorted(set(p[0] for p in all_pairs))
        n = len(scene_names)
        n_train = int(round(0.8 * n))
        train_scenes = set(scene_names[:n_train])
        test_scenes = set(scene_names[n_train:])
        if self.split == "train":
            keep = train_scenes
        elif self.split == "test":
            keep = test_scenes
        elif self.split == "all":
            keep = set(scene_names)
        else:
            raise ValueError(f"split must be train/test/all, got {split!r}")
        self.seqs: List[Tuple[str, List[str], List[str]]] = [
            (f"{scene}__{fps}fps", lq, gt) for (scene, fps, lq, gt) in all_pairs
            if scene in keep
        ]
        if not self.seqs:
            raise RuntimeError(
                f"BVIRLVDataset: no sequences under {root!r} matching split={split!r}. "
                f"Did you unzip BVI-Lowlight-videos.zip into <root>?"
            )
        n_frames = sum(len(lq) for _, lq, _ in self.seqs)
        print(f"[BVI-RLV] {self.split}: {len(self.seqs)} sequences "
              f"({len(keep)} scenes × ≤{len(fps_variants)} fps), {n_frames} frames total")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq_name, lq_paths, gt_paths = self.seqs[idx]
        n_total = len(lq_paths)
        if self.is_train:
            rng = make_train_rng()
            t_idx = sample_clip_indices(n_total, self.clip_len, rng=rng)
            lq_arr = np.stack([read_image_01(lq_paths[i]) for i in t_idx], axis=0)
            gt_arr = np.stack([read_image_01(gt_paths[i]) for i in t_idx], axis=0)
            lq_arr, gt_arr = paired_random_crop(lq_arr, gt_arr, self.crop_size, rng)
        else:
            # eval: deterministic center temporal clip
            T = min(self.clip_len, n_total)
            start = (n_total - T) // 2
            t_idx = list(range(start, start + T))
            while len(t_idx) < self.clip_len:
                t_idx.append(t_idx[-1])
            lq_arr = np.stack([read_image_01(lq_paths[i]) for i in t_idx], axis=0)
            gt_arr = np.stack([read_image_01(gt_paths[i]) for i in t_idx], axis=0)
            lq_arr, gt_arr = paired_center_crop(lq_arr, gt_arr, self.crop_size)
        lq = clip_to_tensor(lq_arr)  # (3, T, H, W)
        gt = clip_to_tensor(gt_arr)
        return {"lq": lq, "hq": gt, "name": seq_name}
