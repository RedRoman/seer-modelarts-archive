"""SDSD paired low-light video dataset.

Supports the SDSD static_np layout (Retinexformer/SNR-Net release):
    <root>/indoor_static_np/{input,GT}/<seq>/<frame>.npy
    <root>/outdoor_static_np/{input,GT}/<seq>/<frame>.npy

When `subset='direct'`, `<root>` itself is the subset root (handles users who
already point at SDSD_indoor or SDSD_outdoor). Layout auto-detection covers:
  A: original SDSD     <root>/{LQ,GT}/<seq>/...
  B: Retinexformer     <root>/{Train,Test}/{input,target}/<seq>/...
  C: static_np         <root>/indoor_static_np/{input,GT}/<seq>/...

If your layout differs, override `lq_name` / `gt_name` in config.

Output schema:
    {"lq": (3,T,H,W) float in [0,1],
     "hq": (3,T,H,W) float in [0,1],
     "meta": dict}
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from .common import (
    clip_to_tensor, list_images, list_sequences,
    make_eval_rng, make_train_rng,
    paired_center_crop, paired_hflip, paired_random_crop, paired_vflip,
    read_image_01, sample_clip_indices,
)


def _detect_pair_dirs(subset_root, split, lq_name, gt_name):
    """Returns (lq_dir, gt_dir, split_in_dir_layout)."""
    split_cap = {"train": "Train", "test": "Test",
                 "eval": "Test", "val": "Test"}.get(split.lower(), "Train")

    # Layout B: split baked in
    for split_dir in (split_cap, split_cap.lower()):
        for ln, gn in (("input", "target"), ("low", "normal"), ("LQ", "GT")):
            lq_dir = os.path.join(subset_root, split_dir, ln)
            gt_dir = os.path.join(subset_root, split_dir, gn)
            if os.path.isdir(lq_dir) and os.path.isdir(gt_dir):
                return lq_dir, gt_dir, True

    # Layout C: static_np wrapper
    for wrapper in ("indoor_static_np", "outdoor_static_np",
                    "indoor_np", "outdoor_np", "static_np"):
        wrap_root = os.path.join(subset_root, wrapper)
        if not os.path.isdir(wrap_root):
            continue
        for ln, gn in (("input", "GT"), ("input", "gt"),
                       ("input", "target"), ("LQ", "GT")):
            lq_dir = os.path.join(wrap_root, ln)
            gt_dir = os.path.join(wrap_root, gn)
            if os.path.isdir(lq_dir) and os.path.isdir(gt_dir):
                return lq_dir, gt_dir, False

    # Layout A: flat
    for ln, gn in ((lq_name, gt_name), ("LQ", "GT"),
                   ("input", "GT"), ("input", "gt"),
                   ("low", "normal"), ("low", "gt"), ("input", "target")):
        lq_dir = os.path.join(subset_root, ln)
        gt_dir = os.path.join(subset_root, gn)
        if os.path.isdir(lq_dir) and os.path.isdir(gt_dir):
            return lq_dir, gt_dir, False

    raise FileNotFoundError(
        f"Could not auto-detect SDSD layout under {subset_root!r}. "
        f"Tried Retinexformer/static_np/original conventions."
    )


def _load_split_file(path):
    if path is None:
        return None
    with open(path, "r") as f:
        names = [s.strip() for s in f if s.strip() and not s.startswith("#")]
    return names


def _gather_pairs(subset_root, split, lq_name, gt_name, allowed):
    lq_dir, gt_dir, split_in_dir = _detect_pair_dirs(
        subset_root, split, lq_name, gt_name,
    )
    out: List[Tuple[str, List[str], List[str]]] = []
    for seq_dir in list_sequences(lq_dir):
        seq = os.path.basename(seq_dir)
        if allowed is not None and seq not in allowed:
            continue
        lq_frames = list_images(os.path.join(lq_dir, seq))
        gt_frames = list_images(os.path.join(gt_dir, seq))
        if not lq_frames or not gt_frames:
            continue
        n = min(len(lq_frames), len(gt_frames))
        if len(lq_frames) != len(gt_frames):
            print(f"[SDSD] warning: seq {seq} has "
                  f"{len(lq_frames)} LQ vs {len(gt_frames)} GT; using first {n}.")
        out.append((seq, lq_frames[:n], gt_frames[:n]))
    return out, split_in_dir


class SDSDDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str = "indoor",
        split: str = "train",
        split_file: Optional[str] = None,
        clip_len: int = 16,
        crop_size: int = 256,
        is_train: bool = True,
        lq_name: str = "LQ",
        gt_name: str = "GT",
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset.lower()
        self.split = split.lower()
        self.clip_len = int(clip_len)
        self.crop_size = int(crop_size)
        self.is_train = bool(is_train)

        allowed = _load_split_file(split_file)

        if self.subset == "indoor":
            subset_roots = [os.path.join(root, "indoor")]
        elif self.subset == "outdoor":
            subset_roots = [os.path.join(root, "outdoor")]
        elif self.subset == "all":
            subset_roots = [os.path.join(root, "indoor"),
                            os.path.join(root, "outdoor")]
        elif self.subset == "direct":
            subset_roots = [root]
        else:
            raise ValueError(
                f"subset must be indoor/outdoor/all/direct, got {subset!r}"
            )

        self.seqs: List[Tuple[str, List[str], List[str]]] = []
        split_in_dir_any = False
        for sr in subset_roots:
            if not os.path.isdir(sr):
                print(f"[SDSD] warning: subset root {sr!r} not found, skipping.")
                continue
            seqs, split_in_dir = _gather_pairs(
                sr, self.split, lq_name, gt_name, allowed,
            )
            self.seqs.extend(seqs)
            split_in_dir_any = split_in_dir_any or split_in_dir

        # Layout A/C: apply 80/20 deterministic split if no split file
        if (not split_in_dir_any) and (allowed is None) and (split_file is None):
            names = [s[0] for s in self.seqs]
            order = np.argsort(names)
            n = len(order)
            n_train = int(round(0.8 * n))
            sel = order[:n_train] if self.split == "train" else order[n_train:]
            sel_set = set(int(i) for i in sel)
            self.seqs = [self.seqs[i] for i in range(n) if i in sel_set]

        if not self.seqs:
            raise RuntimeError(
                f"No SDSD sequences found under {root!r} "
                f"(subset={subset}, split={split})."
            )
        layout = "B (Retinexformer)" if split_in_dir_any else "A/C (static_np or original)"
        print(f"[SDSD] layout={layout} subset={self.subset} split={self.split} "
              f"-> {len(self.seqs)} sequences, "
              f"{sum(len(s[1]) for s in self.seqs)} frames total.")

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict:
        seq, lq_paths, gt_paths = self.seqs[idx]
        rng = make_train_rng() if self.is_train else make_eval_rng(idx)
        indices = sample_clip_indices(
            len(lq_paths), self.clip_len,
            rng=rng if self.is_train else None,
        )
        lq_frames = np.stack([read_image_01(lq_paths[i]) for i in indices], axis=0)
        gt_frames = np.stack([read_image_01(gt_paths[i]) for i in indices], axis=0)
        if self.is_train:
            lq_frames, gt_frames = paired_random_crop(
                lq_frames, gt_frames, self.crop_size, rng,
            )
            lq_frames, gt_frames = paired_hflip(lq_frames, gt_frames, 0.5, rng)
            lq_frames, gt_frames = paired_vflip(lq_frames, gt_frames, 0.5, rng)
        else:
            lq_frames, gt_frames = paired_center_crop(
                lq_frames, gt_frames, self.crop_size,
            )
        return {
            "lq": clip_to_tensor(lq_frames),
            "hq": clip_to_tensor(gt_frames),
            "meta": {"dataset": "sdsd", "subset": self.subset, "seq": seq,
                     "indices": list(indices)},
        }
