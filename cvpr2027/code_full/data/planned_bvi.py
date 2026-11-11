"""Deterministic BVI development training data, separate from frozen pilot v1.

Only train directories are enumerated/read. VAL/test names are checked for
scene separation, but their frames are never opened. The legacy 10/20 suffixes
select low_light_10/20 and normal_light_10/20 directories; they do not request
frame-rate conversion or temporal subsampling. Pair by sorted PNG index only
after checking that both complete filename lists identify the same frames.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from torch.utils.data import Dataset

from .common import (clip_to_tensor, paired_hflip, paired_random_crop,
                     paired_vflip, read_image_01, sample_clip_indices)

DEFAULT_COUNTS = {"train": 48, "val": 16, "test": 16}


def parse_sequence(name):
    """Return (scene, condition), accepting the two legacy suffix spellings."""
    if not isinstance(name, str):
        raise ValueError("BVI sequence names must be strings")
    match = re.fullmatch(r"([^/\\]+)__(?:(10|20)fps|fps(10|20))", name)
    if match is None or match.group(1) in (".", ".."):
        raise ValueError(f"Invalid BVI sequence identifier: {name!r}")
    return match.group(1), match.group(2) or match.group(3)


def validate_split(entry, expected_counts=None):
    counts = dict(DEFAULT_COUNTS if expected_counts is None else expected_counts)
    if set(counts) != {"train", "val", "test"} or any(
            not isinstance(v, int) or v <= 0 or v % 2 for v in counts.values()):
        raise ValueError("expected_counts must give positive even train/val/test counts")
    scenes = {}
    for split in ("train", "val", "test"):
        names = entry[split]
        if not isinstance(names, list) or len(names) != counts[split] or len(set(names)) != len(names):
            raise ValueError(f"BVI {split}: expected {counts[split]} distinct sequence identifiers")
        pairs = [parse_sequence(name) for name in names]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"BVI {split}: duplicate scene/condition (possibly alternate suffix spellings)")
        conditions = Counter(scene for scene, _ in pairs)
        if any(n != 2 for n in conditions.values()):
            raise ValueError(f"BVI {split}: every scene must have both conditions 10 and 20")
        scenes[split] = set(conditions)
        if f"{split}_scenes" in entry:
            declared = entry[f"{split}_scenes"]
            if len(declared) != len(set(declared)) or set(declared) != scenes[split]:
                raise ValueError(f"BVI {split}: declared scene list disagrees with sequences")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = scenes[first] & scenes[second]
        if overlap:
            raise ValueError(f"BVI scene leakage {first}/{second}: {sorted(overlap)}")


def make_plan(n_sequences, n_samples, seed):
    """Same plan format/RNG order as pilot v1, without importing its evaluator."""
    if n_sequences <= 0 or n_samples <= 0:
        raise ValueError("Plan needs positive sequence and sample counts")
    rng = np.random.default_rng(seed)
    plan = []
    while len(plan) < n_samples:
        for index in rng.permutation(n_sequences):
            if len(plan) == n_samples:
                break
            plan.append({"sequence_index": int(index),
                         "augmentation_seed": int(rng.integers(0, 2**63 - 1))})
    return plan


def _dataset_root(stored_root, root):
    path = Path(stored_root)
    for historical_root in (Path("/temp/zihaohe/seer_cvpr"), Path("/cache/zihaohe/seer_cvpr")):
        if path.is_relative_to(historical_root):
            return Path(root) / path.relative_to(historical_root)
    return path if path.is_absolute() else Path(root) / path


def _png_frames(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing BVI training frame directory: {directory}")
    return sorted(str(path) for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() == ".png")


def _checked_image(path):
    image = read_image_01(path)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"Invalid RGB frame shape {image.shape}: {path}")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError(f"BVI frame must be finite and in [0, 1]: {path}")
    return image


class PlannedBVI(Dataset):
    """Train-only paired clips with an explicit, index-stable augmentation plan.

    Arguments match PlannedSDSD. root is SEER_ROOT, not the dataset directory.
    expected_counts is an explicit small-fixture override; production defaults
    to frozen train48/val16/test16. Short clips repeat their final paired frame,
    matching pilot v1. Small images are edge-padded before the paired crop.
    """
    def __init__(self, split_path, root, samples, seed, crop, clip_len, *, expected_counts=None):
        if any(not isinstance(v, int) or v <= 0 for v in (samples, crop, clip_len)):
            raise ValueError("samples, crop and clip_len must be positive integers")
        entry = json.loads(Path(split_path).read_text())["BVI_RLV"]
        validate_split(entry, expected_counts)
        base = _dataset_root(entry["root"], root)
        self.items = []
        for name in entry["train"]:
            scene, condition = parse_sequence(name)
            low = _png_frames(base / scene / f"low_light_{condition}")
            high = _png_frames(base / scene / f"normal_light_{condition}")
            if not low or not high:
                raise ValueError(f"Empty BVI training pair: {name}")
            if len(low) != len(high):
                raise ValueError(f"BVI paired frame count mismatch: {name}, LQ={len(low)} GT={len(high)}")
            if [Path(path).name for path in low] != [Path(path).name for path in high]:
                raise ValueError(f"BVI paired PNG basename mismatch: {name}; refusing shifted frame pairing")
            self.items.append((f"BVI/{name}", low, high))
        self.plan = make_plan(len(self.items), samples, seed)
        self.crop, self.clip_len = crop, clip_len

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        spec = self.plan[index]
        name, low, high = self.items[spec["sequence_index"]]
        rng = np.random.default_rng(spec["augmentation_seed"])
        frames = sample_clip_indices(len(low), self.clip_len, rng)
        lq, gt, shape = [], [], None
        for frame in frames:
            a, b = _checked_image(low[frame]), _checked_image(high[frame])
            if a.shape != b.shape:
                raise ValueError(f"BVI paired frame shape mismatch: {name}, index={frame}, {a.shape} != {b.shape}")
            if shape is not None and a.shape != shape:
                raise ValueError(f"BVI frame shape changes within sampled clip: {name}, index={frame}")
            shape = a.shape
            lq.append(a)
            gt.append(b)
        lq, gt = np.stack(lq), np.stack(gt)
        lq, gt = paired_random_crop(lq, gt, self.crop, rng)
        lq, gt = paired_hflip(lq, gt, .5, rng)
        lq, gt = paired_vflip(lq, gt, .5, rng)
        return {"lq": clip_to_tensor(lq), "gt": clip_to_tensor(gt),
                "name": name, "sample_index": index}
