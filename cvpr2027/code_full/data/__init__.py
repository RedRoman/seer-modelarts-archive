"""Dataset factory for V-JEPA-LLVE v5."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from .sdsd_dataset import SDSDDataset


_REGISTRY: Dict[str, type] = {
    "sdsd": SDSDDataset,
}


def build_one(cfg: Dict[str, Any], clip_len: int, crop_size: int,
              is_train: bool) -> Dataset:
    kind = cfg["kind"].lower()
    if kind not in _REGISTRY:
        raise ValueError(f"unknown dataset kind {kind!r}; valid: {list(_REGISTRY)}")
    kwargs = {k: v for k, v in cfg.items() if k not in ("kind", "weight")}
    kwargs["clip_len"] = cfg.get("clip_len", clip_len)
    kwargs["crop_size"] = cfg.get("crop_size", crop_size)
    kwargs["is_train"] = is_train
    return _REGISTRY[kind](**kwargs)


def build_dataset(cfg: Dict[str, Any], clip_len: int = 16,
                  crop_size: int = 256, is_train: bool = True) -> Dataset:
    if "datasets" in cfg:
        sub_cfgs: List[Dict[str, Any]] = cfg["datasets"]
        parts = [build_one(c, clip_len, crop_size, is_train) for c in sub_cfgs]
        return ConcatDataset(parts)
    return build_one(cfg, clip_len, crop_size, is_train)


def build_sampler(cfg: Dict[str, Any], dataset: Dataset):
    """Weighted sampler if cfg.datasets[i].weight differs; None otherwise."""
    if not isinstance(dataset, ConcatDataset):
        return None
    sub_cfgs: List[Dict[str, Any]] = cfg.get("datasets", [])
    weights = [float(c.get("weight", 1.0)) for c in sub_cfgs]
    if all(w == weights[0] for w in weights):
        return None
    if len(weights) != len(dataset.datasets):
        return None
    per_sample = torch.empty(len(dataset), dtype=torch.double)
    offset = 0
    for w, sub in zip(weights, dataset.datasets):
        n = len(sub)
        per_sample[offset:offset + n] = w / max(n, 1)
        offset += n
    return WeightedRandomSampler(
        per_sample, num_samples=len(dataset), replacement=True,
    )


__all__ = ["build_one", "build_dataset", "build_sampler", "SDSDDataset"]
