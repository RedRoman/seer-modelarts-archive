"""Miscellaneous utilities."""
from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(params_iterable: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params_iterable)
