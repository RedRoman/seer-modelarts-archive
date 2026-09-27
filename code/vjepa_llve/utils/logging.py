"""Simple logger that writes both to stdout and to a timestamped file in log_dir."""
from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logger(log_dir: str = ".", name: str = "vjepa_llve") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if called twice
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"train_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
