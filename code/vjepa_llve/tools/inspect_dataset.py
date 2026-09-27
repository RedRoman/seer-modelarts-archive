"""Quick sanity check on a dataset: shape, dtype, range, LQ vs HQ brightness."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--kind", default="sdsd", choices=["sdsd"])
    ap.add_argument("--subset", default="direct")
    ap.add_argument("--split", default="train")
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--crop_size", type=int, default=256)
    args = ap.parse_args()

    cfg = {
        "kind": args.kind,
        "root": args.root,
        "subset": args.subset,
        "split": args.split,
    }
    ds = build_dataset(cfg, clip_len=args.clip_len,
                       crop_size=args.crop_size, is_train=False)
    print(f"Dataset kind={args.kind} root={args.root}: {len(ds)} samples")

    n = min(3, len(ds))
    for i in range(n):
        item = ds[i]
        lq = item["lq"]
        hq = item["hq"]
        meta = item["meta"]
        print(f"\n--- sample {i} (seq={meta.get('seq', '?')}) ---")
        print(f"LQ: shape={tuple(lq.shape)} dtype={lq.dtype} "
              f"range [{lq.min():.3f}, {lq.max():.3f}] mean={lq.mean():.3f}")
        print(f"HQ: shape={tuple(hq.shape)} dtype={hq.dtype} "
              f"range [{hq.min():.3f}, {hq.max():.3f}] mean={hq.mean():.3f}")
        if hq.mean() > lq.mean():
            print("OK: HQ brighter than LQ (expected for low-light pair).")
        else:
            print("WARN: LQ not darker than HQ. Pair may be swapped.")


if __name__ == "__main__":
    main()
