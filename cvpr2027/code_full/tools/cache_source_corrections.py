"""SDSD TRAIN-only frozen observations for source-trained correction selection.

One deterministic clip per source training sequence. Cache GT as loss labels,
never as an input to restoration, flow, candidate construction or selection.
No target-domain frame is read; no GT oracle is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from train_correction_selector import digest, write_json


def source_scene(name):
    if not isinstance(name, str) or re.fullmatch(r"SDSD_(?:indoor|outdoor)/[^/\\]+", name) is None:
        raise ValueError(f"Invalid source name: {name!r}")
    domain, sequence = name.split("/")
    if sequence in (".", ".."):
        raise ValueError("Invalid sequence path")
    return domain + "/" + re.sub(r"_2$", "", sequence)


def source_names(splits, expected=136):
    """Inspect split identifiers only; never enumerate held-out frames."""
    all_train = []
    for key in ("SDSD_indoor", "SDSD_outdoor"):
        entry = splits[key]
        groups = {}
        for split in ("train", "val", "test"):
            names = entry[split]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate {key}/{split} sequence")
            groups[split] = {source_scene(f"{key}/{name}") for name in names}
        if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"Source scene leakage in {key}")
        all_train.extend(f"{key}/{name}" for name in entry["train"])
    if len(all_train) != expected:
        raise ValueError(f"Expected source train{expected}, got {len(all_train)}")
    return all_train


def source_hashes(*extra):
    paths = ("tools/cache_source_corrections.py", "tools/cache_correction_oracle.py",
             "tools/eval_e4.py", "tools/checkpoint_config.py", "models/pixel_branch_v2.py",
             "models/vjepa_backbone.py", "models/token_projection.py", "data/common.py",
             "data/planned_bvi.py", "tools/train_method_pilot.py", "tools/train_correction_selector.py", *extra)
    return {p: digest(CODE / p) for p in dict.fromkeys(paths)}


def check_source_hashes(mapping):
    for relative, expected in mapping.items():
        path = CODE / relative
        if not path.resolve().is_relative_to(CODE) or digest(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def check_base_args(saved, arm, length):
    if (arm not in ("pretrained", "none") or length != 16 or saved.get("seed") != 0
            or saved.get("tag") != f"e17_small_{arm}_s0" or saved.get("iters") != 20000
            or saved.get("heldout") is not None or saved.get("only") is not None):
        raise ValueError("Expected the declared SDSD-only E17 small final20k seed0 base")


def backbone_tokens(backbone, low):
    """Match eval_e4.run_clip for both pretrained and no-token control arms."""
    if backbone is None:
        return [low.new_zeros((low.shape[0], 2048, 1024)) for _ in range(4)]
    if backbone.training:
        raise ValueError("Frozen encoder must remain in eval mode")
    return [z.float() for z in backbone(low)["feats"]]


class SourceClips:
    """Strict paired source frames with an explicit augmentation plan."""
    def __init__(self, split_path, root, seed=20261113, *, expected=136, crop=256, clip_len=16):
        from data.common import list_images
        from data.planned_bvi import _dataset_root
        self.splits = json.loads(Path(split_path).read_text())
        self.names = source_names(self.splits, expected)
        self.items, self.inventory = [], []
        self.crop, self.clip_len = crop, clip_len
        for name in self.names:
            domain, sequence = name.split("/")
            folder = _dataset_root(self.splits[domain]["root"], root)
            low = list_images(str(folder / "input" / sequence))
            gt = list_images(str(folder / "GT" / sequence))
            if len(low) < clip_len or len(low) != len(gt):
                raise ValueError(f"Empty, short or unequal source frame lists: {name}")
            pairs = [[Path(a).name, Path(b).name] for a, b in zip(low, gt)]
            self.items.append((low, gt))
            self.inventory.append({"name": name, "paired_frames": len(low),
                                   "matching_complete_basename_lists": all(a == b for a, b in pairs),
                                   "pairing": "sorted index; SDSD _2 passes retain different original frame numbers",
                                   "first_last_pairs": [pairs[0], pairs[-1]],
                                   "complete_basename_pairs": pairs,
                                   "basename_pairs_sha256": hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()})
        rng = np.random.default_rng(seed)
        self.plan = [{"sequence_index": i, "augmentation_seed": int(rng.integers(0, 2**63 - 1))}
                     for i in range(len(self.names))]

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        from data.common import (clip_to_tensor, paired_random_crop, paired_hflip,
                                 paired_vflip, read_image_01, sample_clip_indices)
        spec = self.plan[index]
        low, gt = self.items[spec["sequence_index"]]
        rng = np.random.default_rng(spec["augmentation_seed"])
        indices = sample_clip_indices(len(low), self.clip_len, rng)
        a, b = [], []
        shape = None
        for i in indices:
            x, y = read_image_01(low[i]), read_image_01(gt[i])
            if (x.ndim != 3 or x.shape[-1] != 3 or x.shape != y.shape
                    or (shape is not None and x.shape != shape)
                    or not all(np.isfinite(z).all() and z.min() >= 0 and z.max() <= 1 for z in (x, y))):
                raise ValueError(f"Invalid source RGB pairing: {self.names[index]} frame {i}")
            shape = x.shape
            a.append(x); b.append(y)
        a, b = paired_random_crop(np.stack(a), np.stack(b), self.crop, rng)
        a, b = paired_hflip(a, b, .5, rng)
        a, b = paired_vflip(a, b, .5, rng)
        return {"name": self.names[index], "lq": clip_to_tensor(a), "gt": clip_to_tensor(b)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--raft-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20261113)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
    from torchvision.models.optical_flow import raft_small
    from cache_correction_oracle import donor_correction
    from eval_e4 import SPLITS, VJEPA_CKPT, build_model
    from train_method_pilot import state_digest
    torch.set_num_threads(4)
    torch.manual_seed(0)
    dev = torch.device(args.device + ":0")
    dataset = SourceClips(SPLITS, os.environ["SEER_ROOT"], args.data_seed)
    model, backbone, refiner, exposure, arm, length, saved = build_model(args.init, dev)
    check_base_args(saved, arm, length)
    if refiner is not None or exposure is not None:
        raise ValueError("Expected unmodified E17 base")
    model.eval().requires_grad_(False)
    base_sha = state_digest(model.state_dict())
    raft = raft_small(weights=None).eval()
    raft.load_state_dict(torch.load(args.raft_weights, map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False).to(dev)
    raft_sha = state_digest(raft.state_dict())
    metadata = {"schema_version": 1, "purpose": "SOURCE SDSD TRAIN-only observations; no target frames or oracle",
                "domain": "sdsd", "source_names": dataset.names, "split_sha256": digest(SPLITS),
                "base_checkpoint_sha256": digest(args.init), "base_saved_args": saved,
                "encoder_weights_sha256": digest(VJEPA_CKPT) if arm == "pretrained" else None,
                "encoder_eval": backbone is None or not backbone.training,
                "encoder_provenance": "Pretrained weight-file SHA plus eval mode; no full backbone-state fingerprint",
                "base_reconstruction": model.eval_reconstruction, "base_state_sha256": base_sha,
                "raft_weights_sha256": digest(args.raft_weights), "raft_state_sha256": raft_sha,
                "args": {"data_seed": args.data_seed, "crop": 256, "clip_len": 16, "offset": 2,
                         "flow_steps": 12, "flow_batch": 2, "patch_size": 8},
                "plan": dataset.plan, "paired_frame_inventory": dataset.inventory,
                "source_sha256": source_hashes(),
                "historical_base_scope": "E17 SDSD-only run identity checked; historical training split SHA not inferred",
                "gt_firewall": "GT stored only as source loss labels; restoration and RAFT receive low/prediction only"}
    write_json(args.out / "provenance.json", metadata)
    rows = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            low, gt = sample["lq"][None].to(dev), sample["gt"][None]
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                tokens = backbone_tokens(backbone, low)
                base = model(low, tokens)["pred_pixels"].float().clamp(0, 1)
            corrections, flows, masks = zip(*(donor_correction(base, raft, d, 12, 2) for d in (-2, 2)))
            tensors = {"lq": low.cpu(), "gt": gt, "base": base.cpu(),
                       "restored_corrections": torch.stack(corrections, 1).cpu(),
                       "flow": torch.stack(flows, 1).cpu(), "valid_mask": torch.stack(masks, 1).cpu()}
            if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
                raise FloatingPointError(sample["name"])
            path = args.out / f"clip_{i:03d}.pt"
            scene = source_scene(sample["name"])
            torch.save({"name": sample["name"], "scene": scene, "plan": dataset.plan[i], "tensors": tensors}, path)
            rows.append({"name": sample["name"], "scene": scene, "cache_path": path.name, "cache_sha256": digest(path)})
            write_json(args.out / "progress.json", {"n": len(rows), "per_sequence": rows})
            print(json.dumps(rows[-1]), flush=True)
    if state_digest(model.state_dict()) != base_sha or state_digest(raft.state_dict()) != raft_sha:
        raise AssertionError("Source caching altered frozen base/RAFT state")
    write_json(args.out / "cache_summary.json", {"n": len(rows), "per_sequence": rows,
               "base_state_unchanged": True, "raft_state_unchanged": True, "oracle_computed": False})


if __name__ == "__main__":
    main()
