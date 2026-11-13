"""Reverse Cross-Domain Training of EID-TT v2 on BVI TRAIN48 only (for BVI -> SDSD transfer).

Strictly uses ONLY BVI_RLV["train"] (48 sequences = 24 scenes x 2 conditions).
Never opens BVI VAL16, BVI TEST16, or any SDSD file. Trains 3 seeds (0, 1, 2)
of EID-TT v2 (`eid_full`) with identical hyperparameters to `train_source_eid.py`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RAFT_SHA256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
SPLIT_SHA256 = "fb6be5ea78ef73f5cc9db90bca75813bf597275f7609fe39d82f5b2f883e9ad0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        arr = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(tuple(arr.shape)).encode())
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("/cache/zihaohe/seer_baseline_replay_20261112/release_v1"),
    )
    parser.add_argument(
        "--raft-weights",
        type=Path,
        default=Path("/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    _require(not args.out.exists(), f"Output already exists: {args.out}")

    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    _require(vis in {"0", "1", "2", "3", "4", "5", "6", "7"}, "Set explicit single ASCEND_RT_VISIBLE_DEVICES")
    smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    _require(f"No running processes found in NPU {vis}" in smi, f"Physical NPU {vis} is not idle")

    os.environ["SEER_ROOT"] = "/temp/zihaohe/seer_cvpr"
    code_root = args.release.resolve() / "code_full"
    sys.path[:0] = [
        str(code_root),
        str(code_root / "tools"),
        "/temp/zihaohe/seer_cvpr/code/baselines",
        "/temp/zihaohe/seer_cvpr/code/baselines/common",
    ]

    import torch
    import torch_npu  # noqa: F401
    import eval_e4
    from torchvision.models.optical_flow import raft_small
    from eid_transport import (
        EIDTrajectoryTransport,
        OFFSETS,
        compute_eid_training_loss,
        compute_reciprocal_raft_flows,
    )
    from eval_cross_domain_eid import build_backbone_predictor

    torch.set_num_threads(4)
    device = torch.device("npu:0")

    split_path = Path(os.environ["SEER_ROOT"]) / "datasets/splits_cvpr.json"
    _require(digest(split_path) == SPLIT_SHA256, "splits_cvpr.json SHA mismatch")
    eval_e4.SPLITS = str(split_path)

    _require(digest(args.raft_weights) == RAFT_SHA256, "RAFT-small SHA mismatch")
    raft = raft_small(weights=None, progress=False).eval()
    raft.load_state_dict(torch.load(str(args.raft_weights), map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False).to(device)

    predict_hvi, check_hvi, hvi_prov = build_backbone_predictor(
        "hvi_cidnet", args.release.resolve(), Path("/nonexistent"), device
    )

    args.out.mkdir(parents=True, exist_ok=False)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    print("Preparing BVI TRAIN48 clips (centre16/crop256, zero VAL/TEST/SDSD access)...", flush=True)
    t_prep0 = time.monotonic()
    helper = EIDTrajectoryTransport(mode="eid_full").eval().to(device)
    train_clips = []
    with torch.no_grad():
        for name, low_u8, gt_u8, kind in eval_e4.held_out_sequences("bvi", "train", 16, 256):
            _require(kind == "video" and low_u8.shape == (16, 256, 256, 3), f"Invalid BVI train clip: {name}")
            low_t = torch.from_numpy(low_u8.copy()).float().div_(255.0).permute(3, 0, 1, 2).unsqueeze(0)
            gt_t = torch.from_numpy(gt_u8.copy()).float().div_(255.0).permute(3, 0, 1, 2).unsqueeze(0)
            base_d = predict_hvi(low_u8)
            flows_t = compute_reciprocal_raft_flows(base_d, raft, offsets=OFFSETS, flow_iters=12, pair_batch=16)
            geom_d, feats_d, cache_d = helper.prepare_static_inputs(low_t.to(device), base_d, flows_t.to(device))
            train_clips.append(
                {
                    "name": name,
                    "low": low_t,
                    "base": base_d.cpu(),
                    "gt": gt_t,
                    "flows": flows_t,
                    "geom": {k: v.cpu() for k, v in geom_d.items()},
                    "feats": feats_d.cpu(),
                    "cache": {k: v.cpu() for k, v in cache_d.items()},
                }
            )
    _require(len(train_clips) == 48, f"Expected 48 BVI train clips, got {len(train_clips)}")
    _require(check_hvi(), "HVI state changed during BVI train preparation")
    prep_sec = time.monotonic() - t_prep0
    print(f"Prepared 48 BVI TRAIN clips + static features in {prep_sec:.2f}s", flush=True)

    del raft
    del helper

    logs = []
    for seed in args.seeds:
        tag = f"eid_full_s{seed}"
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(20000 + seed)

        model = EIDTrajectoryTransport(mode="eid_full").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps_per_ep = math.ceil(len(train_clips) / args.batch_size)
        total_steps = args.epochs * steps_per_ep
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

        t0 = time.monotonic()
        step_count = 0
        final_loss = 0.0
        for _ in range(args.epochs):
            perm = rng.permutation(len(train_clips))
            model.train()
            ep_tot = []
            for start in range(0, len(perm), args.batch_size):
                bidx = [int(i) for i in perm[start:start + args.batch_size]]
                low_b = torch.cat([train_clips[i]["low"] for i in bidx], dim=0).to(device)
                base_b = torch.cat([train_clips[i]["base"] for i in bidx], dim=0).to(device)
                gt_b = torch.cat([train_clips[i]["gt"] for i in bidx], dim=0).to(device)
                flows_b = torch.cat([train_clips[i]["flows"] for i in bidx], dim=0).to(device)
                geom_b = {k: torch.cat([train_clips[i]["geom"][k] for i in bidx], dim=0).to(device) for k in train_clips[0]["geom"]}
                feats_b = torch.cat([train_clips[i]["feats"] for i in bidx], dim=0).to(device)
                scache_b = {k: torch.cat([train_clips[i]["cache"][k] for i in bidx], dim=0).to(device) for k in train_clips[0]["cache"]}

                optimizer.zero_grad(set_to_none=True)
                out = model(low_b, base_b, flows_b, geom=geom_b, static_inputs=(feats_b, scache_b))
                ld = compute_eid_training_loss(out["pred_pixels"], base_b, gt_b, flows_b, geom_b, loss_mode="decoupled")
                ld["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                step_count += 1
                ep_tot.append(float(ld["total"].item()))
            final_loss = float(np.mean(ep_tot))

        model.eval().requires_grad_(False)
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        cpath = ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "tag": tag,
                "variant": "eid_full",
                "arch_mode": "eid_full",
                "loss_mode": "decoupled",
                "source_domain": "BVI_TRAIN48",
                "seed": seed,
                "epochs": args.epochs,
                "steps": step_count,
                "config": model.get_config(),
                "state_dict": cpu_state,
                "state_sha256": state_digest(cpu_state),
            },
            str(cpath),
        )
        logs.append(
            {
                "tag": tag,
                "seed": seed,
                "steps": step_count,
                "elapsed_seconds": time.monotonic() - t0,
                "final_loss": final_loss,
                "checkpoint": str(cpath),
                "checkpoint_sha256": digest(cpath),
                "state_sha256": state_digest(cpu_state),
            }
        )
        print(json.dumps(logs[-1]), flush=True)

    write_new_json(
        args.out / "bvi_reverse_train_summary.json",
        {
            "schema_version": 2,
            "scope": "BVI TRAIN48 only (24 scenes x 2 conditions); zero BVI VAL/TEST or SDSD access",
            "n_train_clips": 48,
            "prep_elapsed_seconds": prep_sec,
            "backbone_provenance": hvi_prov,
            "raft_weights_sha256": RAFT_SHA256,
            "split_sha256": SPLIT_SHA256,
            "runs": logs,
            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
