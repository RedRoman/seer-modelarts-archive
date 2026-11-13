"""Cache GT-free canonical-RGB reciprocal RAFT-small flows on the 136 source HVI clips.

Reads ONLY /cache/zihaohe/seer_color_contract_20261112/cache_rgb_hvi_v1 (which
contains zero GT tensors) and frozen RAFT-small weights. No ground-truth file
or target-domain sequence is opened. Supports --smoke (first 2 FIT clips to
measure exact throughput) and full 136-clip export.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
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
HVI_WEIGHT_SHA256 = "6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27"
HVI_STATE_SHA256 = "afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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
        "--hvi-cache",
        type=Path,
        default=Path("/cache/zihaohe/seer_color_contract_20261112/cache_rgb_hvi_v1"),
    )
    parser.add_argument(
        "--raft-weights",
        type=Path,
        default=Path("/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--flow-iters", type=int, default=12)
    parser.add_argument("--pair-batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="Run only first 2 clips to measure throughput")
    args = parser.parse_args()

    _require(not args.out.exists(), f"Output directory already exists: {args.out}")
    _require(digest(args.raft_weights) == RAFT_SHA256, "RAFT-small weight SHA256 mismatch")

    hvi_manifest_path = args.hvi_cache / "manifest.json"
    hvi_manifest_sha = digest(hvi_manifest_path)
    hvi_manifest = json.loads(hvi_manifest_path.read_text())
    _require(
        hvi_manifest.get("complete") is True
        and hvi_manifest.get("n") == 136
        and hvi_manifest.get("input_color") == "RGB"
        and hvi_manifest.get("output_color") == "RGB"
        and hvi_manifest.get("weight_sha256") == HVI_WEIGHT_SHA256
        and hvi_manifest.get("model_state_sha256") == HVI_STATE_SHA256
        and hvi_manifest.get("gt_storage_entries_read") == 0,
        "Invalid canonical RGB HVI136 manifest",
    )

    if args.device == "npu":
        vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        _require(vis in {"0", "1", "2", "3", "4", "5", "6", "7"}, "Set explicit single ASCEND_RT_VISIBLE_DEVICES")
        smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
        _require(f"No running processes found in NPU {vis}" in smi, f"Physical NPU {vis} is not idle")
        import torch
        import torch_npu  # noqa: F401
        device = torch.device("npu:0")
    else:
        import torch
        device = torch.device("cpu")

    torch.set_num_threads(4)
    from torchvision.models.optical_flow import raft_small
    from eid_transport import OFFSETS, compute_cycle_geometry, compute_reciprocal_raft_flows

    raft = raft_small(weights=None, progress=False).eval()
    raft.load_state_dict(torch.load(str(args.raft_weights), map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False).to(device)
    raft_state_before = state_digest(raft.state_dict())

    args.out.mkdir(parents=True, exist_ok=False)
    rows_to_run = hvi_manifest["per_sequence"][:2] if args.smoke else hvi_manifest["per_sequence"]

    started = time.monotonic()
    out_rows = []
    for idx, item in enumerate(rows_to_run):
        clip_t0 = time.monotonic()
        hvi_file = args.hvi_cache / item["file"]
        _require(digest(hvi_file) == item["file_sha256"], f"HVI cache SHA mismatch: {hvi_file}")
        payload = torch.load(str(hvi_file), map_location="cpu", weights_only=True)
        _require(set(payload.keys()) == {"name", "scene", "base", "lineage"}, "Unexpected keys in HVI payload")
        base = payload["base"]
        _require(
            base.shape == (1, 3, 16, 256, 256)
            and base.dtype == torch.float32
            and tensor_digest(base) == item["base_tensor_sha256"],
            f"Base tensor lineage mismatch on {item['name']}",
        )

        flows = compute_reciprocal_raft_flows(
            video_rgb=base,
            raft=raft,
            offsets=OFFSETS,
            flow_iters=args.flow_iters,
            pair_batch=args.pair_batch,
        )
        _require(bool(torch.isfinite(flows).all()), f"Nonfinite RAFT flow on {item['name']}")
        geom = compute_cycle_geometry(flows)
        mean_valid = float(geom["valid_mask"].mean().item())
        mean_fb_conf = float(geom["fb_conf"].mean().item())

        flow_fname = f"rgb_flow_{idx:03d}.pt"
        flow_path = args.out / flow_fname
        flow_f16 = flows.to(torch.float16)
        torch.save(
            {
                "name": item["name"],
                "scene": item["scene"],
                "offsets": list(OFFSETS),
                "flows_f16": flow_f16,
                "hvi_file": item["file"],
                "hvi_file_sha256": item["file_sha256"],
                "base_tensor_sha256": item["base_tensor_sha256"],
            },
            str(flow_path),
        )
        elapsed_clip = time.monotonic() - clip_t0
        row_record = {
            "index": idx,
            "name": item["name"],
            "scene": item["scene"],
            "file": flow_fname,
            "file_sha256": digest(flow_path),
            "hvi_file_sha256": item["file_sha256"],
            "base_tensor_sha256": item["base_tensor_sha256"],
            "mean_valid_fraction": mean_valid,
            "mean_fb_conf": mean_fb_conf,
            "elapsed_seconds": elapsed_clip,
        }
        out_rows.append(row_record)
        if (idx + 1) % 10 == 0 or args.smoke or (idx + 1) == len(rows_to_run):
            print(json.dumps({"done": idx + 1, "total": len(rows_to_run), "last": row_record}), flush=True)

    raft_state_after = state_digest(raft.state_dict())
    _require(raft_state_after == raft_state_before, "RAFT state changed during flow export")

    total_elapsed = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "complete": True,
        "smoke": bool(args.smoke),
        "n": len(out_rows),
        "offsets": list(OFFSETS),
        "flow_iters": args.flow_iters,
        "pair_batch": args.pair_batch,
        "storage_dtype": "float16",
        "gt_storage_entries_read": 0,
        "target_inputs_read": 0,
        "hvi_manifest_path": str(hvi_manifest_path.resolve()),
        "hvi_manifest_sha256": hvi_manifest_sha,
        "raft_weights_path": str(args.raft_weights.resolve()),
        "raft_weights_sha256": RAFT_SHA256,
        "raft_state_sha256": raft_state_before,
        "raft_state_unchanged": True,
        "device": str(device),
        "elapsed_seconds": total_elapsed,
        "sec_per_clip": total_elapsed / max(len(out_rows), 1),
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "per_sequence": out_rows,
    }
    write_new_json(args.out / "manifest.json", manifest)
    print(json.dumps({"status": "ok", "n": len(out_rows), "sec_per_clip": manifest["sec_per_clip"]}), flush=True)


if __name__ == "__main__":
    main()
