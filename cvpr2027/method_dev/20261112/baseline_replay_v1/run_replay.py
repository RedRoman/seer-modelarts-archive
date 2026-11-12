"""Frozen GT-free image-baseline forward pass with the existing v2 scorer."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

METHODS = ("hvi_cidnet", "darkir", "retinexformer")
METRIC_SHA = "a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def verify_release(root):
    manifest = json.loads((root / "release_manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        if digest(root / rel) != expected:
            raise ValueError(f"Frozen release changed: {rel}")
    return manifest


def load_model(method, root, device):
    import torch
    sources = root / "upstream"
    if method == "hvi_cidnet":
        sys.path.insert(0, str(sources / "hvi_cidnet"))
        from net.CIDNet import CIDNet
        model, multiple = CIDNet(), 16
    elif method == "darkir":
        # Direct architecture import avoids the upstream training package's
        # ptflops/DDP initializer; all network source is unmodified.
        sys.path.insert(0, str(sources / "darkir"))
        from DarkIR import DarkIR
        model = DarkIR(img_channel=3, width=32, middle_blk_num_enc=2,
                       middle_blk_num_dec=2, enc_blk_nums=[1, 2, 3],
                       dec_blk_nums=[3, 1, 1], dilations=[1, 4, 9],
                       extra_depth_wise=True)
        multiple = 8
    elif method == "retinexformer":
        sys.path.insert(0, str(sources / "retinexformer"))
        from RetinexFormer_arch import RetinexFormer
        model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40,
                              stage=1, num_blocks=[1, 2, 2])
        multiple = 4
    else:
        raise ValueError(method)
    checkpoint = root / "weights" / (method + ".pt")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("params", payload.get("params_ema", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state or not all(isinstance(x, torch.Tensor) for x in state.values()):
        raise ValueError("Expected a tensor state dictionary")
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(loaded)
    model = model.float().requires_grad_(False).eval().to(device)
    return model, multiple, {"strict_load": True, "state_keys": len(state),
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": digest(checkpoint), "checkpoint": str(checkpoint)}


def forward_frames(model, multiple, low, device, batch=1):
    """Only the LQ observation enters prediction; no ground-truth argument."""
    import torch
    import torch.nn.functional as F
    low = np.ascontiguousarray(low)
    if low.ndim != 4 or low.shape[-1] != 3 or low.dtype != np.uint8:
        raise ValueError("Expected THWC uint8 RGB")
    h, w = low.shape[1:3]
    output = np.empty_like(low)
    with torch.inference_mode():
        for start in range(0, len(low), batch):
            x = torch.from_numpy(low[start:start + batch]).float().permute(0, 3, 1, 2).div_(255).to(device)
            ph, pw = (-h) % multiple, (-w) % multiple
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            if not torch.isfinite(y).all():
                raise ValueError("Nonfinite baseline prediction")
            output[start:start + batch] = (y[:, :, :h, :w].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model-device", choices=("cpu", "npu"), required=True)
    parser.add_argument("--metric-device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.release.resolve()
    manifest = verify_release(root)
    os.environ["SEER_ROOT"] = "/temp/zihaohe/seer_cvpr"
    code = root / "code_full"
    sys.path[:0] = [str(code), str(code / "tools"), "/temp/zihaohe/seer_cvpr/code/baselines", "/temp/zihaohe/seer_cvpr/code/baselines/common"]
    import torch
    if "npu" in (args.model_device, args.metric_device) and not (args.smoke and args.model_device == "cpu"):
        import torch_npu
    torch.set_num_threads(4)
    device = torch.device("cpu" if args.model_device == "cpu" else "npu:0")
    from eval_stcd_transfer import state_digest
    model, multiple, provenance = load_model(args.method, root, device)
    state_before = state_digest(model.state_dict())
    began = time.monotonic()
    if args.smoke:
        rng = np.random.default_rng(0)
        low = rng.integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        result = forward_frames(model, multiple, low, device)
        batched = forward_frames(model, multiple, low, device, batch=2)
        # Float arithmetic may straddle a uint8 rounding boundary.
        maximum = int(np.max(np.abs(result.astype(np.int16) - batched.astype(np.int16))))
        if maximum > 1 or result.shape != low.shape:
            raise AssertionError((maximum, result.shape))
        report = {"purpose": "synthetic strict-load/shape/batch smoke; no data access",
                  "shape": list(result.shape), "maximum_uint8_batch_difference": maximum,
                  "prediction_sha256": array_digest(result), "provenance": provenance}
    else:
        import common.eval_lib as scoring_lib
        import eval_e4
        import pyiqa
        from eval_stcd_transfer import validation_inventory, score_uint8
        if digest(scoring_lib.__file__) != METRIC_SHA:
            raise AssertionError("The existing v2 scoring library changed")
        split = Path(os.environ["SEER_ROOT"]) / "datasets/splits_cvpr.json"
        inventory = validation_inventory(split, Path(os.environ["SEER_ROOT"]))
        expected = [r["name"] for r in inventory]
        eval_e4.SPLITS = str(split)
        reference = json.loads((root / "reference_raw.json").read_text())
        reference_rows = {r["name"]: r["methods"]["base"] for r in reference["per_sequence"]}
        metric_device = torch.device("cpu" if args.metric_device == "cpu" else "npu:0")
        rows, inputs = [], []
        for name, low, gt, kind in eval_e4.held_out_sequences("bvi", "val", 16, 256):
            if kind != "video" or low.shape != (16, 256, 256, 3) or gt.shape != low.shape:
                raise ValueError(name)
            inputs.append({"name": name, "shape": list(low.shape), "dtype": str(low.dtype),
                           "lq_sha256": array_digest(low), "gt_sha256": array_digest(gt)})
            prediction = forward_frames(model, multiple, low, device)
            row = {"name": name, **score_uint8(prediction, gt, metric_device),
                   "prediction_sha256": array_digest(prediction)}
            if not all(np.isfinite(row[k]) for k in ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")):
                raise ValueError("Nonfinite metrics")
            if row["gt_tlp"] != reference_rows[name]["gt_tlp"]:
                raise AssertionError(f"GT temporal reference differs: {name}")
            rows.append(row)
            print(json.dumps(row), flush=True)
        if [r["name"] for r in rows] != expected or len(rows) != 16:
            raise ValueError("Incomplete frozen VAL16 coverage")
        report = {"method": args.method, "scope": "frozen historical checkpoint replay; no target optimization",
            "domain": "bvi", "split": "val", "n": 16, "protocol": "centre16/crop256; uncalibrated rounded uint8 RGB/Y PSNR, LPIPS and absolute tLP gap",
            "gt_policy": "GT enters scores only; no GT rescaling or CP or target fitting",
            "per_sequence": rows, "inputs": inputs,
            "means": {k: float(np.mean([r[k] for r in rows])) for k in ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")},
            "validation_pair_inventory": inventory, "split_sha256": digest(split),
            "provenance": provenance, "gt_temporal_reference_exact": True,
            "scoring_source_sha256": {"eval_lib": digest(scoring_lib.__file__), "eval_e4": digest(eval_e4.__file__),
                                      "score_helper": digest(code / "tools/eval_stcd_transfer.py")},
            "runtime": {"torch": str(torch.__version__), "pyiqa": str(pyiqa.__version__),
                        "model_device": str(device), "metric_device": str(metric_device), "threads": 4, "batch": 1}}
    after = state_digest(model.state_dict())
    if after != state_before:
        raise AssertionError("Baseline state changed during inference")
    report.update({"state_unchanged": True, "state_sha256": state_before,
        "release_manifest_sha256": digest(root / "release_manifest.json"),
        "adapter_sha256": digest(__file__), "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - began})
    write_new(args.out, report)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
