"""Post-hoc mean/basis conditioning controls on the complete SDSD validation split.

This tool does not train, tune, select a checkpoint, or read the test split. It
uses one full-token-trained E17 checkpoint per invocation. Run the two planned
seeds separately and retain all six conditions for all 16 validation sequences.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / "tools")]
TAPS = (5, 11, 17, 23)
RANK = 4
CONDITIONS = {
    "identity": {"mean": "unchanged", "basis": "identity", "rank": "full"},
    "clip_mean_only": {"mean": "clip", "basis": "none", "rank": 0},
    "clip_mean_clip_basis": {"mean": "clip", "basis": "clip_centered", "rank": RANK},
    "clip_mean_training_basis": {"mean": "clip", "basis": "training", "rank": RANK},
    "training_mean_clip_basis": {"mean": "training", "basis": "clip_centered", "rank": RANK},
    "training_mean_training_basis": {"mean": "training", "basis": "training", "rank": RANK},
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().float().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare_token_statistics(tokens, rank=RANK):
    """One CPU float32 SVD per tap/clip; its basis never uses the training mean."""
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError("expected a single clip's tokens with shape (1, N, D)")
    if rank <= 0 or rank > min(tokens.shape[1:]):
        raise ValueError("rank must fit the clip token matrix")
    with torch.autocast(device_type="cpu", enabled=False):
        field = tokens.detach().cpu().float().contiguous()
        matrix = field[0]
        mean = matrix.mean(0)
        basis = torch.linalg.svd(matrix - mean, full_matrices=False)[2][:rank].contiguous()
    return {"field": field, "clip_mean": mean, "clip_basis": basis}


def apply_token_condition(statistics, training_mean, training_basis, condition):
    """mean + (Z - mean) V V^T, with V stored as orthonormal rows (rank, D)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")
    field = statistics["field"]
    dim = field.shape[-1]
    if tuple(training_mean.shape) != (dim,) or tuple(training_basis.shape) != (RANK, dim):
        raise ValueError("training mean/basis dimensions do not match the token field")
    if training_mean.device.type != "cpu" or training_basis.device.type != "cpu":
        raise ValueError("control projection is evaluated on CPU")
    if condition == "identity":
        return field
    if condition == "clip_mean_only":
        return statistics["clip_mean"].reshape(1, 1, dim).expand_as(field).contiguous()
    specification = CONDITIONS[condition]
    mean = statistics["clip_mean"] if specification["mean"] == "clip" else training_mean
    basis = statistics["clip_basis"] if specification["basis"] == "clip_centered" else training_basis
    with torch.autocast(device_type="cpu", enabled=False):
        matrix = field[0]
        projected = ((matrix - mean) @ basis.T) @ basis + mean
    return projected[None]


class ControlledBackbone(nn.Module):
    """Cache one encoder pass and one SVD per tap for a named validation clip."""
    def __init__(self, backbone, training_mean, training_basis):
        super().__init__()
        self.backbone = backbone
        self.training_mean, self.training_basis = training_mean, training_basis
        self.begin_clip("")

    def begin_clip(self, name):
        self.clip_name = name
        self.condition = "identity"
        self.result = self.statistics = self.input_shape = None
        self.last_token_fingerprint = None

    def forward(self, video):
        if not self.clip_name:
            raise RuntimeError("call begin_clip(name) before scoring a clip")
        if self.result is None:
            self.result = dict(self.backbone(video))
            self.input_shape = tuple(video.shape)
            if len(self.result["feats"]) != len(TAPS):
                raise ValueError("this control requires the four default hierarchical taps")
            # The enclosing evaluator autocasts the encoder. Explicitly leave
            # that context before CPU SVD and projection, as training did.
            with torch.autocast(device_type=video.device.type, enabled=False):
                self.statistics = [prepare_token_statistics(z) for z in self.result["feats"]]
        elif tuple(video.shape) != self.input_shape:
            raise ValueError("a cached clip's shape changed; call begin_clip for each sequence")
        fields = [apply_token_condition(stat, self.training_mean[i], self.training_basis[i], self.condition)
                  for i, stat in enumerate(self.statistics)]
        self.last_token_fingerprint = tensor_fingerprint(fields)
        result = dict(self.result)
        result["feats"] = [field.to(video.device) for field in fields]
        return result


def validate_selection(names, expected):
    if len(expected) != 16 or len(set(expected)) != 16:
        raise ValueError("the registered SDSD validation selection must contain exactly 16 unique sequences")
    if len(names) != 16 or len(set(names)) != 16 or sorted(names) != sorted(expected):
        raise ValueError("score all 16 frozen SDSD validation sequences, without missing or duplicate rows")


def validate_checkpoint(saved):
    expected = {"arch": "v2", "branch": "small", "vjepa": "pretrained",
                "xattn": "global", "kv": "per_tubelet", "layers": "single",
                "token_proj": "none", "edc": "off", "clip_len": 16, "crop": 256}
    # Earlier E17 checkpoints predate these CLI options; their defaults match.
    defaults = {"arch": "v2", "branch": "full", "vjepa": "pretrained",
                "xattn": "global", "kv": "per_tubelet", "layers": "single",
                "token_proj": "none", "edc": "off", "clip_len": 16, "crop": 256}
    for key, value in expected.items():
        if saved.get(key, defaults[key]) != value:
            raise ValueError(f"requires default full-token E17 training: {key} must be {value!r}")
    taps = saved.get("taps", "5,11,17,23")
    taps = tuple(int(t) for t in taps.split(",")) if isinstance(taps, str) else tuple(taps)
    if taps != TAPS or saved.get("seed") not in (0, 1):
        raise ValueError("requires default taps and one of the two planned E17 seeds (0, 1)")
    if saved.get("tag") != f"e17_small_pretrained_s{saved['seed']}":
        raise ValueError("checkpoint tag must identify the planned E17 pretrained arm")
    if saved.get("token_corrupt", 0) != 0 or saved.get("tdelta", 0) != 0:
        raise ValueError("corruption/temporal-loss intervention checkpoints are outside this control")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--basis", required=True, type=Path, help="training-split global basis JSON")
    ap.add_argument("--split", choices=["val"], default="val")
    ap.add_argument("--device", choices=["npu", "cuda"], default="npu")
    ap.add_argument("--out", required=True, type=Path, help="new output path; existing files are never overwritten")
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")
    if args.device == "npu":
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "7":
            ap.error("this validation control is scheduled on visible NPU 7; set ASCEND_RT_VISIBLE_DEVICES=7")
        import torch_npu  # noqa: F401
    import eval_e4 as evaluator
    dev = torch.device(f"{args.device}:0")
    amp_dtype = torch.bfloat16
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    validate_checkpoint(state.get("args", {}))
    del state
    basis = json.loads(args.basis.read_text())
    if basis.get("split") != "train" or tuple(basis.get("taps", ())) != TAPS or basis.get("rank", 0) < RANK:
        raise ValueError("basis must come from the train split and the four default taps, with rank >= 4")
    training_mean = torch.tensor(basis["mean"], dtype=torch.float32)
    training_basis = torch.tensor(basis["components"], dtype=torch.float32)[:, :RANK].contiguous()
    if tuple(training_mean.shape) != (4, 1024) or tuple(training_basis.shape) != (4, RANK, 1024):
        raise ValueError("training basis dimensions must be four 1024-dimensional tap fields")
    torch.testing.assert_close(training_basis @ training_basis.transpose(-1, -2),
                               torch.eye(RANK).expand(4, -1, -1), rtol=1e-4, atol=1e-4)
    splits = json.loads(Path(evaluator.SPLITS).read_text())
    expected = [f"{domain}/{name}" for domain in ("SDSD_indoor", "SDSD_outdoor")
                for name in splits[domain]["val"]]
    validate_selection(expected, expected)
    pb, backbone, refiner, exposure, arm, clip_len, saved = evaluator.build_model(
        str(args.ckpt), dev, reconstruction="saved")
    if refiner is not None or exposure is not None:
        raise ValueError("the planned E17 control has no refiner or exposure head")
    controlled = ControlledBackbone(backbone, training_mean, training_basis).eval()
    rows = {name: [] for name in CONDITIONS}
    observed = []
    sequence_fingerprints = []
    with torch.no_grad():
        for name, lq, gt, kind in evaluator.held_out_sequences("sdsd", "val", 16, 256):
            if kind != "video" or len(lq) != 16 or len(gt) != 16:
                raise ValueError(f"{name}: the fixed protocol requires 16 paired video frames")
            if name not in expected or name in observed:
                raise ValueError(f"unexpected/duplicate validation sequence: {name}")
            observed.append(name)
            controlled.begin_clip(name)
            for condition in CONDITIONS:
                controlled.condition = condition
                pred = evaluator.run_clip(pb, controlled, None, None, arm, lq, clip_len, dev, amp_dtype)
                to_tensor = lambda x: torch.from_numpy(x.copy()).float().div(255.).permute(0, 3, 1, 2)
                row = {"name": name, "kind": kind, "frames": len(pred),
                       "psnr_rgb": float(np.mean([evaluator.rgb_psnr_uint8(p, g) for p, g in zip(pred, gt)])),
                       "psnr_y": float(np.mean([evaluator.psnr_y_uint8(p, g) for p, g in zip(pred, gt)])),
                       "lpips": evaluator.lpips_of(to_tensor(pred), to_tensor(gt), dev),
                       "token_sha256": controlled.last_token_fingerprint}
                rows[condition].append(row)
                print(f"{len(observed):02d}/16 {name} {condition}: PSNR-Y={row['psnr_y']:.6f} LPIPS={row['lpips']:.6f}", flush=True)
            sequence_fingerprints.append({"name": name,
                "clip_mean_sha256": tensor_fingerprint([s["clip_mean"] for s in controlled.statistics]),
                "clip_basis_sha256": tensor_fingerprint([s["clip_basis"] for s in controlled.statistics])})
    validate_selection(observed, expected)
    source_files = [Path(__file__), Path(evaluator.__file__), CODE / "tools" / "checkpoint_config.py",
                    CODE / "models" / "vjepa_backbone.py", CODE / "models" / "pixel_branch_v2.py",
                    CODE / "models" / "pixel_branch_v8.py", Path(inspect.getfile(type(backbone.model)))]
    report = {"analysis": "post-hoc validation mechanism control; no test-set claim",
              "ckpt": str(args.ckpt.resolve()), "checkpoint_sha256": file_sha256(args.ckpt),
              "basis": str(args.basis.resolve()), "basis_sha256": file_sha256(args.basis),
              "source_sha256": {str(path.resolve()): file_sha256(path) for path in source_files},
              "split_file": evaluator.SPLITS, "split_sha256": file_sha256(evaluator.SPLITS),
              "requested_split": args.split, "split": "val", "heldout": "sdsd",
              "seed": saved["seed"], "iters": saved.get("iters"), "arm": arm,
              "n_sequences": 16, "n_frames": 16, "crop": 256, "tile": 0,
              "selection": "all 16 frozen validation sequences; no tuning or selection",
              "sequence_names": observed, "reconstruction": pb.eval_reconstruction,
              "adaptive_basis": "CPU float32 right singular vectors of clip-centered token field, once per clip/tap",
              "projection_formula": "mean + (Z - mean) @ V.T @ V; V stores basis vectors as rows",
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "device": str(dev), "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES") if args.device == "npu" else os.environ.get("CUDA_VISIBLE_DEVICES"),
              "sequence_fingerprints": sequence_fingerprints, "conditions": {}}
    for condition, specification in CONDITIONS.items():
        report["conditions"][condition] = {"mean_basis": specification,
            "aggregate": {metric: float(np.mean([row[metric] for row in rows[condition]]))
                          for metric in ("psnr_rgb", "psnr_y", "lpips")},
            "per_sequence": rows[condition]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
