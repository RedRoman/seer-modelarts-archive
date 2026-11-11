"""SDSD-indoor released STCD reference on frozen BVI VAL16, without adaptation.

This is an indoor-only source reference, not a same-data comparison with Seer's
SDSD indoor+outdoor training. STCD consumes two RGB images after the official
RGB->BGR conversion. Output t uses (t, min(t+1,15)) within the centre16 clip;
the final neighbour is explicitly duplicated. No five-frame context is used.

The two pinned upstream generator files are loaded without the training/DKM
wrapper. Exactly two constructor-only .to('cuda') calls are removed in memory
to permit CPU/NPU execution; upstream files remain unchanged. All other network
operations, strict checkpoint loading and output index [2] are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types

import numpy as np

CODE = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "models/archs/arch_util.py": "7939420cd8105cf430b3e4a92333435ce1a015e1afd0644610dca94627d13bd8",
    "models/archs/low_light_transformer42.py": "b55c74879b6a0c3de44f60b69e79bd6c34a7380490d22038cd125d6f2a0b034b",
}
CHECKPOINT_SHA256 = "dc50957303b2ac15d2f71a32c3d4231c47c5d3f8dc21daa52bccf48faea49cda"
REFERENCE_LABEL = "STCD official SDSD-indoor released reference; zero BVI fine-tuning"
FRAME_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_sources(root, expected=None):
    expected = SOURCE_SHA256 if expected is None else expected
    sources = {}
    for relative, sha in expected.items():
        path = Path(root) / relative
        if digest(path) != sha:
            raise ValueError(f"STCD source SHA256 mismatch: {relative}")
        sources[relative] = path.read_text()
    return sources


def adapt_architecture(source):
    """Isolate the pinned arch import and remove only two forced CUDA moves."""
    imported = "import models.archs.arch_util as arch_util"
    forced_cuda = ".to('cuda')"
    if source.count(imported) != 1 or source.count(forced_cuda) != 2:
        raise ValueError("Unexpected upstream STCD architecture; refusing unguarded adaptation")
    return source.replace(imported, "# arch_util injected from the pinned isolated module").replace(forced_cuda, "")


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_stcd(root, checkpoint, device):
    import torch
    sources = verified_sources(root)
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("STCD released-reference checkpoint SHA256 mismatch")
    utility = types.ModuleType("_seer_stcd_arch_util")
    utility.__file__ = str(Path(root) / "models/archs/arch_util.py")
    exec(compile(sources["models/archs/arch_util.py"], utility.__file__, "exec"), utility.__dict__)
    architecture = types.ModuleType("_seer_stcd_architecture")
    architecture.__file__ = str(Path(root) / "models/archs/low_light_transformer42.py")
    architecture.arch_util = utility
    adapted = adapt_architecture(sources["models/archs/low_light_transformer42.py"])
    exec(compile(adapted, architecture.__file__, "exec"), architecture.__dict__)
    model = architecture.low_light_transformer(
        nf=64, nframes=5, groups=8, front_RBs=1, back_RBs=1,
        center=None, predeblur=True, HR_in=True, w_TSA=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Expected the released raw STCD tensor state dictionary")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed DataParallel prefixes in STCD checkpoint")
    if all(prefixed):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.float().requires_grad_(False).to(device).eval()
    provenance = {
        "label": REFERENCE_LABEL,
        "source_domain": "SDSD indoor only, as documented by the official release",
        "source_repo": "https://github.com/xiaogang00/LLVE_STCD",
        "public_checkpoint": "https://drive.google.com/file/d/1sUzhwKWxJOjITFL8ypb3UQeofz0KI7E4/view",
        "public_archive_name": "pretrained_model_LLVE_STCD.zip",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "public_archive_byte_identity_reverified": False,
        "checkpoint_origin_note": "Existing archived official-release file; no new public archive download/hash manifest",
        "source_root": str(Path(root).resolve()),
        "upstream_source_sha256": dict(SOURCE_SHA256),
        "in_memory_adaptation": "Remove exactly two constructor .to('cuda') calls; inject isolated pinned arch_util; no network-operation changes",
        "adapted_architecture_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "state_sha256": state_digest(model.state_dict()),
        "parameters": sum(value.numel() for value in model.parameters()),
        "load_state_dict_strict": True,
        "precision": "float32; no autocast",
    }
    return model, provenance


def enhance_clip(model, clip_rgb, chunk=1):
    """GT-free (T,3,H,W) float RGB -> same shape, with last-frame duplication."""
    import torch
    if (clip_rgb.ndim != 4 or clip_rgb.shape[1] != 3 or clip_rgb.shape[0] < 2
            or min(clip_rgb.shape[-2:]) < 4 or any(size % 4 for size in clip_rgb.shape[-2:])
            or not isinstance(chunk, int) or chunk < 1):
        raise ValueError("STCD expects T>=2 RGB frames, spatial multiples of four and a positive chunk")
    if not torch.isfinite(clip_rgb).all() or clip_rgb.min() < 0 or clip_rgb.max() > 1:
        raise ValueError("STCD input must be finite RGB in [0,1]")
    with torch.no_grad():
        x = clip_rgb[:, [2, 1, 0]].float()
        neighbour = torch.cat((x[1:], x[-1:]), dim=0)
        outputs = []
        for start in range(0, len(x), chunk):
            result = model(x[start:start + chunk], neighbour[start:start + chunk])
            if not isinstance(result, (tuple, list)) or len(result) != 6:
                raise ValueError("Expected the six official STCD generator outputs")
            value = result[2]
            if value.shape != x[start:start + chunk].shape or not torch.isfinite(value).all():
                raise ValueError("Invalid STCD enhanced output")
            outputs.append(value)
        return torch.cat(outputs, dim=0)[:, [2, 1, 0]].float().clamp(0, 1)


def predict_uint8(model, low_u8, device, chunk):
    import torch
    low = torch.from_numpy(np.ascontiguousarray(low_u8)).float().permute(0, 3, 1, 2).div_(255).to(device)
    prediction = enhance_clip(model, low, chunk)
    return (prediction.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)


def _scene_condition(name):
    match = re.fullmatch(r"([^/\\]+)__(10|20)fps", name)
    if match is None or match.group(1) in (".", ".."):
        raise ValueError(f"Unsupported frozen BVI identifier: {name!r}")
    return match.group(1), match.group(2)


def validation_inventory(split_path, root):
    """Check whole frame-name lists before eval_e4's index-paired clip iterator."""
    entry = json.loads(Path(split_path).read_text())["BVI_RLV"]
    scenes = {}
    for split, count in (("train", 48), ("val", 16), ("test", 16)):
        names = entry[split]
        if len(names) != count or len(set(names)) != count:
            raise ValueError(f"Expected frozen BVI {split}{count}")
        pairs = [_scene_condition(name) for name in names]
        if len(set(pairs)) != count or any(value != 2 for value in Counter(scene for scene, _ in pairs).values()):
            raise ValueError(f"BVI {split}: expected both conditions of every scene")
        scenes[split] = {scene for scene, _ in pairs}
    if any(scenes[a] & scenes[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Frozen BVI splits overlap by scene")
    # This is the exact root remap performed by eval_e4.remap.
    base = Path(entry["root"].replace("/temp/zihaohe/seer_cvpr", str(root)))
    inventory = []
    for name in entry["val"]:
        scene, condition = _scene_condition(name)
        folders = [base / scene / f"{kind}_light_{condition}" for kind in ("low", "normal")]
        names_by_kind = [sorted(path.name for path in folder.iterdir()
                                if path.is_file() and path.name.lower().endswith(FRAME_EXTS))
                         for folder in folders]
        low, high = names_by_kind
        if len(low) < 16 or low != high:
            raise ValueError(f"Incomplete or differently named BVI paired frame lists: {name}")
        start = (len(low) - 16) // 2
        inventory.append({"name": f"BVI/{name}", "scene": scene, "condition": condition,
                          "paired_frame_count": len(low), "paired_frame_names": low,
                          "centre_start": start, "selected_frame_names": low[start:start + 16],
                          "directories": [str(folder) for folder in folders]})
    return inventory


def evaluate_sequences(sequences, predict, score, expected_names):
    """Prediction accepts only LQ; scoring is called strictly afterwards."""
    rows = []
    for name, low, gt, kind in sequences:
        if name not in expected_names or name in {row["name"] for row in rows}:
            raise ValueError(f"Unexpected or duplicate BVI validation sequence: {name}")
        if (kind != "video" or low.shape != (16, 256, 256, 3) or gt.shape != low.shape
                or low.dtype != np.uint8 or gt.dtype != np.uint8):
            raise ValueError(f"Expected paired centre16/crop256 uint8 RGB: {name}")
        prediction = predict(low)
        if prediction.shape != low.shape or prediction.dtype != np.uint8:
            raise ValueError("Prediction must match the RGB uint8 scoring protocol")
        result = score(prediction, gt)
        if not result or not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError(f"Nonfinite STCD score: {name}")
        rows.append({"name": name, **result})
        print(json.dumps(rows[-1]), flush=True)
    if [row["name"] for row in rows] != list(expected_names):
        raise ValueError("Expected complete, ordered coverage of all sixteen BVI validation sequences")
    return rows


def score_uint8(prediction, gt, device):
    import torch
    from eval_e4 import lpips_of, psnr_y_uint8, rgb_psnr_uint8
    p = torch.from_numpy(prediction.copy()).float().permute(0, 3, 1, 2).div_(255).to(device)
    g = torch.from_numpy(gt.copy()).float().permute(0, 3, 1, 2).div_(255).to(device)
    tlp = lpips_of(p[1:], p[:-1], device)
    gt_tlp = lpips_of(g[1:], g[:-1], device)
    return {"psnr_y": float(np.mean([psnr_y_uint8(a, b) for a, b in zip(prediction, gt)])),
            "psnr_rgb": float(np.mean([rgb_psnr_uint8(a, b) for a, b in zip(prediction, gt)])),
            "lpips": lpips_of(p, g, device), "tlp": tlp, "gt_tlp": gt_tlp,
            "tlp_gap": abs(tlp - gt_tlp)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stcd-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seer-root", type=Path, default=Path(os.environ.get("SEER_ROOT", "/temp/zihaohe/seer_cvpr")))
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="npu")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--cpu-smoke", action="store_true", help="Load strict state and run synthetic CPU shapes only; no BVI access")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.chunk < 1 or args.cpu_smoke and args.device != "cpu":
        raise ValueError("Positive chunk required; --cpu-smoke requires --device cpu")
    os.environ["SEER_ROOT"] = str(args.seer_root)
    sys.path.insert(0, str(CODE))
    sys.path.insert(0, str(CODE / "tools"))
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    device = torch.device("cpu" if args.device == "cpu" else args.device + ":0")
    model, provenance = load_stcd(args.stcd_root, args.checkpoint, device)
    if args.cpu_smoke:
        generator = torch.Generator().manual_seed(0)
        low = torch.rand(3, 3, 32, 32, generator=generator)
        output = enhance_clip(model, low, chunk=1)
        torch.testing.assert_close(output, enhance_clip(model, low, chunk=2), rtol=1e-4, atol=1e-5)
        result = {"purpose": "synthetic CPU strict-load/shape smoke; no data evaluation",
                  "shape": list(output.shape), "finite": bool(torch.isfinite(output).all()),
                  "provenance": provenance}
    else:
        import eval_e4
        split_path = args.split_path or args.seer_root / "datasets/splits_cvpr.json"
        inventory = validation_inventory(split_path, args.seer_root)
        eval_e4.SPLITS = str(split_path)
        expected = [row["name"] for row in inventory]
        rows = evaluate_sequences(
            eval_e4.held_out_sequences("bvi", "val", 16, 256),
            lambda low: predict_uint8(model, low, device, args.chunk),
            lambda prediction, gt: score_uint8(prediction, gt, device), expected)
        import common.eval_lib as eval_lib
        result = {
            "method": REFERENCE_LABEL, "domain": "bvi", "split": "val", "n": len(rows),
            "protocol": "frozen BVI VAL16; centre16/centre256; rounded uint8 RGB/Y PSNR, pyiqa LPIPS and consecutive-frame tLP",
            "context": "two frames: out_t=netG(BGR_t,BGR_min(t+1,15))[2]; no extra context beyond centre16; last neighbour duplicated",
            "gt_policy": "No GT input, brightness fitting, adaptation, checkpoint selection or oracle weighting; GT used only for post-prediction scores",
            "comparison_scope": "SDSD-indoor-only released source reference; not a matched-union136 training or SOTA comparison",
            "per_sequence": rows,
            "means": {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "name"},
            "validation_pair_inventory": inventory, "split_sha256": digest(split_path),
            "provenance": provenance,
            "scoring_source_sha256": {str(path): digest(path) for path in (
                Path(__file__), Path(eval_e4.__file__), CODE / "data/common.py", Path(eval_lib.__file__))},
            "runtime": {"torch": str(torch.__version__), "device": str(device), "chunk": args.chunk},
        }
    if state_digest(model.state_dict()) != provenance["state_sha256"]:
        raise AssertionError("STCD inference changed the frozen state")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
