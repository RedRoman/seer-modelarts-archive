"""Frozen color-contract repair. Source parity never opens target/GT storage."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHAS = {
    "pretrained": "5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740",
    "none": "fa3c0c9e720a28b56f01069f7a2cd407bc5b4d752a2a1d4482f1bd63557b5516"}
COLOR_SHA = "7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60"
RECEIPT_SHA = "fcec10f792569e1f1202c7ed4d29e7e774fbc7ea0ba5198cdbd454f9ed174a70"
ENCODER_SHA = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
SPLIT_SHA = "fb6be5ea78ef73f5cc9db90bca75813bf597275f7609fe39d82f5b2f883e9ad0"
MAX_ABS, MEAN_ABS = 1e-4, 1e-5
INDICES = (108, 109)
SCENE = "SDSD_outdoor/MVI_0975"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_release(release):
    manifest = read_json(release / "release_manifest.json")
    for relative, expected in manifest["files"].items():
        path = release / relative
        if not path.resolve().is_relative_to(release.resolve()) or digest(path) != expected:
            raise ValueError(f"Release file differs: {relative}")
    for path, expected in manifest["external_files"].items():
        if digest(path) != expected:
            raise ValueError(f"External dependency differs: {path}")
    receipt = read_json(release / "phase/source_color_receipt.json")
    if (receipt["status"] != "closed" or receipt["native_source_color"] != "BGR"
            or receipt["verified"] is not True):
        raise ValueError("Closed BGR source proof required")
    for proof in receipt["proof_files"]:
        if digest(proof["path"]) != proof["sha256"]:
            raise ValueError("Original source color proof changed")
    if digest(release / "phase/color_contract.py") != COLOR_SHA:
        raise ValueError("Color adapter differs")
    if digest(release / "phase/source_color_receipt.json") != RECEIPT_SHA:
        raise ValueError("Color receipt differs")
    return manifest


def validate_source_row(row, index, fit):
    expected_name = SCENE + ("_2" if index == 109 else "")
    if index not in INDICES or row["scene"] != SCENE or row["name"] != expected_name:
        raise ValueError("Only the fixed two FIT clips are authorized")
    if SCENE not in fit["fit"] or SCENE in fit["internal_holdout"]:
        raise ValueError("Selected source scene is outside FIT")


def validate_authorization(path, protocol_sha, source_results):
    gate = read_json(path)
    if gate.get("authorized_stage") != "target_bvi_val16" or gate.get("protocol_sha256") != protocol_sha:
        raise ValueError("Separate exact-protocol target authorization required")
    if set(source_results) != set(SHAS):
        raise ValueError("Both fixed arms must pass source parity first")
    for arm, path in source_results.items():
        row = read_json(path)
        if (not row.get("passed") or row.get("arm") != arm or row.get("target_pixel_reads") != 0
                or row.get("gt_storage_reads") != 0 or row.get("protocol_sha256") != protocol_sha
                or gate["source_parity_sha256"].get(arm) != digest(path)):
            raise ValueError(f"Source gate mismatch: {arm}")


def capture_wrapper(predict_native, native_low):
    """One actual forward, with exact permutation identity checked on its output."""
    import torch
    from color_contract import convert_video, predict_native_as_rgb
    low_rgb = convert_video(native_low, source_order="BGR", target_order="RGB")
    captured = {}
    def capture(low):
        if not torch.equal(low, native_low):
            raise AssertionError("Whole-pipeline native input changed")
        captured["native"] = predict_native(low)
        return captured["native"]
    rgb = predict_native_as_rgb(capture, low_rgb, native_order="BGR")
    direct_rgb = convert_video(captured["native"], source_order="BGR", target_order="RGB")
    if not torch.equal(rgb, direct_rgb):
        raise AssertionError("Wrapper is not an exact output permutation")
    return rgb, captured["native"]


def check_old_base(native, old):
    difference = (native.detach().float().cpu() - old).abs()
    result = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
              "max_abs_limit": MAX_ABS, "mean_abs_limit": MEAN_ABS}
    result["passed"] = result["max_abs"] <= MAX_ABS and result["mean_abs"] <= MEAN_ABS
    return result


def load_native(config, arm, device):
    import torch
    from eval_e4 import build_model
    from cache_source_corrections import check_base_args, backbone_tokens
    from train_method_pilot import state_digest
    checkpoint = Path(config["arms"][arm]["checkpoint"])
    origin = read_json(config["arms"][arm]["cache"] + "/provenance.json")
    if digest(checkpoint) != SHAS[arm] or origin["base_checkpoint_sha256"] != SHAS[arm]:
        raise ValueError("Checkpoint differs from source lineage")
    loads = []
    original_load = torch.nn.Module.load_state_dict
    def observed_load(module, state, *args, **kwargs):
        result = original_load(module, state, *args, **kwargs)
        loads.append({"module": module.__class__.__name__, "missing": list(result.missing_keys),
                      "unexpected": list(result.unexpected_keys)})
        return result
    torch.nn.Module.load_state_dict = observed_load
    try:
        model, backbone, refiner, exposure, actual_arm, length, saved = build_model(checkpoint, device)
    finally:
        torch.nn.Module.load_state_dict = original_load
    check_base_args(saved, actual_arm, length)
    if actual_arm != arm or refiner is not None or exposure is not None:
        raise ValueError("Unexpected saved base architecture")
    if model.eval_reconstruction != origin["base_reconstruction"]:
        raise ValueError("Saved reconstruction differs")
    if state_digest(model.state_dict()) != origin["base_state_sha256"]:
        raise ValueError("Pixel state differs")
    for load in loads:
        if load["missing"]:
            raise ValueError(f"Missing model keys: {load}")
        allowed = {"patch_embed_img.proj.weight", "patch_embed_img.proj.bias"} if load["module"] == "VisionTransformer" else set()
        if set(load["unexpected"]) - allowed:
            raise ValueError(f"Unexpected used model keys: {load}")
    if (backbone is None) != (arm == "none"):
        raise ValueError("Frozen token arm differs")
    model.eval().requires_grad_(False)
    if backbone is not None:
        backbone.eval().requires_grad_(False)
    pixel_before = state_digest(model.state_dict())
    encoder_before = state_digest(backbone.state_dict()) if backbone is not None else None
    def predict_native(low):
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            tokens = backbone_tokens(backbone, low)
            return model(low, tokens)["pred_pixels"].float().clamp(0, 1)
    def unchanged():
        return (state_digest(model.state_dict()) == pixel_before
                and (backbone is None or state_digest(backbone.state_dict()) == encoder_before))
    return predict_native, unchanged, {"strict_pixel_keys": True, "load_key_receipts": loads,
        "pixel_state_sha256": pixel_before, "encoder_state_sha256": encoder_before,
        "reconstruction": model.eval_reconstruction, "saved_args": saved}


def source_parity(config, arm, predict, device):
    import torch
    from restricted_source_reader import read_allowed_tensor, tensor_digest
    cache = Path(config["arms"][arm]["cache"])
    summary = read_json(cache / "cache_summary.json")
    fit = read_json(HERE / "source_fit_partition.json")["split"]
    rows = []
    for index in INDICES:
        row = summary["per_sequence"][index]
        validate_source_row(row, index, fit)
        path = cache / row["cache_path"]
        if digest(path) != row["cache_sha256"]:
            raise ValueError("Old source cache differs")
        name, scene, low, lq_lineage = read_allowed_tensor(path, "lq")
        old_name, old_scene, old_base, base_lineage = read_allowed_tensor(path, "base")
        if (name, scene) != (row["name"], SCENE) or (old_name, old_scene) != (name, scene):
            raise ValueError("Source storage identity differs")
        low = low.to(device)
        rgb, native = capture_wrapper(predict, low)
        parity = check_old_base(native, old_base)
        rows.append({"index": index, "name": name, "scene": scene,
            "cache_path": str(path), "cache_sha256": row["cache_sha256"],
            "lq_read": lq_lineage, "base_read": base_lineage,
            "native_lq_sha256": tensor_digest(low), "prediction_rgb_sha256": tensor_digest(rgb),
            "wrapper_input_exact": True, "wrapper_output_permutation_exact": True,
            "old_base_parity": parity})
        print(json.dumps(rows[-1]), flush=True)
    return {"passed": all(row["old_base_parity"]["passed"] for row in rows), "n": 2,
        "scene": SCENE, "per_sequence": rows, "gt_storage_reads": 0,
        "target_pixel_reads": 0, "quality_scores_computed": False}


def target_score(predict, device):
    import numpy as np
    import torch
    from color_contract import predict_native_as_rgb
    from eval_e4 import SPLITS, held_out_sequences
    from eval_correction_selector import score_prediction
    from data.common import list_images
    from data.planned_bvi import _dataset_root, parse_sequence, validate_split
    entry = read_json(SPLITS)["BVI_RLV"]
    validate_split(entry)
    root = _dataset_root(entry["root"], os.environ["SEER_ROOT"])
    inventory = []
    for name in entry["val"]:
        scene, condition = parse_sequence(name)
        low = list_images(str(root / scene / f"low_light_{condition}"))
        gt = list_images(str(root / scene / f"normal_light_{condition}"))
        if len(low) < 16 or len(low) != len(gt) or [Path(p).name for p in low] != [Path(p).name for p in gt]:
            raise ValueError("Target complete filename pairing differs")
        inventory.append({"name": "BVI/" + name, "paired_frames": len(low), "centre_start": (len(low)-16)//2})
    rows = []
    for name, low_u8, gt_u8, kind in held_out_sequences("bvi", "val", 16, 256):
        if kind != "video" or low_u8.shape != (16, 256, 256, 3) or gt_u8.shape != low_u8.shape:
            raise ValueError("Target frozen shape differs")
        low = torch.from_numpy(low_u8.copy()).float().div_(255).permute(3, 0, 1, 2)[None].to(device)
        prediction = predict_native_as_rgb(predict, low, native_order="BGR")
        scores = score_prediction(prediction, gt_u8, device)
        if not all(np.isfinite(value) for value in scores.values()):
            raise ValueError("Nonfinite target score")
        rows.append({"name": name, "metrics": scores})
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) != 16 or sorted(row["name"] for row in rows) != sorted(row["name"] for row in inventory):
        raise ValueError("Incomplete target VAL16")
    return {"passed": True, "n": 16, "per_sequence": rows, "pair_inventory": inventory,
        "means": {key: float(np.mean([row["metrics"][key] for row in rows])) for key in rows[0]["metrics"]},
        "gt_mean_alignment": False, "selection": "none", "predicted_rgb_saved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--stage", choices=("source", "target"), required=True)
    parser.add_argument("--arm", choices=tuple(SHAS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--source-pretrained", type=Path)
    parser.add_argument("--source-none", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    release = args.release.resolve()
    manifest = verify_release(release)
    protocol_sha = digest(release / "phase/PLAN.md")
    if args.stage == "target":
        if not all((args.authorization, args.source_pretrained, args.source_none)):
            raise ValueError("Target requires separate authorization and both source receipts")
        validate_authorization(args.authorization, protocol_sha,
            {"pretrained": args.source_pretrained, "none": args.source_none})
    config = read_json(release / "phase/config.json")
    sys.path[:0] = [str(release / "phase"), str(release / "code_full"), str(release / "code_full/tools")]
    physical = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if physical not in ("6", "7"):
        raise ValueError("Only an explicitly free physical NPU6/7 is allowed")
    smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    if f"No running processes found in NPU {physical}" not in smi:
        raise ValueError("Selected physical NPU has another process")
    import torch
    import torch_npu  # noqa: F401
    torch.set_num_threads(4)
    device = torch.device("npu:0")
    predict, unchanged, model_receipt = load_native(config, args.arm, device)
    result = source_parity(config, args.arm, predict, device) if args.stage == "source" else target_score(predict, device)
    state_unchanged = unchanged()
    result.update({"stage": args.stage, "arm": args.arm, "state_unchanged": state_unchanged,
        "model": model_receipt, "protocol_sha256": protocol_sha,
        "release_manifest_sha256": digest(release / "release_manifest.json"),
        "checkpoint_sha256": SHAS[args.arm], "color_receipt_sha256": RECEIPT_SHA,
        "color_helper_sha256": COLOR_SHA, "runtime_source_sha256": manifest["files"],
        "external_sha256": manifest["external_files"], "npu_before": smi,
        "physical_npu": int(physical), "cpu_threads": torch.get_num_threads(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "numpy", "pyiqa")}})
    result["passed"] = result["passed"] and state_unchanged
    write_new(args.out, result)
    if not result["passed"]:
        raise SystemExit("Preserved failed source parity; target forbidden")


if __name__ == "__main__":
    main()
