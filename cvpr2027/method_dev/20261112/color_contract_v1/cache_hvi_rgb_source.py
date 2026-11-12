"""Frozen selective-LQ source export with one explicit native BGR -> HVI RGB boundary."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

LEGACY_SHA = "26c0c36656ad7b7d3b0ce04275578a45286c27931a476c9dcfb1ce1981e69db2"
SOURCE_SHA = "0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0"
SUMMARY_SHA = "63df74b887dc66543137569d58b6fd57c6fbdb931c6f4c6077c53b77585031ae"
COLOR_CONTRACT_SHA = "7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60"
BOUNDARY = {"source_cache": "BGR", "hvi_input": "RGB", "hvi_output": "RGB",
            "input_permutation": [2, 1, 0], "output_permutation": [0, 1, 2]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_legacy(path):
    """Do not fork the previously frozen selective reader/model implementation."""
    path = Path(path)
    require(digest(path) == LEGACY_SHA, "Frozen legacy adapter hash mismatch")
    name = "seer_frozen_lq_only_hvi"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def canonical_rgb(native):
    """This function has one declared BGR input; there is no order inference."""
    import torch
    require(isinstance(native, torch.Tensor) and native.ndim == 5 and native.shape[:2] == (1, 3),
            "Expected source B,3,T,H,W tensor")
    require(native.dtype == torch.float32, "Expected float32 source LQ")
    require(bool(torch.isfinite(native).all()) and float(native.min()) >= 0 and float(native.max()) <= 1,
            "Expected finite unit-range source LQ")
    return native.index_select(1, torch.tensor([2, 1, 0], device=native.device)).contiguous()


def read_rgb_lq(path, legacy, *, expected_shape=(1, 3, 16, 256, 256), archive_class=None):
    """GT storage firewall is the exact legacy function, with test instrumentation."""
    options = {} if archive_class is None else {"archive_class": archive_class}
    name, scene, native, lineage = legacy.read_lq_only(path, expected_shape, **options)
    rgb = canonical_rgb(native)
    require(legacy.tensor_digest(native) == lineage["lq_tensor_sha256"], "Native input mutated")
    lineage.update({"native_lq_tensor_sha256": lineage["lq_tensor_sha256"],
                    "rgb_lq_tensor_sha256": legacy.tensor_digest(rgb),
                    "native_source_color": "BGR", "input_color": "RGB", "output_color": "RGB",
                    "input_swap": "source native BGR -> canonical HVI RGB",
                    "channel_boundary": BOUNDARY})
    return name, scene, rgb, lineage


def validate_evidence(path):
    """A source-color closure receipt is a data dependency, not a score selection."""
    path = Path(path)
    value = json.loads(path.read_text())
    require(value.get("status") == "closed", "Native BGR evidence is not closed")
    require(value.get("native_source_color") == "BGR", "Evidence does not establish BGR")
    require(value.get("source_provenance_sha256") == SOURCE_SHA, "Evidence source differs")
    proofs = value.get("proof_files")
    require(isinstance(proofs, list) and len(proofs) > 0, "Missing BGR closure proof files")
    for proof in proofs:
        require(Path(proof["path"]).is_absolute(), "Proof file must have an absolute path")
        require(digest(proof["path"]) == proof["sha256"], "BGR proof file hash mismatch")
    return {"bgr_evidence_file": str(path.resolve()), "bgr_evidence_sha256": digest(path),
            "bgr_evidence_status": "closed", "bgr_proof_files": proofs}


def validate_release(release):
    release = Path(release)
    manifest = json.loads((release / "export_release_manifest.json").read_text())
    for relative, sha in manifest["files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Unsafe release manifest path")
        require(digest(release / relative) == sha, "Release file changed: " + relative)
    require(digest(release / "cache_hvi_rgb_source.py") == digest(__file__), "Executing different export script")
    protocol = json.loads((release / "source_export_protocol.json").read_text())
    require(protocol["source_provenance_sha256"] == SOURCE_SHA, "Protocol source mismatch")
    require(protocol["source_summary_sha256"] == SUMMARY_SHA, "Protocol input list mismatch")
    require(protocol["channel_boundary"] == BOUNDARY, "Channel boundary mismatch")
    require(protocol["color_contract_sha256"] == COLOR_CONTRACT_SHA,
            "Wrong shared boundary reference")
    require(digest(release / "dependencies/color_contract.py") == COLOR_CONTRACT_SHA,
            "Shared boundary reference changed")
    legacy = load_legacy(release / "dependencies/cache_hvi_source.py")
    require(protocol["weight_sha256"] == legacy.WEIGHT_SHA, "Protocol checkpoint mismatch")
    require(protocol["model_state_sha256"] == legacy.STATE_SHA, "Protocol state mismatch")
    return legacy, protocol


def source_rows(source, legacy):
    source = Path(source)
    require(digest(source / "provenance.json") == SOURCE_SHA, "Source provenance changed")
    require(digest(source / "cache_summary.json") == SUMMARY_SHA, "Source summary changed")
    origin = json.loads((source / "provenance.json").read_text())
    summary = json.loads((source / "cache_summary.json").read_text())
    rows = summary["per_sequence"]
    names = [row["name"] for row in rows]
    require(summary["n"] == len(names) == len(set(names)) == 136, "Not the fixed TRAIN136")
    require(origin["domain"] == "sdsd" and names == origin["source_names"], "Source split changed")
    require(len({row["scene"] for row in rows}) == 68, "Source scene count changed")
    return rows


def verify_cache(out, legacy):
    """Reopen only the NEW HVI base payloads, never original source/GT tensors."""
    import torch
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    require(manifest.get("complete") is True and manifest["n"] == 136, "Incomplete export")
    require(manifest["model_state_unchanged"] is True, "Model state changed")
    require(manifest["channel_boundary"] == BOUNDARY, "Wrong output contract")
    files = sorted(p.name for p in out.glob("*.pt"))
    require(files == sorted(row["file"] for row in manifest["per_sequence"]), "Cache coverage mismatch")
    names = set()
    for row in manifest["per_sequence"]:
        path = out / row["file"]
        require(digest(path) == row["file_sha256"], "New cache payload changed")
        value = torch.load(path, map_location="cpu", weights_only=False)
        require(set(value) == {"name", "scene", "base", "lineage"}, "Forbidden output key")
        require(value["name"] == row["name"] and value["scene"] == row["scene"], "Wrong identity")
        base, lineage = value["base"], value["lineage"]
        require(base.shape == (1, 3, 16, 256, 256) and base.dtype == torch.float32, "Wrong base tensor")
        require(bool(torch.isfinite(base).all()) and float(base.min()) >= 0 and float(base.max()) <= 1,
                "Invalid base values")
        require(legacy.tensor_digest(base) == row["base_tensor_sha256"] == lineage["base_tensor_sha256"],
                "New base tensor digest differs")
        for key in ("source_cache_sha256", "native_lq_tensor_sha256", "rgb_lq_tensor_sha256"):
            require(lineage[key] == row[key], "Lineage mismatch: " + key)
        require(lineage["channel_boundary"] == BOUNDARY and lineage["output_color"] == "RGB", "Wrong lineage contract")
        require(lineage["weight_sha256"] == legacy.WEIGHT_SHA and lineage["model_state_sha256"] == legacy.STATE_SHA,
                "Wrong HVI checkpoint")
        require(lineage["source_provenance_sha256"] == SOURCE_SHA, "Wrong source lineage")
        require(sum("/data/" in name for name in lineage["archive_entries_read"]) == 1, "Non-LQ source storage")
        names.add(value["name"])
    require(len(names) == 136, "Duplicate names")
    return {"status": "passed", "n": 136, "base_payloads_verified": 136, "original_source_tensor_reads": 0,
            "manifest_sha256": digest(out / "manifest.json"), "complete": True,
            "channel_boundary": BOUNDARY, "quality_scores_computed": False}


def preflight(release, legacy, protocol):
    import torch
    model = legacy.load_hvi(release, torch.device("cpu"))
    before = legacy.state_digest(model.state_dict())
    native = torch.linspace(0, 1, 3 * 2 * 64 * 64).reshape(1, 3, 2, 64, 64)
    rgb = canonical_rgb(native)
    base = legacy.predict_float_base(model, rgb, torch.device("cpu"))
    require(base.shape == rgb.shape and bool(torch.isfinite(base).all()), "Synthetic preflight failed")
    require(legacy.state_digest(model.state_dict()) == before == legacy.STATE_SHA, "Preflight changed model")
    return {"status": "passed", "device": "cpu", "strict_state_keys": len(model.state_dict()),
            "weight_sha256": legacy.WEIGHT_SHA, "model_state_sha256": before, "model_state_unchanged": True,
            "synthetic_input_shape": list(native.shape), "native_synthetic_sha256": legacy.tensor_digest(native),
            "rgb_synthetic_sha256": legacy.tensor_digest(rgb), "base_synthetic_sha256": legacy.tensor_digest(base),
            "source_real_inputs_read": 0, "gt_storage_entries_read": 0, "quality_scores_computed": False,
            "torch_version": torch.__version__, "channel_boundary": BOUNDARY,
            "bgr_evidence_required_before_export": True}


def export(release, out, legacy, protocol, evidence, device_name):
    import torch
    if device_name == "npu":
        import torch_npu
    device = torch.device("cpu" if device_name == "cpu" else "npu:0")
    source, out = Path(protocol["source_cache"]), Path(out)
    require(str(out) == protocol["output"], "Only the fixed new output path is allowed")
    if out.exists():
        raise FileExistsError(out)
    rows = source_rows(source, legacy)
    model = legacy.load_hvi(release, device)
    before = legacy.state_digest(model.state_dict())
    began = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    result_rows = []
    for index, row in enumerate(rows):
        source_path = source / row["cache_path"]
        require(digest(source_path) == row["cache_sha256"], "Source payload changed")
        name, scene, rgb, lineage = read_rgb_lq(source_path, legacy)
        require(name == row["name"] and scene == row["scene"], "Source identity mismatch")
        base = legacy.predict_float_base(model, rgb, device)
        require(base.shape == rgb.shape, "Output shape mismatch")
        lineage.update({"source_cache_file": str(source_path), "source_cache_sha256": row["cache_sha256"],
            "source_provenance_sha256": SOURCE_SHA, "source_summary_sha256": SUMMARY_SHA,
            "weight_sha256": legacy.WEIGHT_SHA, "model_state_sha256": before,
            "config_sha256": digest(release / "source_export_protocol.json"),
            "release_manifest_sha256": digest(release / "export_release_manifest.json"),
            "adapter_sha256": digest(__file__), "legacy_adapter_sha256": LEGACY_SHA,
            "color_contract_sha256": COLOR_CONTRACT_SHA,
            "bgr_evidence_sha256": evidence["bgr_evidence_sha256"],
            "source_split": "original SDSD TRAIN136", "prediction_clamped": [0, 1],
            "prediction_rounded": False, "gt_storage_entries_read": 0, "target_inputs_read": 0})
        payload = legacy.output_payload(name, scene, base, lineage)
        target = out / f"hvi_rgb_{index:03d}.pt"
        with target.open("xb") as stream:
            torch.save(payload, stream)
        result_rows.append({"name": name, "scene": scene, "file": target.name,
            "file_sha256": digest(target), "base_tensor_sha256": payload["lineage"]["base_tensor_sha256"],
            "source_cache_sha256": row["cache_sha256"], "native_lq_tensor_sha256": lineage["native_lq_tensor_sha256"],
            "rgb_lq_tensor_sha256": lineage["rgb_lq_tensor_sha256"], "shape": list(base.shape),
            "dtype": "float32", "finite_range_verified": True})
        print(json.dumps({"completed": index + 1, "total": 136, "name": name,
                          "output_sha256": result_rows[-1]["file_sha256"]}), flush=True)
    require(legacy.state_digest(model.state_dict()) == before == legacy.STATE_SHA, "Model state changed")
    report = {"purpose": protocol["purpose"], "complete": True, "n": 136, "scenes": 68,
        "per_sequence": result_rows, "source_provenance_sha256": SOURCE_SHA, "source_summary_sha256": SUMMARY_SHA,
        "native_source_color": "BGR", "input_color": "RGB", "output_color": "RGB", "channel_boundary": BOUNDARY,
        "input_swap": "source native BGR -> canonical HVI RGB", "model_state_unchanged": True,
        "model_state_sha256": before, "weight_sha256": legacy.WEIGHT_SHA, "strict_state_keys": len(model.state_dict()),
        "config_sha256": digest(release / "source_export_protocol.json"),
        "release_manifest_sha256": digest(release / "export_release_manifest.json"),
        "adapter_sha256": digest(__file__), "legacy_adapter_sha256": LEGACY_SHA, **evidence,
        "color_contract_sha256": COLOR_CONTRACT_SHA,
        "gt_storage_entries_read": 0, "target_inputs_read": 0, "quality_scores_computed": False,
        "source_split": "original SDSD TRAIN136", "device": device_name, "torch_version": torch.__version__,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - began}
    write_new(out / "manifest.json", report)
    return verify_cache(out, legacy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "export", "verify"), required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--bgr-evidence", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    import torch
    torch.set_num_threads(4)
    legacy, protocol = validate_release(args.release)
    if args.mode == "preflight":
        require(args.device == "cpu", "Preflight is explicitly CPU only")
        report = preflight(args.release, legacy, protocol)
    elif args.mode == "verify":
        report = verify_cache(protocol["output"], legacy)
    else:
        require(args.bgr_evidence is not None, "BGR closure receipt required before export")
        evidence = validate_evidence(args.bgr_evidence)
        report = export(args.release, protocol["output"], legacy, protocol, evidence, args.device)
    write_new(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
