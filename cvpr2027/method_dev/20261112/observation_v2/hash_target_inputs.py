"""Read-only BVI VAL16 selected-file and exact scoring-input fingerprints.

Only 16 selected LQ and 16 selected GT frames per VAL sequence are opened.
No model, Torch import, score, TEST image, recursive dataset walk or mutation.
The pinned scoring loader/crop and STCD inventory functions run in isolated
NumPy/imageio namespaces, with their module-level side effects excluded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

METRIC_SHA = "a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d"
STCD_SHA = "420fdc33046c90bfda486df3162336f1da56c52562c719d41be102c7bbf63545"
DATA_SHA = "6041b43e832ab67def75b9e7285927dddd77d45d0bc25945bb600167b04e4ded"
SPLIT_SHA = "fb6be5ea78ef73f5cc9db90bca75813bf597275f7609fe39d82f5b2f883e9ad0"
FRAME_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")
CACHE_ROOT = Path("/cache")
MAX_RAW_BYTES = 20_000_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Refusing existing output file")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def new_output(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(CACHE_ROOT)
            and path.resolve().is_relative_to(CACHE_ROOT.resolve()), "Output must resolve under /cache")
    require(not path.exists() and not path.is_symlink(), "Use a new exclusive audit directory")
    path.mkdir(parents=True, exist_ok=False)
    return path


def definitions(path, names, namespace):
    source = Path(path).read_text()
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names), "Pinned function definitions differ")
    fingerprints = {n.name: hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest() for n in nodes}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return fingerprints


def primitives(code, metric_lib):
    # imageio is exactly the pinned scorer's decoder, rather than a substitute
    # PIL implementation with potentially different format/range behavior.
    import imageio.v2 as imageio
    code, metric_lib = Path(code), Path(metric_lib)
    paths = {"metric_library": (metric_lib, METRIC_SHA),
             "inventory_helper": (code / "tools/eval_stcd_transfer.py", STCD_SHA),
             "frame_listing_source": (code / "data/common.py", DATA_SHA)}
    for label, (path, expected) in paths.items():
        require(digest(path) == expected, f"Pinned source differs: {label}")
    ns = {"np": np, "imageio": imageio, "json": json, "Path": Path, "re": re,
          "Counter": Counter, "FRAME_EXTS": FRAME_EXTS}
    metric_functions = definitions(metric_lib, ("_load_img_any", "center_crop"), ns)
    inventory_functions = definitions(code / "tools/eval_stcd_transfer.py",
                                      ("_scene_condition", "validation_inventory"), ns)
    provenance = {label: {"path": str(path.resolve()), "sha256": expected} for label, (path, expected) in paths.items()}
    provenance["function_sha256"] = {**metric_functions, **inventory_functions}
    provenance["imageio_version"] = __import__("imageio").__version__
    return ns, provenance


def selection(split, seer_root, ns):
    split, seer_root = Path(split), Path(seer_root)
    require(digest(split) == SPLIT_SHA, "Frozen split SHA differs")
    inventory = ns["validation_inventory"](split, seer_root)
    require(len(inventory) == 16 and len({row["scene"] for row in inventory}) == 8, "Need VAL16/8scene")
    paths, summary = [], []
    for row in inventory:
        require(len(row["selected_frame_names"]) == 16, "Need selected centre16")
        selected = [Path(directory) / name for directory in row["directories"] for name in row["selected_frame_names"]]
        paths.extend(selected)
        summary.append({"name": row["name"], "scene": row["scene"], "condition": row["condition"],
                        "paired_frame_count": row["paired_frame_count"], "centre_start": row["centre_start"],
                        "paired_frame_names_sha256": json_digest(row["paired_frame_names"]),
                        "selected_frame_names": row["selected_frame_names"], "directories": row["directories"],
                        "selected_raw_bytes": sum(path.stat().st_size for path in selected)})
    require(len(paths) == len(set(paths)) == 512, "Expected exactly512 distinct selected original files")
    total = sum(path.stat().st_size for path in paths)
    require(total <= MAX_RAW_BYTES, f"Selected originals exceed20GB; stop before image reads: {total}")
    return inventory, summary, total


def _stat_identity(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def capture_content(split, seer_root, code, metric_lib):
    ns, source = primitives(code, metric_lib)
    inventory, summary, total = selection(split, seer_root, ns)
    clips, opened = [], []
    for item in inventory:
        roles = {}
        for role, directory in zip(("lq", "gt"), item["directories"]):
            records, clip_hash = [], hashlib.sha256()
            for clip_index, name in enumerate(item["selected_frame_names"]):
                path = Path(directory) / name
                before = _stat_identity(path)
                file_sha = digest(path)
                loaded = ns["_load_img_any"](str(path))
                require(loaded.ndim == 3 and loaded.shape[-1] == 3 and loaded.dtype == np.uint8,
                        "Pinned loader did not produce RGB uint8")
                crop = np.ascontiguousarray(ns["center_crop"](loaded, 256))
                require(crop.shape == (256, 256, 3) and crop.dtype == np.uint8, "Scoring crop shape/dtype differs")
                require(_stat_identity(path) == before, "Source changed while hashing/loading")
                contents = crop.tobytes(order="C")
                clip_hash.update(contents)
                records.append({"clip_frame_index": clip_index, "basename": name, "path": str(path.absolute()),
                                "resolved_path": str(path.resolve()), "file_size_bytes": before[2], "file_sha256": file_sha,
                                "decoded_shape": list(loaded.shape), "decoded_dtype": str(loaded.dtype),
                                "center256_shape": [256, 256, 3], "center256_dtype": "uint8",
                                "center256_rgb_sha256": hashlib.sha256(contents).hexdigest()})
                opened.append(str(path.absolute()))
            roles[role] = {"frames": records, "scoring_tensor_shape": [16, 256, 256, 3],
                           "scoring_tensor_dtype": "uint8", "scoring_tensor_layout": "THWC contiguous RGB",
                           "scoring_tensor_sha256": clip_hash.hexdigest()}
        clips.append({"name": item["name"], "roles": roles})
        print(json.dumps({"event": "clip_hashed", "name": item["name"], "selected_files": 32}), flush=True)
    require(len(opened) == len(set(opened)) == 512, "Image-read coverage differs")
    content = {"selection": summary, "clips": clips}
    return {"n_sequences": 16, "n_scenes": 8, "n_selected_original_files": 512,
            "total_selected_original_bytes": total, "content": content, "content_sha256": json_digest(content),
            "primitive_sources": source}


def compare_content(reference, current):
    """Fail closed on both original bytes and actual evaluation tensors."""
    require(reference["primitive_sources"] == current["primitive_sources"], "Pinned decoder/inventory provenance changed")
    require(reference["content_sha256"] == json_digest(reference["content"]), "Reference content manifest is corrupt")
    changes = []
    old = {c["name"]: c for c in reference["content"]["clips"]}
    new = {c["name"]: c for c in current["content"]["clips"]}
    if reference["content"]["selection"] != current["content"]["selection"] or set(old) != set(new):
        changes.append({"kind": "selection_or_inventory"})
    for name in sorted(set(old) & set(new)):
        for role in ("lq", "gt"):
            a, b = old[name]["roles"][role], new[name]["roles"][role]
            if a != b:
                changes.append({"kind": "file_or_scoring_tensor", "name": name, "role": role,
                                "scoring_tensor_equal": a["scoring_tensor_sha256"] == b["scoring_tensor_sha256"]})
    require(bool(changes) == (reference["content_sha256"] != current["content_sha256"]), "Content comparison consistency guard")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--split", type=Path, required=True)
    capture.add_argument("--seer-root", type=Path, required=True)
    capture.add_argument("--code", type=Path, required=True)
    capture.add_argument("--metric-lib", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = new_output(args.out)
    started, clock = utc(), time.monotonic()
    if args.action == "capture":
        inputs = {"split": str(args.split.absolute()), "seer_root": str(args.seer_root.absolute()),
                  "code": str(args.code.absolute()), "metric_lib": str(args.metric_lib.absolute())}
        reference = None
    else:
        reference = json.loads(args.manifest.read_text())
        require(reference["schema_version"] == 1 and reference["action"] == "capture"
                and reference["auditor_sha256"] == digest(__file__), "Frozen reference/tool identity differs")
        inputs = reference["input_paths"]
    atomic_json(out / "started.json", {"action": args.action, "started_utc": started,
                                       "auditor_sha256": digest(__file__), "input_paths": inputs})
    current = capture_content(inputs["split"], inputs["seer_root"], inputs["code"], inputs["metric_lib"])
    result = {"schema_version": 1, "action": args.action, "started_utc": started, "completed_utc": utc(),
              "auditor_sha256": digest(__file__), "split_sha256": SPLIT_SHA, "input_paths": inputs,
              "scope": "BVI VAL16 centre16 LQ+GT only; no TEST images or scoring/model execution",
              "torch_imported": "torch" in sys.modules, "score_computed": False,
              "elapsed_seconds": time.monotonic() - clock, "numpy_version": str(np.__version__), **current}
    require(result["torch_imported"] is False, "Audit unexpectedly imported Torch")
    if reference is not None:
        changes = compare_content(reference, current)
        result.update(reference_manifest_sha256=digest(args.manifest), content_unchanged=not changes, changes=changes)
    path = out / "manifest.json"
    atomic_json(path, result)
    atomic_json(out / "manifest.sha256.json", {"manifest_sha256": digest(path), "content_sha256": result["content_sha256"]})
    print(json.dumps({"event": "complete", "action": args.action, "manifest_sha256": digest(path),
                      "content_sha256": result["content_sha256"], "selected_files": 512,
                      "selected_raw_bytes": current["total_selected_original_bytes"],
                      "content_unchanged": result.get("content_unchanged")}), flush=True)
    if reference is not None and changes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
