"""Base-only source transport and fixed RAFT adjacent-flow cache.

export-slim verifies the complete source TRAIN136 v1 cache before exporting
only base RGB. flows accepts that slim transport or the complete v1 cache.
GT tensors are never read by field, returned, scored or passed to RAFT. Every
new output directory must be absent and resolve under /cache, never SFS.
"""
from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import time

CODE = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path("/cache")
EXPECTED_CLIPS = 136
BASE_SHAPE = (1, 3, 16, 256, 256)
RAFT_WEIGHTS_SHA = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
DONOR_SOURCE_SHA = "7c989a8873101e6139af4cd2c81d0675e5c440024e177a1ae90c10544359fadd"
DONOR_FUNCTION_SHA = "92ea05233443480976a78604576a457ad4bc35731ae0c88f63f1993662c35125"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _scene(name):
    _require(isinstance(name, str) and re.fullmatch(r"SDSD_(?:indoor|outdoor)/[^/\\]+", name)
             and name.split("/")[1] not in (".", ".."), "Invalid source TRAIN name")
    return re.sub(r"_2$", "", name)


def _atomic_bytes(path, contents):
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), f"Refusing existing output: {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _atomic_tensor(path, value):
    import torch
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), "Refusing existing tensor output")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _new_output(path):
    path = Path(path)
    _require(path.is_absolute() and path.is_relative_to(CACHE_ROOT), "Output must be under /cache")
    _require(not path.exists() and not path.is_symlink(), "Output directory already exists")
    _require(path.resolve().is_relative_to(CACHE_ROOT.resolve()), "Output resolves outside /cache")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _check_origin(provenance, summary):
    names, plan, rows = provenance.get("source_names", []), provenance.get("plan", []), summary.get("per_sequence", [])
    _require(provenance.get("domain") == "sdsd" and summary.get("n") == EXPECTED_CLIPS
             and len(names) == EXPECTED_CLIPS and len(set(names)) == EXPECTED_CLIPS
             and len(plan) == EXPECTED_CLIPS and len(rows) == EXPECTED_CLIPS,
             "Require complete source TRAIN136 metadata")
    _require(summary.get("oracle_computed") is False and summary.get("base_state_unchanged") is True
             and summary.get("raft_state_unchanged") is True, "Require completed unmodified source cache")
    for key in ("split_sha256", "base_checkpoint_sha256", "base_state_sha256",
                "raft_weights_sha256", "raft_state_sha256"):
        _require(_sha(provenance.get(key)), f"Invalid original provenance SHA: {key}")
    _require(provenance["raft_weights_sha256"] == RAFT_WEIGHTS_SHA, "Original RAFT weights differ")
    args = provenance.get("args", {})
    _require(args.get("clip_len") == BASE_SHAPE[2] and args.get("crop") == BASE_SHAPE[-1]
             and args.get("offset") == 2 and args.get("flow_steps") == 12 and args.get("flow_batch") == 2,
             "Unexpected source v1 observation recipe")
    saved = provenance.get("base_saved_args", {})
    _require(saved.get("tag") in ("e17_small_pretrained_s0", "e17_small_none_s0")
             and saved.get("seed") == 0 and saved.get("iters") == 20000
             and saved.get("heldout") is None and saved.get("only") is None,
             "Require the source-only E17 final20k seed0 base")
    for i, (name, spec, row) in enumerate(zip(names, plan, rows)):
        _require(spec.get("sequence_index") == i and row.get("name") == name
                 and row.get("scene") == _scene(name) and row.get("cache_path") == f"clip_{i:03d}.pt"
                 and _sha(row.get("cache_sha256")), "Original cache order/name/scene/plan mismatch")


def base_manifest(directory):
    """Validate every source/slim file SHA before creating any output.

    Slim provenance embeds the original v1 provenance and summary, including
    the ordered original cache file hashes. No original video/cache is needed
    after transport. Per-file base tensor hashes are checked when loaded.
    """
    directory = Path(directory)
    provenance_path, summary_path = directory / "provenance.json", directory / "cache_summary.json"
    p, s = json.loads(provenance_path.read_text()), json.loads(summary_path.read_text())
    if p.get("kind") == "source_base_only_slim":
        _require(p.get("schema_version") == 1 and s.get("kind") == "source_base_only_slim"
                 and s.get("n") == EXPECTED_CLIPS, "Incomplete slim transport")
        origin = p["source_cache"]
        source_p, source_s = origin["provenance"], origin["summary"]
        for key in ("provenance_sha256", "summary_sha256"):
            _require(_sha(origin.get(key)), "Missing original metadata SHA")
        _check_origin(source_p, source_s)
        rows = s.get("per_sequence", [])
        expected = [f"base_{i:03d}.pt" for i in range(EXPECTED_CLIPS)]
        _require(len(rows) == EXPECTED_CLIPS and [r.get("cache_path") for r in rows] == expected,
                 "Slim file order differs")
        for i, row in enumerate(rows):
            old = source_s["per_sequence"][i]
            _require(row.get("name") == old["name"] and row.get("scene") == old["scene"]
                     and row.get("source_cache_sha256") == old["cache_sha256"]
                     and _sha(row.get("base_tensor_sha256")), "Slim source lineage differs")
        _require(s.get("provenance_sha256") == digest(provenance_path), "Slim provenance SHA differs")
        kind = "slim"
    else:
        _check_origin(p, s)
        source_p, source_s, rows = p, s, s["per_sequence"]
        origin = {"provenance": p, "summary": s, "provenance_sha256": digest(provenance_path),
                  "summary_sha256": digest(summary_path)}
        expected, kind = [f"clip_{i:03d}.pt" for i in range(EXPECTED_CLIPS)], "full"
    observed = sorted(path.name for path in directory.glob("base_*.pt" if kind == "slim" else "clip_*.pt"))
    _require(observed == expected, "Missing or unexpected source cache files")
    for filename, row in zip(expected, rows):
        path = directory / filename
        _require(path.is_file() and not path.is_symlink() and digest(path) == row.get("cache_sha256"),
                 f"Cache file SHA mismatch: {filename}")
    return {"directory": directory, "kind": kind, "rows": rows, "source_cache": origin,
            "input_provenance_sha256": digest(provenance_path), "input_summary_sha256": digest(summary_path)}


def tensor_digest(value):
    """SHA256 of contiguous CPU float32 bytes; shape/dtype are checked separately."""
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _base_tensor(base, *, fixed_shape=True):
    import torch
    _require(isinstance(base, torch.Tensor) and base.ndim == 5 and base.shape[0:2] == (1, 3),
             "base must be [1,3,T,H,W]")
    _require(base.dtype == torch.float32 and base.shape[2] >= 2 and base.shape[-2] % 8 == 0
             and base.shape[-1] % 8 == 0, "Need float32 base with T>=2 and RAFT-compatible dimensions")
    _require(not fixed_shape or tuple(base.shape) == BASE_SHAPE, "Unexpected source base shape")
    _require(bool(torch.isfinite(base).all()) and not bool(((base < 0) | (base > 1)).any()),
             "base must be finite RGB in [0,1]")


def load_base(manifest, index):
    """Read only identity, plan and the base field; ignore all other v1 tensors."""
    import torch
    row = manifest["rows"][index]
    path = manifest["directory"] / row["cache_path"]
    _require(digest(path) == row["cache_sha256"], "Input cache changed after validation")
    record = torch.load(path, map_location="cpu", weights_only=True)
    origin = manifest["source_cache"]
    expected_plan = origin["provenance"]["plan"][index]
    _require(record.get("name") == row["name"] and record.get("scene") == row["scene"]
             and record.get("plan") == expected_plan, "Cached base identity or plan differs")
    if manifest["kind"] == "slim":
        _require(set(record) == {"schema_version", "kind", "name", "scene", "plan", "source_cache_sha256",
                                  "base_tensor_sha256", "tensors"}
                 and record["schema_version"] == 1 and record["kind"] == "source_base_only_slim"
                 and set(record["tensors"]) == {"base"}, "Slim tensor schema must contain base only")
    # Deliberately no access to low, GT, errors, oracle, existing flow or labels.
    base = record["tensors"]["base"].detach().contiguous()
    _base_tensor(base)
    base_sha = tensor_digest(base)
    source_sha = origin["summary"]["per_sequence"][index]["cache_sha256"]
    if manifest["kind"] == "slim":
        _require(record["source_cache_sha256"] == source_sha == row["source_cache_sha256"]
                 and record["base_tensor_sha256"] == base_sha == row["base_tensor_sha256"],
                 "Slim base tensor SHA or source cache SHA differs")
    return {"name": row["name"], "scene": row["scene"], "plan": expected_plan,
            "source_cache_sha256": source_sha, "base_tensor_sha256": base_sha, "base": base}


def export_slim(cache, out):
    manifest = base_manifest(cache)
    out = _new_output(out)
    provenance = {"schema_version": 1, "kind": "source_base_only_slim",
                  "source_cache": manifest["source_cache"], "tool_sha256": digest(__file__),
                  "base_shape": list(BASE_SHAPE), "base_tensor_hash_format": "contiguous float32 CPU bytes",
                  "scope": "source TRAIN only; base tensor transport; no GT tensor read by field or exported"}
    _atomic_json(out / "provenance.json", provenance)
    rows = []
    for i in range(EXPECTED_CLIPS):
        sample = load_base(manifest, i)
        base = sample.pop("base")
        filename = f"base_{i:03d}.pt"
        record = {"schema_version": 1, "kind": "source_base_only_slim", **sample, "tensors": {"base": base}}
        _atomic_tensor(out / filename, record)
        rows.append({"name": sample["name"], "scene": sample["scene"], "cache_path": filename,
                     "cache_sha256": digest(out / filename), "source_cache_sha256": sample["source_cache_sha256"],
                     "base_tensor_sha256": sample["base_tensor_sha256"]})
        print(json.dumps({"event": "exported_base", "index": i, "name": sample["name"]}), flush=True)
    summary = {"schema_version": 1, "kind": "source_base_only_slim", "n": len(rows), "per_sequence": rows,
               "provenance_sha256": digest(out / "provenance.json"), "gt_tensor_exported": False}
    _atomic_json(out / "cache_summary.json", summary)
    return summary


@lru_cache(maxsize=1)
def _donor_correction():
    """Load the one frozen function, without executing the GT-aware module.

    This is only the local hard-coded source file. Its full-file and exact
    FunctionDef source hashes are pinned. Cache/payload paths cannot select or
    inject code. The function is evaluated once in a torch/F-only namespace.
    """
    import torch
    from torch.nn import functional as F
    path = CODE / "tools/cache_correction_oracle.py"
    _require(digest(path) == DONOR_SOURCE_SHA, "Frozen donor source file SHA differs")
    source = path.read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "donor_correction"]
    _require(len(nodes) == 1, "Require exactly one frozen donor_correction FunctionDef")
    function_source = ast.get_source_segment(source, nodes[0])
    _require(hashlib.sha256(function_source.encode()).hexdigest() == DONOR_FUNCTION_SHA,
             "Frozen donor function source SHA differs")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["donor_correction"]


def adjacent_flows(base, raft):
    """Shared source/target helper: fixed ±1, 12 RAFT updates, batch2.

    Accepts only base RGB and an eval-mode frozen RAFT. All computation stays
    on base.device; outputs are [1,2,2,T,H,W] flow and [1,2,1,T,H,W] masks.
    The original donor helper computes a throwaway RGB correction internally;
    it sees no additional observation, label or GT. No outputs are fitted.
    """
    import torch
    _base_tensor(base, fixed_shape=False)
    _require(not raft.training and not any(p.requires_grad for p in raft.parameters()),
             "RAFT must be frozen in eval mode")
    function = _donor_correction()
    with torch.no_grad():
        pairs = [function(base, raft, delta, 12, 2)[1:] for delta in (-1, 1)]
        flows, masks = torch.stack([p[0] for p in pairs], 1), torch.stack([p[1] for p in pairs], 1)
    _require(flows.device == base.device and masks.device == base.device and bool(torch.isfinite(flows).all()),
             "Adjacent flow device/finite guard failed")
    return flows, masks


def state_digest(state):
    """Match the v1 cache's train_method_pilot state fingerprint format."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def cache_flows(cache, out, raft, raft_weights, device):
    """Cache only flow/mask plus lineage; no GT input/output or quality metric."""
    import torch
    _require(digest(raft_weights) == RAFT_WEIGHTS_SHA, "Predeclared RAFT-small weights SHA differs")
    manifest = base_manifest(cache)
    _require(not raft.training and not any(p.requires_grad for p in raft.parameters()), "RAFT must be frozen")
    before = state_digest(raft.state_dict())
    out = _new_output(out)
    provenance = {"schema_version": 1, "kind": "source_adjacent_flow_cache", "source_cache": manifest["source_cache"],
                  "input_kind": manifest["kind"], "input_provenance_sha256": manifest["input_provenance_sha256"],
                  "input_summary_sha256": manifest["input_summary_sha256"], "tool_sha256": digest(__file__),
                  "donor_source_sha256": DONOR_SOURCE_SHA, "donor_function_sha256": DONOR_FUNCTION_SHA,
                  "raft_weights_sha256": RAFT_WEIGHTS_SHA, "raft_state_sha256_before": before,
                  "args": {"offsets": [-1, 1], "flow_steps": 12, "flow_batch": 2, "device": str(device)},
                  "base_shape": list(BASE_SHAPE), "flow_inputs": "only source frozen base RGB in [-1,1]",
                  "gt_input": False, "target_input": False, "metric_computed": False}
    _atomic_json(out / "provenance.json", provenance)
    provenance_sha = digest(out / "provenance.json")
    rows, started = [], time.monotonic()
    for i in range(EXPECTED_CLIPS):
        sample = load_base(manifest, i)
        base = sample.pop("base").to(device)
        flows, masks = adjacent_flows(base, raft)
        filename = f"adjacent_{i:03d}.pt"
        record = {"schema_version": 1, "kind": "source_adjacent_flow_cache", **sample,
                  "provenance_sha256": provenance_sha, "tensors": {"flow": flows.cpu(), "valid_mask": masks.cpu()}}
        _atomic_tensor(out / filename, record)
        rows.append({"name": sample["name"], "scene": sample["scene"], "cache_path": filename,
                     "cache_sha256": digest(out / filename), "source_cache_sha256": sample["source_cache_sha256"],
                     "base_tensor_sha256": sample["base_tensor_sha256"]})
        print(json.dumps({"event": "cached_adjacent", "index": i, "name": sample["name"],
                          "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
    after = state_digest(raft.state_dict())
    _require(before == after, "Adjacent-flow computation mutated RAFT state")
    summary = {"schema_version": 1, "kind": "source_adjacent_flow_cache", "n": len(rows), "per_sequence": rows,
               "provenance_sha256": provenance_sha, "raft_state_sha256_before": before,
               "raft_state_sha256_after": after, "raft_state_unchanged": True,
               "gt_input": False, "metric_computed": False, "elapsed_seconds": time.monotonic() - started}
    _atomic_json(out / "cache_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    slim = commands.add_parser("export-slim")
    slim.add_argument("--cache", type=Path, required=True)
    slim.add_argument("--out", type=Path, required=True)
    flow = commands.add_parser("flows")
    flow.add_argument("--cache", type=Path, required=True)
    flow.add_argument("--out", type=Path, required=True)
    flow.add_argument("--raft-weights", type=Path, required=True)
    flow.add_argument("--device", choices=("cpu", "cuda", "npu"), default="npu")
    args = parser.parse_args()
    if args.command == "export-slim":
        export_slim(args.cache, args.out)
        return
    _require(digest(args.raft_weights) == RAFT_WEIGHTS_SHA, "Predeclared RAFT-small weights SHA differs")
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
    from torchvision.models.optical_flow import raft_small
    device = torch.device("cpu" if args.device == "cpu" else args.device + ":0")
    raft = raft_small(weights=None).eval()
    raft.load_state_dict(torch.load(args.raft_weights, map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False).to(device)
    cache_flows(args.cache, args.out, raft, args.raft_weights, device)


if __name__ == "__main__":
    main()
