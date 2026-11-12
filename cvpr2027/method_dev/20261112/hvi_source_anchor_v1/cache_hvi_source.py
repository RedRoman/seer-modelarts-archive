"""TRAIN-only float HVI export; a selective archive reader touches LQ storage only."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import time
import zipfile

import numpy as np

WEIGHT_SHA = "6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27"
STATE_SHA = "afa01e29d1314dae7fadc68d7c95e3ddaf3da3a5b469fa83febb2759dc0bc41b"
SOURCE_PROVENANCE_SHA = "0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0"
UPSTREAM = {
    "CIDNet.py": "dc9b9232f225f5760417175d25f8efc146f51c0acd1ee014df6b9f0a6b2e8d84",
    "HVI_transform.py": "8be0fea6e61c51d6290722f0cf81be1f334b1b998dbfd409435343d9d4bd5bb2",
    "transformer_utils.py": "7bfb2659e97289f0e8005935e15edbe0af3ae5ef05c50863a074fd745617b13f",
    "LCA.py": "271f487091bd13ed211a8b0c9cb07036957ba535f5dd3aedc131498500979d26"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


@dataclass(frozen=True)
class StorageSpec:
    dtype: str
    key: str
    location: str
    numel: int


@dataclass(frozen=True)
class TensorSpec:
    storage: StorageSpec
    offset: int
    size: tuple
    stride: tuple


def rebuild_tensor(storage, offset, size, stride, *unused):
    return TensorSpec(storage, int(offset), tuple(size), tuple(stride))


class MetadataOnlyUnpickler(pickle.Unpickler):
    """Read tensor metadata without opening any tensor storage or arbitrary class."""
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return rebuild_tensor
        dtypes = {"FloatStorage": "float32", "BoolStorage": "bool", "LongStorage": "int64", "IntStorage": "int32"}
        if module == "torch" and name in dtypes:
            return dtypes[name]
        raise ValueError(f"Disallowed pickle global: {module}.{name}")

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage":
            raise ValueError("Unsupported storage reference")
        _, dtype, key, location, numel = value
        if dtype not in ("float32", "bool", "int64", "int32") or not str(key).isdecimal():
            raise ValueError("Unsupported storage metadata")
        return StorageSpec(dtype, str(key), str(location), int(numel))


def tensor_specs(value, path=(), depth=0):
    if depth > 12:
        raise ValueError("Unexpected recursive metadata")
    if isinstance(value, TensorSpec):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_specs(item, path + (str(key),), depth + 1)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from tensor_specs(item, path + (str(i),), depth + 1)


def read_lq_only(path, expected_shape=(1, 3, 16, 256, 256), *, archive_class=zipfile.ZipFile):
    """Return exactly the LQ tensor; all other storage entries remain unopened."""
    import torch
    entries_read = []
    with archive_class(path) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if n.endswith("/data.pkl")]
        if len(metadata_names) != 1:
            raise ValueError("Expected one standard torch-save archive")
        metadata_name = metadata_names[0]
        root = metadata_name[:-len("data.pkl")]
        def read(name):
            entries_read.append(name)
            return archive.read(name)
        metadata = MetadataOnlyUnpickler(io.BytesIO(read(metadata_name))).load()
        if not isinstance(metadata, dict) or not isinstance(metadata.get("tensors"), dict):
            raise ValueError("Unexpected source-cache schema")
        value = metadata["tensors"]["lq"]
        if not isinstance(value, TensorSpec) or value.storage.dtype != "float32":
            raise ValueError("Expected float32 LQ metadata")
        if value.size != tuple(expected_shape) or value.storage.numel != math.prod(value.size):
            raise ValueError("Unexpected LQ shape/storage length")
        if value.offset != 0 or any(s < 0 for s in value.stride):
            raise ValueError("Unsupported LQ storage view")
        for path_parts, other in tensor_specs(metadata):
            if path_parts != ("tensors", "lq") and other.storage.key == value.storage.key:
                raise ValueError("LQ storage is shared with another tensor")
        order_name = root + "byteorder"
        byteorder = read(order_name).decode() if order_name in names else "little"
        if byteorder != "little":
            raise ValueError("Expected the frozen little-endian source storage")
        storage_entry = root + "data/" + value.storage.key
        raw = read(storage_entry)
        if len(raw) != value.storage.numel * 4:
            raise ValueError("Unexpected LQ byte length")
        maximum_index = sum((n - 1) * s for n, s in zip(value.size, value.stride))
        if maximum_index >= value.storage.numel:
            raise ValueError("LQ stride escapes its storage")
        array = np.ndarray(value.size, dtype="<f4", buffer=raw,
                           strides=tuple(s * 4 for s in value.stride)).copy(order="C")
        low = torch.from_numpy(array)
        lineage = {"archive_entries_read": entries_read, "lq_storage_entry": storage_entry,
                   "lq_tensor_sha256": tensor_digest(low), "lq_shape": list(value.size),
                   "lq_dtype": "float32", "input_reencoded_to_uint8": False}
        return str(metadata["name"]), str(metadata["scene"]), low, lineage


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_hvi(release, device):
    import torch
    release = Path(release)
    for name, expected in UPSTREAM.items():
        assert digest(release / "upstream/hvi_cidnet/net" / name) == expected
    weight = release / "weights/hvi_cidnet.pt"
    assert digest(weight) == WEIGHT_SHA
    sys.path.insert(0, str(release / "upstream/hvi_cidnet"))
    from net.CIDNet import CIDNet
    model = CIDNet(channels=[36, 36, 72, 144], heads=[1, 2, 4, 8], norm=False)
    payload = torch.load(str(weight), map_location="cpu", weights_only=False)
    state = payload.get("params", payload) if isinstance(payload, dict) else payload
    assert isinstance(state, dict) and len(state) == 191
    model.load_state_dict(state, strict=True)
    model = model.float().requires_grad_(False).eval().to(device)
    assert state_digest(model.state_dict()) == STATE_SHA
    return model


def predict_float_base(model, low, device):
    """Only observable float LQ enters inference; no GT or original-base parameter."""
    import torch
    if low.ndim != 5 or low.shape[:2] != (1, 3) or low.dtype != torch.float32:
        raise ValueError("Expected float32 BCTHW RGB")
    if not torch.isfinite(low).all() or low.min() < 0 or low.max() > 1:
        raise ValueError("Invalid LQ values")
    output = []
    with torch.inference_mode():
        for frame in low.unbind(dim=2):
            prediction = model(frame.to(device)).float().clamp(0, 1)
            output.append(prediction.cpu())
    return torch.stack(output, dim=2).contiguous()


def output_payload(name, scene, base, lineage):
    import torch
    if base.dtype != torch.float32 or not torch.isfinite(base).all() or base.min() < 0 or base.max() > 1:
        raise ValueError("Invalid float HVI output")
    return {"name": name, "scene": scene, "base": base,
            "lineage": {**lineage, "base_tensor_sha256": tensor_digest(base)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source, release = args.source_cache, args.release
    release_manifest = json.loads((release / "release_manifest.json").read_text())
    for path, sha in release_manifest["files"].items():
        assert digest(release / path) == sha, path
    assert digest(source / "provenance.json") == SOURCE_PROVENANCE_SHA
    origin = json.loads((source / "provenance.json").read_text())
    summary = json.loads((source / "cache_summary.json").read_text())
    names = [row["name"] for row in summary["per_sequence"]]
    assert summary["n"] == len(names) == len(set(names)) == 136
    assert origin["domain"] == "sdsd" and names == origin["source_names"]
    assert len({row["scene"] for row in summary["per_sequence"]}) == 68
    import torch
    if args.device == "npu":
        import torch_npu
    torch.set_num_threads(4)
    device = torch.device("cpu" if args.device == "cpu" else "npu:0")
    model = load_hvi(release, device)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    began = time.monotonic()
    for index, row in enumerate(summary["per_sequence"]):
        path = source / row["cache_path"]
        assert digest(path) == row["cache_sha256"]
        name, scene, low, lineage = read_lq_only(path)
        assert name == row["name"] and scene == row["scene"]
        base = predict_float_base(model, low, device)
        assert base.shape == low.shape
        lineage.update({"source_cache_file": str(path), "source_cache_sha256": row["cache_sha256"],
            "source_provenance_sha256": SOURCE_PROVENANCE_SHA, "weight_sha256": WEIGHT_SHA,
            "config_sha256": digest(release / "config.json"), "model_state_sha256": STATE_SHA,
            "release_manifest_sha256": digest(release / "release_manifest.json"),
            "adapter_sha256": digest(__file__)})
        payload = output_payload(name, scene, base, lineage)
        target = args.out / f"hvi_{index:03d}.pt"
        with target.open("xb") as f:
            torch.save(payload, f)
        check = torch.load(str(target), map_location="cpu", mmap=True, weights_only=False)
        assert set(check) == {"name", "scene", "base", "lineage"}
        assert check["base"].shape == (1, 3, 16, 256, 256)
        assert tensor_digest(check["base"]) == payload["lineage"]["base_tensor_sha256"]
        rows.append({"name": name, "scene": scene, "file": target.name,
            "file_sha256": digest(target), "base_tensor_sha256": lineage.get("base_tensor_sha256", payload["lineage"]["base_tensor_sha256"]),
            "source_cache_sha256": row["cache_sha256"], "lq_tensor_sha256": lineage["lq_tensor_sha256"],
            "shape": [1, 3, 16, 256, 256], "dtype": "float32", "finite_range_verified": True})
        print(json.dumps({"completed": index + 1, "total": 136, "name": name, "output_sha256": rows[-1]["file_sha256"]}), flush=True)
    assert state_digest(model.state_dict()) == STATE_SHA
    report = {"purpose": "GT-free source preparation only; no scores or training", "n": 136,
        "scenes": 68, "per_sequence": rows, "source_provenance_sha256": SOURCE_PROVENANCE_SHA,
        "source_summary_sha256": digest(source / "cache_summary.json"),
        "model_state_unchanged": True, "model_state_sha256": STATE_SHA, "weight_sha256": WEIGHT_SHA,
        "config_sha256": digest(release / "config.json"), "release_manifest_sha256": digest(release / "release_manifest.json"),
        "gt_storage_entries_read": 0, "target_inputs_read": 0, "quality_scores_computed": False,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - began}
    write_new(args.out / "manifest.json", report)


if __name__ == "__main__":
    main()
