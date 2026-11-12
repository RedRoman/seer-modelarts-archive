"""Selective source-cache reader adapted from frozen HVI export; no GT storage."""
from collections import OrderedDict
from dataclasses import dataclass
import io
import math
import pickle
import zipfile
import hashlib
import numpy as np

def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

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


def read_allowed_tensor(path, key, expected_shape=(1, 3, 16, 256, 256), *, archive_class=zipfile.ZipFile):
    """Read only an explicitly allowed LQ/base storage; never GT or another label."""
    if key not in ("lq", "base"):
        raise ValueError("Only observable LQ/base storage is permitted")
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
        value = metadata["tensors"][key]
        if not isinstance(value, TensorSpec) or value.storage.dtype != "float32":
            raise ValueError("Expected float32 LQ metadata")
        if value.size != tuple(expected_shape) or value.storage.numel != math.prod(value.size):
            raise ValueError("Unexpected LQ shape/storage length")
        if value.offset != 0 or any(s < 0 for s in value.stride):
            raise ValueError("Unsupported LQ storage view")
        for path_parts, other in tensor_specs(metadata):
            if path_parts != ("tensors", key) and other.storage.key == value.storage.key:
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
        lineage = {"archive_entries_read": entries_read, "selected_storage_entry": storage_entry,
                   "selected_tensor_sha256": tensor_digest(low), "lq_shape": list(value.size),
                   "lq_dtype": "float32", "input_reencoded_to_uint8": False}
        return str(metadata["name"]), str(metadata["scene"]), low, lineage


