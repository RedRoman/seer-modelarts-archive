"""YAML config loading with recursive `_base_:` inheritance and `key.sub=val` overrides."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

import yaml  # type: ignore


def _parse_value(v: str) -> Any:
    if v.lower() in ("none", "null", "~"):
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v and v[0] in "[{":
        try:
            return json.loads(v)
        except Exception:
            return v
    try:
        if "." not in v and "e" not in v.lower():
            return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _set_by_dotted_key(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update `dst` with `src`. Nested dicts merge; scalars overwrite."""
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _load_one(path: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Recursively load a YAML config with top-level `_base_: <path>` support."""
    if _seen is None:
        _seen = set()
    real = os.path.realpath(path)
    if real in _seen:
        raise ValueError(f"config inheritance cycle involving {real}")
    _seen = _seen | {real}

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("_base_", None)
    if base_path is None:
        return cfg

    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(path), base_path)
    base = _load_one(base_path, _seen=_seen)

    merged: Dict[str, Any] = {}
    deep_update(merged, base)
    deep_update(merged, cfg)
    return merged


def load_config(
    path: str, overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load YAML at `path`, recursively resolving `_base_:` chain + apply CLI overrides."""
    cfg = _load_one(path)
    if overrides:
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"bad override {ov!r}; expected key=value")
            k, v = ov.split("=", 1)
            _set_by_dotted_key(cfg, k.strip(), _parse_value(v.strip()))
    return cfg


def dump_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
