"""Resolve inference settings without importing torch or allocating a model.

`saved` follows the training checkpoint. `legacy` explicitly reproduces the
archived evaluator's omitted settings; it is for forensic comparisons only.
Neither mode changes image preprocessing, metric definitions or split lists.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

DEFAULT_TAPS = (5, 11, 17, 23)
RANDOM_VJEPA_SEED = 1234  # seer_train_cvpr.py and seer_train_e4.py
RECONSTRUCTION_VERSION = 2


def resolve_config(saved, force_arm=None, mode="saved"):
    if mode not in ("saved", "legacy"):
        raise ValueError(f"unknown reconstruction mode: {mode!r}")
    arch = saved.get("arch", "v2")
    trained_arm = saved.get("vjepa", "pretrained")
    arm = force_arm if force_arm is not None else trained_arm
    for name, value, allowed in (
        ("arch", arch, ("v1", "v2")),
        ("trained vjepa", trained_arm, ("pretrained", "none", "random")),
        ("vjepa", arm, ("pretrained", "none", "random")),
    ):
        if value not in allowed:
            raise ValueError(f"invalid {name}: {value!r}")
    if force_arm is not None and (force_arm == "none" or trained_arm == "none"):
        raise ValueError("force_arm requires a checkpoint trained with tokens and "
                         "a pretrained or random replacement encoder")
    clip_len = int(saved.get("clip_len", 16))
    if clip_len <= 0 or clip_len % 2:
        raise ValueError("clip_len must be a positive multiple of the tubelet size 2")
    branch = saved.get("branch", "full") if arch == "v2" else "full"
    xattn = saved.get("xattn", "global") if arch == "v2" else "global"
    kv = saved.get("kv", "per_tubelet") if mode == "saved" and arch == "v2" else "per_tubelet"
    layers = saved.get("layers", "single") if mode == "saved" and arch == "v2" else "single"
    projection = (saved.get("token_proj", "none")
                  if mode == "saved" and arch == "v2" and trained_arm != "none" else "none")
    for name, value, allowed in (
        ("branch", branch, ("full", "small")),
        ("xattn", xattn, ("global", "spatial", "gated")),
        ("kv", kv, ("per_tubelet", "full", "hybrid")),
        ("layers", layers, ("single", "all")),
        ("token_proj", projection, ("none", "clip_rank", "global_rank")),
    ):
        if value not in allowed:
            raise ValueError(f"invalid {name}: {value!r}")
    if layers == "all" and kv != "per_tubelet":
        raise ValueError("the archived branch supports layers=all only with kv=per_tubelet")
    raw_taps = saved.get("taps", DEFAULT_TAPS) if mode == "saved" else DEFAULT_TAPS
    try:
        taps = tuple(int(t) for t in (raw_taps.split(",") if isinstance(raw_taps, str) else raw_taps))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid taps: {raw_taps!r}") from exc
    if len(taps) != 4 or any(t < 0 or t > 23 for t in taps):
        raise ValueError("this branch requires four V-JEPA ViT-L taps in [0, 23]")
    rank = int(saved.get("token_proj_k", 4))
    if projection != "none" and rank <= 0:
        raise ValueError("token_proj_k must be positive")
    return dict(version=RECONSTRUCTION_VERSION, mode=mode, arch=arch,
                trained_arm=trained_arm, arm=arm, clip_len=clip_len,
                branch=branch, xattn=xattn, kv=kv, layers=layers, taps=list(taps),
                token_proj=projection, token_proj_k=rank,
                random_encoder_seed=(RANDOM_VJEPA_SEED
                                     if arm == "random" and mode == "saved" else None))


def resolve_basis_path(saved_path, root, ckpt, override=None):
    """Only relocate known host roots; never silently substitute another basis."""
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"token basis override does not exist: {path}")
        return path.resolve()
    if not saved_path:
        raise ValueError("global_rank checkpoint has no token_basis; pass --token_basis")
    path = Path(saved_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path(ckpt).resolve().parent / path, path]
    for prefix in ("/temp/zihaohe/seer_cvpr", "/cache/zihaohe/seer_cvpr"):
        try:
            suffix = path.relative_to(prefix)
        except ValueError:
            continue
        candidates.append(Path(root) / suffix)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"saved token basis {saved_path!r} was not found; pass --token_basis")


@contextmanager
def encoder_initialization_seed(seed):
    """Seed CPU construction exactly as training did, without changing caller RNG.

    VJepaBackboneLocal21 builds its ViT on CPU before `.to(dev)`. Only that CPU
    generator needs seeding; avoid changing CUDA/NPU RNG state as a side effect.
    """
    if seed is None:
        yield
        return
    import torch
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)
