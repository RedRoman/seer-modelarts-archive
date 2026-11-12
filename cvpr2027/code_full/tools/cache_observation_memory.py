"""Export frozen block-17 V-JEPA addresses from cached SDSD TRAIN inputs only.

The original cached LQ crop/flip is the only encoder input. GT tensors are never
accessed by field. One CPU float32 projected token field is saved per clip; the
trainer can reshape [B,2048,64] to [B,8,16,16,64] and repeat tubelets twice.
No target-domain frames, pixel decoder, RAFT, optimization or metrics are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
for _path in (CODE, CODE / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from train_correction_selector import digest, local_module
from train_source_selector import cache_manifest

ENCODER_SHA = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
BLOCK, GRID, DIM, PROJECTION_SEED = 17, (8, 16, 16), 64, 1729
SOURCE_FILES = ("tools/cache_observation_memory.py", "models/observation_memory.py",
                "models/vjepa_backbone.py", "tools/train_source_selector.py",
                "tools/train_correction_selector.py", "tools/cache_source_corrections.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha(module):
    """Full parameter/buffer bytes, names, shapes and dtypes; no checksum proxy."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def clean_encoder_state(checkpoint, expected):
    """Require exact video encoder keys; discard only its unused image patcher.

    The unchanged Local21 video wrapper sets img_temporal_dim_size=None. The
    official checkpoint also stores a 1-frame image patcher, which is never
    instantiated or called on our 16-frame clips. All live keys load strictly.
    """
    require(isinstance(checkpoint, dict) and isinstance(checkpoint.get("encoder"), dict),
            "Expected the official encoder checkpoint branch")
    cleaned = {}
    for key, value in checkpoint["encoder"].items():
        require(key.startswith("module.backbone."), "Unexpected encoder key prefix")
        name = key[len("module.backbone."):]
        require(name not in cleaned, "Duplicate cleaned encoder key")
        cleaned[name] = value
    ignored = {"patch_embed_img.proj.weight": (1024, 3, 1, 16, 16),
               "patch_embed_img.proj.bias": (1024,)}
    extra = set(cleaned) - set(expected)
    require(extra == set(ignored), "Unexpected encoder state outside unused image patcher")
    for key, shape in ignored.items():
        require(tuple(cleaned[key].shape) == shape, "Unexpected unused image patcher shape")
        del cleaned[key]
    require(set(cleaned) == set(expected), "Missing live video encoder keys")
    for key in expected:
        require(cleaned[key].shape == expected[key].shape, f"Encoder shape mismatch: {key}")
    return cleaned, sorted(ignored)


def load_lq(path, row, plan):
    import torch
    require(digest(path) == row["cache_sha256"], "Source cache changed after validation")
    record = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    require(record.get("name") == row["name"] and record.get("scene") == row["scene"]
            and record.get("plan") == plan, "Source cache identity/plan mismatch")
    # Never access GT or any derived target statistic.
    low = record["tensors"]["lq"]
    require(isinstance(low, torch.Tensor) and low.device.type == "cpu"
            and low.dtype == torch.float32 and tuple(low.shape) == (1, 3, 16, 256, 256),
            "Expected cached CPU float32 LQ [1,3,16,256,256]")
    require(bool(torch.isfinite(low).all()) and bool(((low >= 0) & (low <= 1)).all()),
            "Invalid cached low-light RGB")
    return low


def project_clip(encoded, memory):
    import torch
    require(encoded.get("token_grid") == GRID and encoded.get("hidden_dim") == 1024
            and len(encoded.get("feats", [])) == 1, "Unexpected block17 encoder output")
    tokens = encoded["feats"][0].detach().to(device="cpu", dtype=torch.float32)
    require(tuple(tokens.shape) == (1, 2048, 1024) and bool(torch.isfinite(tokens).all()),
            "Expected finite block17 tokens [1,2048,1024]")
    projected = memory.project_prior_tokens(tokens, descriptor_dim=DIM,
                                            projection_seed=PROJECTION_SEED)
    require(projected.device.type == "cpu" and projected.dtype == torch.float32
            and tuple(projected.shape) == (1, 2048, DIM)
            and bool(torch.isfinite(projected).all()), "Invalid projected address field")
    return projected.contiguous()


def check_origin(origin):
    saved = origin.get("base_saved_args", {})
    require(saved.get("tag") == "e17_small_pretrained_s0" and saved.get("seed") == 0
            and saved.get("iters") == 20000 and saved.get("taps") == "5,11,17,23"
            and saved.get("vjepa") == "pretrained" and saved.get("heldout") is None
            and saved.get("only") is None, "Require declared pretrained E17 SDSD source cache")
    require(origin.get("encoder_weights_sha256") == ENCODER_SHA
            and origin["args"]["data_seed"] == 20261113
            and origin["args"]["crop"] == 256 and origin["args"]["clip_len"] == 16,
            "Encoder weights or source sampling recipe differ")


def upstream_sources(repo):
    repo = Path(repo).resolve()
    paths = {Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m, "__file__", None) and str(m.__file__).endswith(".py")
             and Path(m.__file__).resolve().is_relative_to(repo)}
    return {str(p.relative_to(repo)): digest(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seer-root", type=Path,
                        default=Path(os.environ.get("SEER_ROOT", "/temp/zihaohe/seer_cvpr")))
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    args = parser.parse_args()
    require(args.out.is_absolute() and args.out.resolve().is_relative_to(Path("/cache"))
            and not args.out.exists() and not args.out.is_symlink(),
            "Output must be a new directory resolving under /cache")
    splits = args.seer_root / "datasets/splits_cvpr.json"
    files, origin, rows = cache_manifest(args.cache, splits)
    check_origin(origin)
    weights = args.seer_root / "weights/vjepa2_1_vitl_dist_vitG_384.pt"
    require(digest(weights) == ENCODER_SHA, "Encoder checkpoint SHA differs")
    source_sha = {p: digest(CODE / p) for p in SOURCE_FILES}
    import torch
    torch.set_num_threads(4)
    if args.device == "npu":
        import torch_npu  # noqa: F401
    dev = torch.device(args.device + ":0")
    memory = local_module("_memory_export_feature", CODE / "models/observation_memory.py")
    wrapper = local_module("_memory_export_vjepa", CODE / "models/vjepa_backbone.py")
    repo = args.seer_root / "modelzoo/vjepa2-main"
    # Construction consumes local RNG only; frozen loaded state determines output.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        backbone = wrapper.VJepaBackboneLocal21(
            local_repo=str(repo), local_ckpt="", feature_layers=(BLOCK,),
            image_size=256, frames_per_clip=16, patch_size=16, tubelet_size=2, freeze=True)
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False, mmap=True)
    state, ignored = clean_encoder_state(checkpoint, backbone.model.state_dict())
    backbone.model.load_state_dict(state, strict=True)
    del state, checkpoint
    backbone.eval().requires_grad_(False)
    require(backbone.feature_layers == [BLOCK] and backbone.model.out_layers == [BLOCK]
            and backbone.model.hierarchical_layers[2] == BLOCK
            and backbone.model.patch_embed_img is None, "Video tap selection mismatch")
    before = state_sha(backbone)
    upstream = upstream_sources(repo)
    require(upstream, "No imported upstream source provenance")
    backbone.to(dev)
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "kind": "source_observation_memory_tokens",
                "n": 136, "domain": "sdsd_train", "token_grid": list(GRID),
                "descriptor_dim": DIM, "tubelet_repeat": 2, "feature_layer": BLOCK,
                "feature_layer_indexing": "zero-based block17, norms_block[2] output",
                "projection": memory.memory_feature_config(),
                "tensor_layout": "B,N,D; N is tubelet,row,column with grid8x16x16",
                "expanded_layout": "reshape(B,8,16,16,64).repeat_interleave(2,dim=1)",
                "source_cache": str(args.cache.resolve()),
                "source_cache_provenance_sha256": digest(args.cache / "provenance.json"),
                "source_cache_summary_sha256": digest(args.cache / "cache_summary.json"),
                "split_sha256": digest(splits), "source_names": origin["source_names"],
                "encoder_checkpoint": str(weights), "encoder_weights_sha256": ENCODER_SHA,
                "encoder_state_before": before, "encoder_eval": True,
                "encoder_requires_grad": False, "video_state_load_strict": True,
                "ignored_unused_image_only_keys": ignored, "source_sha256": source_sha,
                "upstream_repo": str(repo), "upstream_source_sha256": upstream,
                "encoder_autocast": "bfloat16", "projection_backend": "cpu_float32",
                "encoder_input": "cached original LQ only, fixed crop/flip/temporal plan",
                "gt_input": False, "target_input": False, "metric_computed": False,
                "oracle_computed": False, "optimization_performed": False}
    write_json_new(args.out / "provenance.json", metadata)
    output_rows = []
    start = time.monotonic()
    with torch.inference_mode():
        for i, (path, row) in enumerate(zip(files, rows)):
            low = load_lq(path, row, origin["plan"][i])
            require(not backbone.training and not any(p.requires_grad for p in backbone.parameters()),
                    "Frozen encoder state flags changed")
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                encoded = backbone(low.to(dev))
            projected = project_clip(encoded, memory)
            target = args.out / f"tokens_{i:03d}.pt"
            payload = {"name": row["name"], "scene": row["scene"], "plan": origin["plan"][i],
                       "source_cache_sha256": row["cache_sha256"],
                       "lq_tensor_sha256": tensor_sha(low), "tokens": projected}
            with target.open("xb") as stream:
                torch.save(payload, stream)
            result = {"name": row["name"], "scene": row["scene"], "cache_path": target.name,
                      "cache_sha256": digest(target), "source_cache_sha256": row["cache_sha256"],
                      "lq_tensor_sha256": payload["lq_tensor_sha256"],
                      "tokens_tensor_sha256": tensor_sha(projected)}
            output_rows.append(result)
            print(json.dumps({"n": i + 1, "name": row["name"],
                              "elapsed_seconds": time.monotonic() - start}), flush=True)
    after = state_sha(backbone)
    require(before == after and not backbone.training
            and not any(p.requires_grad for p in backbone.parameters()), "Frozen encoder state changed")
    require(source_sha == {p: digest(CODE / p) for p in SOURCE_FILES}
            and upstream == upstream_sources(repo), "Source changed during export")
    require(metadata["source_cache_provenance_sha256"] == digest(args.cache / "provenance.json")
            and metadata["source_cache_summary_sha256"] == digest(args.cache / "cache_summary.json"),
            "Source cache metadata changed during export")
    write_json_new(args.out / "manifest.json", dict(metadata, per_sequence=output_rows,
                   encoder_state_after=after, encoder_state_unchanged=True, complete=True,
                   elapsed_seconds=time.monotonic() - start,
                   provenance_sha256=digest(args.out / "provenance.json")))


if __name__ == "__main__":
    main()
