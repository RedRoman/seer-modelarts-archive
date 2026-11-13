"""Zero-Shot Cross-Domain and Cross-Backbone Evaluator for EID-TT v2.

Evaluates frozen checkpoints trained ONLY on SDSD FIT51 (or BVI TRAIN48 for
reverse transfer) with ZERO target tuning and ZERO GT access during inference.

Supported frozen backbones:
  - hvi_cidnet       (CIDNet, generalization.pth, NPU)
  - darkir           (DarkIR_384.pt, unmodified CPU FFT forward + NPU RAFT/EID/scorer)
  - retinexformer    (RetinexFormer 160k SDSD-split, NPU)
  - seer_v2          (Seer-v2 E17 pretrained s0 wrapped in canonical RGB, NPU)

Supported domains/splits:
  - bvi / val        (BVI VAL16: 16 clips, 8 scenes x 2 light conditions)
  - bvi / test       (BVI TEST16: 16 clips, 8 held-out scenes; requires --allow-final-test)
  - sdsd / test      (SDSD TEST39: 39 clips in canonical RGB; requires --allow-final-test)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

RAFT_SHA256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
METRIC_SHA256 = "a26ff0c81f0e6fa380ca26ac87dc2197c66248d3b6945e0da1c99e5b3d3f451d"
SPLIT_SHA256 = "fb6be5ea78ef73f5cc9db90bca75813bf597275f7609fe39d82f5b2f883e9ad0"
SEER_PRETRAINED_SHA256 = "5ab6b6d04d7b13743965007d3754bd66f034d170ec7193abc2abb5a8aa6c8740"

BACKBONES = ("hvi_cidnet", "darkir", "retinexformer", "seer_v2")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def state_digest(state: dict) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        arr = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(tuple(arr.shape)).encode())
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def parse_scene_id(seq_name: str) -> str:
    if seq_name.startswith("BVI/"):
        return re.sub(r"__(?:10|20)fps$", "", seq_name)
    if seq_name.startswith("SDSD_"):
        return re.sub(r"_2$", "", seq_name)
    return seq_name


def scene_bootstrap_ci(
    rows: List[dict],
    method_key: str,
    ref_key: str,
    metric_keys: Tuple[str, ...] = ("psnr_y", "psnr_rgb", "lpips", "tlp_gap"),
    draws: int = 10000,
    seed: int = 20261112,
) -> Dict[str, dict]:
    scenes = sorted({r["scene"] for r in rows})
    by_scene = {s: [r for r in rows if r["scene"] == s] for s in scenes}
    n_scenes = len(scenes)
    rng = np.random.default_rng(seed)
    idx_matrix = rng.integers(0, n_scenes, size=(draws, n_scenes))

    out = {}
    for mk in metric_keys:
        scene_deltas = np.array(
            [
                np.mean([r["methods"][method_key][mk] - r["methods"][ref_key][mk] for r in by_scene[s]])
                for s in scenes
            ],
            dtype=np.float64,
        )
        boot = scene_deltas[idx_matrix].mean(axis=1)
        better_count = int(np.sum(scene_deltas > 0.0 if "psnr" in mk else scene_deltas < 0.0))
        out[mk] = {
            "mean_delta": float(scene_deltas.mean()),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "improved_scene_count": better_count,
            "total_scenes": n_scenes,
        }
    return out


def build_backbone_predictor(
    backbone_name: str,
    release_root: Path,
    seer_ckpt_path: Path,
    npu_device,
) -> Tuple[Callable[[np.ndarray], object], Callable[[], bool], dict]:
    import torch
    import torch.nn.functional as F

    if backbone_name in ("hvi_cidnet", "darkir", "retinexformer"):
        sources = release_root / "upstream"
        if backbone_name == "hvi_cidnet":
            if str(sources / "hvi_cidnet") not in sys.path:
                sys.path.insert(0, str(sources / "hvi_cidnet"))
            from net.CIDNet import CIDNet
            model, multiple = CIDNet(), 16
            run_dev = npu_device
        elif backbone_name == "darkir":
            if str(sources / "darkir") not in sys.path:
                sys.path.insert(0, str(sources / "darkir"))
            from DarkIR import DarkIR
            model = DarkIR(
                img_channel=3,
                width=32,
                middle_blk_num_enc=2,
                middle_blk_num_dec=2,
                enc_blk_nums=[1, 2, 3],
                dec_blk_nums=[3, 1, 1],
                dilations=[1, 4, 9],
                extra_depth_wise=True,
            )
            multiple = 8
            run_dev = torch.device("cpu")
        else:
            if str(sources / "retinexformer") not in sys.path:
                sys.path.insert(0, str(sources / "retinexformer"))
            from RetinexFormer_arch import RetinexFormer
            model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
            multiple = 4
            run_dev = npu_device

        ckpt = release_root / "weights" / f"{backbone_name}.pt"
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        state = payload.get("params", payload.get("params_ema", payload)) if isinstance(payload, dict) else payload
        loaded = model.load_state_dict(state, strict=True)
        _require(not loaded.missing_keys and not loaded.unexpected_keys, f"Non-strict load on {backbone_name}")
        model = model.float().requires_grad_(False).eval().to(run_dev)
        before_sha = state_digest(model.state_dict())

        def predict_rgb_float(low_u8: np.ndarray):
            h, w = low_u8.shape[1:3]
            preds = []
            with torch.inference_mode():
                for t_idx in range(len(low_u8)):
                    x = (
                        torch.from_numpy(low_u8[t_idx:t_idx + 1].copy())
                        .float()
                        .permute(0, 3, 1, 2)
                        .div_(255.0)
                        .to(run_dev)
                    )
                    ph, pw = (-h) % multiple, (-w) % multiple
                    if ph or pw:
                        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
                    y = model(x)
                    if isinstance(y, (tuple, list)):
                        y = y[0]
                    preds.append(y[:, :, :h, :w].float().clamp(0.0, 1.0).to(npu_device))
            return torch.cat(preds, dim=0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()

        def verify_unchanged() -> bool:
            return state_digest(model.state_dict()) == before_sha

        prov = {
            "backbone": backbone_name,
            "checkpoint": str(ckpt),
            "checkpoint_sha256": digest(ckpt),
            "state_sha256": before_sha,
            "parameters": sum(p.numel() for p in model.parameters()),
            "model_device": str(run_dev),
        }
        return predict_rgb_float, verify_unchanged, prov

    if backbone_name == "seer_v2":
        _require(digest(seer_ckpt_path) == SEER_PRETRAINED_SHA256, "Seer-v2 pretrained checkpoint SHA mismatch")
        from eval_e4 import build_model
        from cache_source_corrections import backbone_tokens

        pb, enc, refiner, exposure, actual_arm, length, _ = build_model(str(seer_ckpt_path), npu_device)
        _require(actual_arm == "pretrained" and refiner is None and exposure is None and length == 16, "Invalid Seer-v2 state")
        pb.eval().requires_grad_(False)
        enc.eval().requires_grad_(False)
        pb_sha = state_digest(pb.state_dict())
        enc_sha = state_digest(enc.state_dict())

        def predict_rgb_float(low_u8: np.ndarray):
            low_rgb = torch.from_numpy(low_u8.copy()).float().div_(255.0).permute(3, 0, 1, 2).unsqueeze(0).to(npu_device)
            low_bgr = low_rgb.flip(1).contiguous()
            with torch.inference_mode(), torch.autocast(npu_device.type, dtype=torch.bfloat16):
                tokens = backbone_tokens(enc, low_bgr)
                pred_bgr = pb(low_bgr, tokens)["pred_pixels"].float().clamp(0.0, 1.0)
            return pred_bgr.flip(1).contiguous()

        def verify_unchanged() -> bool:
            return state_digest(pb.state_dict()) == pb_sha and state_digest(enc.state_dict()) == enc_sha

        prov = {
            "backbone": "seer_v2",
            "checkpoint": str(seer_ckpt_path),
            "checkpoint_sha256": SEER_PRETRAINED_SHA256,
            "pixel_state_sha256": pb_sha,
            "encoder_state_sha256": enc_sha,
            "parameters": sum(p.numel() for p in pb.parameters()),
            "model_device": str(npu_device),
            "color_wrapper": "canonical RGB -> native BGR -> canonical RGB",
        }
        return predict_rgb_float, verify_unchanged, prov

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def to_uint8_thwc(video_bcthw) -> np.ndarray:
    return (
        video_bcthw[0]
        .detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
        * 255.0
    ).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("/cache/zihaohe/seer_baseline_replay_20261112/release_v1"),
    )
    parser.add_argument(
        "--seer-ckpt",
        type=Path,
        default=Path("/temp/zihaohe/seer_cvpr/method_dev_20261111/inputs/e17_small_pretrained_s0_final_cpu.pt"),
    )
    parser.add_argument(
        "--raft-weights",
        type=Path,
        default=Path("/temp/zihaohe/seer_cvpr/weights_small/checkpoint/raft_small_C_T_V2-01064c6d.pth"),
    )
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--domain", choices=("bvi", "sdsd"), required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--variants", type=str, nargs="+", default=["eid_full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _require(not args.out.exists(), f"Output already exists: {args.out}")
    if args.split == "test" and not args.allow_final_test:
        raise PermissionError("Refusing held-out TEST evaluation without --allow-final-test")

    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    _require(vis in {"0", "1", "2", "3", "4", "5", "6", "7"}, "Set explicit single ASCEND_RT_VISIBLE_DEVICES")
    smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    _require(f"No running processes found in NPU {vis}" in smi, f"Physical NPU {vis} is not idle")

    os.environ["SEER_ROOT"] = "/temp/zihaohe/seer_cvpr"
    code_root = args.release.resolve() / "code_full"
    sys.path[:0] = [
        str(code_root),
        str(code_root / "tools"),
        "/temp/zihaohe/seer_cvpr/code/baselines",
        "/temp/zihaohe/seer_cvpr/code/baselines/common",
    ]

    import torch
    import torch_npu  # noqa: F401
    import common.eval_lib as scoring_lib
    import eval_e4
    from eval_stcd_transfer import score_uint8
    from torchvision.models.optical_flow import raft_small
    from eid_transport import (
        EIDTrajectoryTransport,
        OFFSETS,
        compute_cycle_geometry,
        compute_reciprocal_raft_flows,
        fixed_baseline_controls,
    )

    torch.set_num_threads(8 if args.backbone == "darkir" else 4)
    npu_dev = torch.device("npu:0")

    _require(digest(Path(scoring_lib.__file__)) == METRIC_SHA256, "Scoring library SHA256 changed")
    split_path = Path(os.environ["SEER_ROOT"]) / "datasets/splits_cvpr.json"
    _require(digest(split_path) == SPLIT_SHA256, "splits_cvpr.json SHA256 changed")
    eval_e4.SPLITS = str(split_path)

    _require(digest(args.raft_weights) == RAFT_SHA256, "RAFT-small SHA256 mismatch")
    raft = raft_small(weights=None, progress=False).eval()
    raft.load_state_dict(torch.load(str(args.raft_weights), map_location="cpu", weights_only=True), strict=True)
    raft.requires_grad_(False).to(npu_dev)
    raft_sha_before = state_digest(raft.state_dict())

    predict_base, check_base_unchanged, base_prov = build_backbone_predictor(
        args.backbone, args.release.resolve(), args.seer_ckpt.resolve(), npu_dev
    )

    eid_models: Dict[str, EIDTrajectoryTransport] = {}
    ckpt_records = []
    for vname in args.variants:
        for seed in args.seeds:
            tag = f"{vname}_s{seed}"
            cpath = args.ckpt_dir / f"{tag}.pt"
            _require(cpath.is_file(), f"Missing checkpoint {cpath}")
            cpayload = torch.load(str(cpath), map_location="cpu", weights_only=False)
            m = EIDTrajectoryTransport(mode=cpayload["arch_mode"])
            m.load_state_dict(cpayload["state_dict"], strict=True)
            _require(state_digest(m.state_dict()) == cpayload["state_sha256"], f"Checkpoint state SHA mismatch: {tag}")
            m.eval().requires_grad_(False).to(npu_dev)
            eid_models[tag] = m
            ckpt_records.append(
                {
                    "tag": tag,
                    "path": str(cpath.resolve()),
                    "file_sha256": digest(cpath),
                    "state_sha256": cpayload["state_sha256"],
                    "arch_mode": cpayload["arch_mode"],
                    "loss_mode": cpayload["loss_mode"],
                }
            )

    ref_gt_tlp = None
    if args.domain == "bvi" and args.split == "val":
        ref_raw = json.loads((args.release / "reference_raw.json").read_text())
        ref_gt_tlp = {r["name"]: r["methods"]["base"]["gt_tlp"] for r in ref_raw["per_sequence"]}

    began = time.monotonic()
    rows = []
    for name, low_raw, gt_raw, kind in eval_e4.held_out_sequences(args.domain, args.split, 16, 256):
        _require(kind == "video" and low_raw.shape == (16, 256, 256, 3) and gt_raw.shape == low_raw.shape, f"Invalid clip {name}")
        _require(low_raw.dtype == np.uint8 and gt_raw.dtype == np.uint8, f"Expected uint8 arrays for {name}")

        if args.domain == "sdsd":
            low_u8 = np.ascontiguousarray(low_raw[..., ::-1])
            gt_u8 = np.ascontiguousarray(gt_raw[..., ::-1])
            color_note = "SDSD BGR .npy converted to canonical RGB via [..., ::-1]"
        else:
            low_u8 = np.ascontiguousarray(low_raw)
            gt_u8 = np.ascontiguousarray(gt_raw)
            color_note = "BVI native canonical RGB PNG"

        with torch.no_grad():
            low_bcthw = (
                torch.from_numpy(low_u8.copy())
                .float()
                .div_(255.0)
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
                .to(npu_dev)
            )
            base_bcthw = predict_base(low_u8)
            flows_cpu = compute_reciprocal_raft_flows(base_bcthw, raft, offsets=OFFSETS, flow_iters=12, pair_batch=16)
            flows_d = flows_cpu.to(npu_dev)
            geom_d = compute_cycle_geometry(flows_d)

            # Reuse static inputs per arch_mode across seeds on the same clip!
            mode_static = {}
            for vname in args.variants:
                first_tag = f"{vname}_s{args.seeds[0]}"
                am = eid_models[first_tag].mode
                if am not in mode_static:
                    _, f_am, c_am = eid_models[first_tag].prepare_static_inputs(
                        low_bcthw, base_bcthw, flows_d, geom=geom_d
                    )
                    mode_static[am] = (f_am, c_am)

            seq_methods = {}
            controls = fixed_baseline_controls(base_bcthw, flows_d, geom_d)
            for cname, c_tensor in controls.items():
                cu8 = to_uint8_thwc(c_tensor)
                sc = score_uint8(cu8, gt_u8, npu_dev)
                if ref_gt_tlp is not None:
                    _require(sc["gt_tlp"] == ref_gt_tlp[name], f"GT tLP reference mismatch on {name}")
                seq_methods[cname] = {**sc, "prediction_sha256": array_digest(cu8)}

            for vname in args.variants:
                seed_scores = []
                am = eid_models[f"{vname}_s{args.seeds[0]}"].mode
                for seed in args.seeds:
                    tag = f"{vname}_s{seed}"
                    out_d = eid_models[tag](low_bcthw, base_bcthw, flows_d, geom=geom_d, static_inputs=mode_static[am])
                    pu8 = to_uint8_thwc(out_d["pred_pixels"])
                    sc = score_uint8(pu8, gt_u8, npu_dev)
                    seq_methods[tag] = {**sc, "prediction_sha256": array_digest(pu8), "aux": out_d["aux"]}
                    seed_scores.append(sc)
                seq_methods[f"{vname}_mean3"] = {
                    mk: float(np.mean([s[mk] for s in seed_scores]))
                    for mk in ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")
                }

        scene_id = parse_scene_id(name)
        rows.append(
            {
                "name": name,
                "scene": scene_id,
                "color_contract": color_note,
                "lq_rgb_u8_sha256": array_digest(low_u8),
                "gt_rgb_u8_sha256": array_digest(gt_u8),
                "methods": seq_methods,
            }
        )
        anchor_py = seq_methods["anchor"]["psnr_y"]
        full_py = seq_methods[f"{args.variants[0]}_mean3"]["psnr_y"]
        anchor_gap = seq_methods["anchor"]["tlp_gap"]
        full_gap = seq_methods[f"{args.variants[0]}_mean3"]["tlp_gap"]
        print(
            json.dumps(
                {
                    "seq": name,
                    "done": len(rows),
                    "anchor_psnr_y": round(anchor_py, 4),
                    "eid_psnr_y": round(full_py, 4),
                    "anchor_tlp_gap": round(anchor_gap, 6),
                    "eid_tlp_gap": round(full_gap, 6),
                }
            ),
            flush=True,
        )

    _require(check_base_unchanged(), "Frozen backbone state changed during evaluation")
    _require(state_digest(raft.state_dict()) == raft_sha_before, "Frozen RAFT state changed during evaluation")
    for rec in ckpt_records:
        _require(state_digest(eid_models[rec["tag"]].state_dict()) == rec["state_sha256"], "EID state changed")

    metric_names = ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")
    all_mkeys = list(rows[0]["methods"].keys())
    means = {
        mk_method: {
            m: float(np.mean([r["methods"][mk_method][m] for r in rows])) for m in metric_names
        }
        for mk_method in all_mkeys
    }

    boot_vs_anchor = {}
    boot_vs_fixed = {}
    for mk_method in all_mkeys:
        if mk_method == "anchor":
            continue
        boot_vs_anchor[mk_method] = scene_bootstrap_ci(rows, mk_method, "anchor")
        if mk_method != "fixed_temporal_mean_0.5":
            boot_vs_fixed[mk_method] = scene_bootstrap_ci(rows, mk_method, "fixed_temporal_mean_0.5")

    report = {
        "schema_version": 2,
        "domain": args.domain,
        "split": args.split,
        "backbone": args.backbone,
        "n_sequences": len(rows),
        "n_scenes": len({r["scene"] for r in rows}),
        "protocol": "centre16/crop256 canonical RGB rounded uint8; zero GT at inference; zero target adaptation",
        "backbone_provenance": base_prov,
        "raft_weights_sha256": RAFT_SHA256,
        "raft_state_sha256": raft_sha_before,
        "scoring_lib_sha256": METRIC_SHA256,
        "split_sha256": SPLIT_SHA256,
        "eid_checkpoints": ckpt_records,
        "all_states_unchanged": True,
        "elapsed_seconds": time.monotonic() - began,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "means": means,
        "scene_bootstrap_vs_anchor": boot_vs_anchor,
        "scene_bootstrap_vs_fixed_temporal_mean": boot_vs_fixed,
        "per_sequence": rows,
    }
    write_new_json(args.out, report)
    print(json.dumps({"completed": str(args.out), "means": {k: means[k] for k in ("anchor", "fixed_temporal_mean_0.5", f"{args.variants[0]}_mean3")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
