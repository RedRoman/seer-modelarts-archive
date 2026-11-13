"""Train EID-TT and matched ablations on SDSD FIT51 only, then score internal17.

Strict data hygiene:
- Only the 51 FIT scenes (102 clips) enter gradient optimization.
- No checkpoint selection on holdout: only the final converged epoch-12 state is saved.
- Zero target-domain (BVI/LOL) or SDSD VAL/TEST files are imported or read.
- Native source LQ/GT tensors in clip_*.pt are converted from declared BGR to canonical RGB.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SOURCE_PROVENANCE_SHA = "0fe83dc30cd7772eb4aa8326deb8911e5b5538ae63470821c1c5aa21129d19f0"
SOURCE_SUMMARY_SHA = "63df74b887dc66543137569d58b6fd57c6fbdb931c6f4c6077c53b77585031ae"
GEOMETRY_MANIFEST_SHA = "d7f90214ab2da0b30da9f3f5b9fba648bb4fcabfee548e34dd7788caa6610805"
HVI_WEIGHT_SHA = "6395370762ec1fa0f6fe92a38c62382b1ebe21edc012f597023bfcec8b95cd27"

VARIANTS: Dict[str, Tuple[str, str]] = {
    "eid_full": ("eid_full", "decoupled"),
    "ablate_raw_loss": ("eid_full", "raw_gt"),
    "ablate_no_rad": ("ablate_no_rad", "decoupled"),
    "ablate_no_deatt_raw": ("ablate_no_deatt_raw", "decoupled"),
    "ablate_no_sensor_inv": ("ablate_no_sensor_inv", "decoupled"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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


def psnr_y_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    p = 0.299 * pred_u8[..., 0].astype(np.float64) + 0.587 * pred_u8[..., 1].astype(np.float64) + 0.114 * pred_u8[..., 2].astype(np.float64)
    g = 0.299 * gt_u8[..., 0].astype(np.float64) + 0.587 * gt_u8[..., 1].astype(np.float64) + 0.114 * gt_u8[..., 2].astype(np.float64)
    mse = float(np.mean((p - g) ** 2))
    return float("inf") if mse <= 1e-10 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def psnr_rgb_uint8(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    p = pred_u8.astype(np.float64)
    g = gt_u8.astype(np.float64)
    mse = float(np.mean((p - g) ** 2))
    return float("inf") if mse <= 1e-10 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def score_clip_metrics(pred_bcthw, base_bcthw, gt_bcthw, lpips_metric, device) -> Dict[str, float]:
    """Compute canonical uint8 PSNR-Y, PSNR-RGB, LPIPS, tLP gap + decoupled PSNR-RGB."""
    import torch
    from eid_transport import clip_static_exposure_align_gt

    pred_u8 = (pred_bcthw[0].clamp(0.0, 1.0).permute(1, 2, 3, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    gt_u8 = (gt_bcthw[0].clamp(0.0, 1.0).permute(1, 2, 3, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

    py = float(np.mean([psnr_y_uint8(a, b) for a, b in zip(pred_u8, gt_u8)]))
    prgb = float(np.mean([psnr_rgb_uint8(a, b) for a, b in zip(pred_u8, gt_u8)]))

    p_tchw = torch.from_numpy(pred_u8.copy()).float().permute(0, 3, 1, 2).div_(255.0).to(device)
    g_tchw = torch.from_numpy(gt_u8.copy()).float().permute(0, 3, 1, 2).div_(255.0).to(device)
    with torch.no_grad():
        lp = float(lpips_metric(p_tchw, g_tchw).mean().item())
        tlp = float(lpips_metric(p_tchw[1:], p_tchw[:-1]).mean().item())
        gt_tlp = float(lpips_metric(g_tchw[1:], g_tchw[:-1]).mean().item())

    aligned_gt = clip_static_exposure_align_gt(base_bcthw, gt_bcthw)
    dec_mse = float((pred_bcthw.clamp(0.0, 1.0) - aligned_gt).square().double().mean().item())
    dec_psnr = float(-10.0 * np.log10(max(dec_mse, 1e-12)))

    return {
        "psnr_y": py,
        "psnr_rgb": prgb,
        "decoupled_psnr_rgb": dec_psnr,
        "lpips": lp,
        "tlp": tlp,
        "gt_tlp": gt_tlp,
        "tlp_gap": abs(tlp - gt_tlp),
    }


def load_bound_partition(
    source_cache: Path,
    hvi_cache: Path,
    flow_cache: Path,
    geometry_manifest: Path,
) -> Tuple[List[dict], dict, dict]:
    _require(digest(source_cache / "provenance.json") == SOURCE_PROVENANCE_SHA, "Source provenance SHA mismatch")
    _require(digest(source_cache / "cache_summary.json") == SOURCE_SUMMARY_SHA, "Source summary SHA mismatch")
    _require(digest(geometry_manifest) == GEOMETRY_MANIFEST_SHA, "Geometry manifest SHA mismatch")

    summary = json.loads((source_cache / "cache_summary.json").read_text())
    geom_meta = json.loads(geometry_manifest.read_text())
    hvi_meta = json.loads((hvi_cache / "manifest.json").read_text())
    flow_meta = json.loads((flow_cache / "manifest.json").read_text())

    _require(
        hvi_meta["n"] == 136
        and flow_meta["n"] == 136
        and hvi_meta["weight_sha256"] == HVI_WEIGHT_SHA
        and flow_meta["hvi_manifest_sha256"] == digest(hvi_cache / "manifest.json"),
        "HVI136 / Flow136 manifest binding mismatch",
    )

    split = geom_meta["split"]
    _require(len(split["fit"]) == 51 and len(split["internal_holdout"]) == 17, "Invalid 51/17 split")
    _require(not (set(split["fit"]) & set(split["internal_holdout"])), "FIT and internal_holdout overlap")

    entries = []
    for idx, (s_row, h_row, f_row) in enumerate(
        zip(summary["per_sequence"], hvi_meta["per_sequence"], flow_meta["per_sequence"])
    ):
        _require(
            s_row["name"] == h_row["name"] == f_row["name"]
            and s_row["scene"] == h_row["scene"] == f_row["scene"],
            f"Sequence mismatch at index {idx}",
        )
        entries.append(
            {
                "index": idx,
                "name": s_row["name"],
                "scene": s_row["scene"],
                "source_path": source_cache / s_row["cache_path"],
                "source_sha256": s_row["cache_sha256"],
                "hvi_path": hvi_cache / h_row["file"],
                "hvi_sha256": h_row["file_sha256"],
                "flow_path": flow_cache / f_row["file"],
                "flow_sha256": f_row["file_sha256"],
                "rgb_lq_tensor_sha256": h_row["rgb_lq_tensor_sha256"],
                "base_tensor_sha256": h_row["base_tensor_sha256"],
            }
        )
    lineage = {
        "source_provenance_sha256": SOURCE_PROVENANCE_SHA,
        "source_summary_sha256": SOURCE_SUMMARY_SHA,
        "geometry_manifest_sha256": GEOMETRY_MANIFEST_SHA,
        "hvi_manifest_sha256": digest(hvi_cache / "manifest.json"),
        "flow_manifest_sha256": digest(flow_cache / "manifest.json"),
    }
    return entries, split, lineage


def load_clip_bundle(entry: dict, allowed_scenes: set) -> dict:
    import torch

    _require(entry["scene"] in allowed_scenes, f"Refusing unauthorized scene access: {entry['scene']}")
    _require(digest(entry["source_path"]) == entry["source_sha256"], "Source clip SHA mismatch")
    _require(digest(entry["hvi_path"]) == entry["hvi_sha256"], "HVI clip SHA mismatch")
    _require(digest(entry["flow_path"]) == entry["flow_sha256"], "Flow clip SHA mismatch")

    src = torch.load(str(entry["source_path"]), map_location="cpu", weights_only=True)
    hvi = torch.load(str(entry["hvi_path"]), map_location="cpu", weights_only=True)
    flw = torch.load(str(entry["flow_path"]), map_location="cpu", weights_only=True)

    # Convert native BGR -> canonical RGB
    low_rgb = src["tensors"]["lq"].flip(1).contiguous()
    gt_rgb = src["tensors"]["gt"].flip(1).contiguous()
    base_rgb = hvi["base"].contiguous()
    flows = flw["flows_f16"].float().contiguous()

    _require(tensor_digest(low_rgb) == entry["rgb_lq_tensor_sha256"], "Canonical RGB LQ digest mismatch")
    _require(tensor_digest(base_rgb) == entry["base_tensor_sha256"], "Canonical RGB base digest mismatch")
    return {
        "name": entry["name"],
        "scene": entry["scene"],
        "low": low_rgb,
        "base": base_rgb,
        "gt": gt_rgb,
        "flows": flows,
    }


def scene_bootstrap_ci(
    rows: List[dict],
    method_key: str,
    ref_key: str,
    metric_keys: Tuple[str, ...] = ("psnr_y", "psnr_rgb", "decoupled_psnr_rgb", "lpips", "tlp_gap"),
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
        out[mk] = {
            "mean_delta": float(scene_deltas.mean()),
            "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)),
            "positive_scene_count": int(np.sum(scene_deltas > 0.0 if "psnr" in mk else scene_deltas < 0.0)),
            "total_scenes": n_scenes,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-cache",
        type=Path,
        default=Path("/cache/zihaohe/seer_method_dev_20261111/source_transfer_v1/cache_sdsd_p_s0"),
    )
    parser.add_argument(
        "--hvi-cache",
        type=Path,
        default=Path("/cache/zihaohe/seer_color_contract_20261112/cache_rgb_hvi_v1"),
    )
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        default=Path("/cache/zihaohe/seer_observation_geometry_20261112/prepared_v1/manifest.json"),
    )
    parser.add_argument("--flow-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", type=str, nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--smoke", action="store_true", help="4-clip 1-epoch timing smoke test on FIT only")
    args = parser.parse_args()

    _require(not args.out.exists(), f"Output directory already exists: {args.out}")
    for v in args.variants:
        _require(v in VARIANTS, f"Unknown variant {v!r}")

    vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    _require(vis in {"0", "1", "2", "3", "4", "5", "6", "7"}, "Set explicit single ASCEND_RT_VISIBLE_DEVICES")
    smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    _require(f"No running processes found in NPU {vis}" in smi, f"Physical NPU {vis} is not idle")

    import torch
    import torch_npu  # noqa: F401
    import pyiqa
    from eid_transport import (
        EIDTrajectoryTransport,
        compute_cycle_geometry,
        compute_eid_training_loss,
        fixed_baseline_controls,
    )

    torch.set_num_threads(4)
    device = torch.device("npu:0")

    entries, split, lineage = load_bound_partition(
        args.source_cache, args.hvi_cache, args.flow_cache, args.geometry_manifest
    )
    fit_scenes = set(split["fit"])
    holdout_scenes = set(split["internal_holdout"])

    fit_entries = [e for e in entries if e["scene"] in fit_scenes]
    _require(len(fit_entries) == 102, "Expected 102 FIT clips")
    if args.smoke:
        fit_entries = fit_entries[:4]

    args.out.mkdir(parents=True, exist_ok=False)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    print(f"Loading {len(fit_entries)} FIT clips into host memory...", flush=True)
    t_load0 = time.monotonic()
    fit_data = [load_clip_bundle(e, fit_scenes) for e in fit_entries]
    print(f"Loaded {len(fit_data)} FIT clips in {time.monotonic() - t_load0:.2f}s", flush=True)

    run_variants = ["eid_full"] if args.smoke else list(args.variants)
    run_seeds = [0] if args.smoke else list(args.seeds)
    run_epochs = 1 if args.smoke else args.epochs

    trained_models: Dict[str, EIDTrajectoryTransport] = {}
    training_logs = []

    t_train_all = time.monotonic()
    for variant_name in run_variants:
        arch_mode, loss_mode = VARIANTS[variant_name]
        for seed in run_seeds:
            tag = f"{variant_name}_s{seed}"
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(10000 + seed)

            model = EIDTrajectoryTransport(mode=arch_mode).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            steps_per_epoch = math.ceil(len(fit_data) / args.batch_size)
            total_steps = max(run_epochs * steps_per_epoch, 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

            t0 = time.monotonic()
            epoch_history = []
            step_count = 0
            for ep in range(run_epochs):
                ep_t0 = time.monotonic()
                perm = rng.permutation(len(fit_data))
                model.train()
                ep_losses = []
                for start in range(0, len(perm), args.batch_size):
                    batch_idx = perm[start:start + args.batch_size]
                    low_b = torch.cat([fit_data[int(i)]["low"] for i in batch_idx], dim=0).to(device)
                    base_b = torch.cat([fit_data[int(i)]["base"] for i in batch_idx], dim=0).to(device)
                    gt_b = torch.cat([fit_data[int(i)]["gt"] for i in batch_idx], dim=0).to(device)
                    flows_b = torch.cat([fit_data[int(i)]["flows"] for i in batch_idx], dim=0).to(device)

                    geom_b = compute_cycle_geometry(flows_b)
                    optimizer.zero_grad(set_to_none=True)
                    out = model(low_b, base_b, flows_b, geom=geom_b)
                    loss_dict = compute_eid_training_loss(
                        pred=out["pred_pixels"],
                        base=base_b,
                        gt=gt_b,
                        flows=flows_b,
                        geom=geom_b,
                        loss_mode=loss_mode,
                    )
                    loss_dict["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    step_count += 1
                    ep_losses.append({k: float(v.item()) for k, v in loss_dict.items()})

                ep_summary = {
                    "epoch": ep + 1,
                    "elapsed_sec": time.monotonic() - ep_t0,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    **{k: float(np.mean([x[k] for x in ep_losses])) for k in ep_losses[0]},
                }
                epoch_history.append(ep_summary)

            model.eval().requires_grad_(False)
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt_path = ckpt_dir / f"{tag}.pt"
            ckpt_payload = {
                "tag": tag,
                "variant": variant_name,
                "arch_mode": arch_mode,
                "loss_mode": loss_mode,
                "seed": seed,
                "epochs": run_epochs,
                "steps": step_count,
                "config": model.get_config(),
                "state_dict": cpu_state,
                "state_sha256": state_digest(cpu_state),
                "lineage": lineage,
            }
            torch.save(ckpt_payload, str(ckpt_path))
            trained_models[tag] = model

            rec = {
                "tag": tag,
                "variant": variant_name,
                "arch_mode": arch_mode,
                "loss_mode": loss_mode,
                "seed": seed,
                "steps": step_count,
                "elapsed_seconds": time.monotonic() - t0,
                "sec_per_step": (time.monotonic() - t0) / max(step_count, 1),
                "checkpoint": str(ckpt_path),
                "checkpoint_sha256": digest(ckpt_path),
                "state_sha256": ckpt_payload["state_sha256"],
                "final_epoch": epoch_history[-1],
                "epoch_history": epoch_history,
            }
            training_logs.append(rec)
            print(
                json.dumps(
                    {
                        "trained": tag,
                        "elapsed_s": round(rec["elapsed_seconds"], 2),
                        "sec_per_step": round(rec["sec_per_step"], 4),
                        "final_loss": round(rec["final_epoch"]["total"], 5),
                    }
                ),
                flush=True,
            )

    if args.smoke:
        write_new_json(
            args.out / "smoke_report.json",
            {
                "smoke": True,
                "internal_holdout_opened": False,
                "target_inputs_opened": False,
                "runs": training_logs,
            },
        )
        return

    # Free FIT training memory before evaluating internal17 holdout
    del fit_data

    print("Evaluating converged checkpoints + fixed controls on SDSD internal17 (34 clips)...", flush=True)
    lpips_metric = pyiqa.create_metric("lpips", device=device, as_loss=False).eval()
    holdout_entries = [e for e in entries if e["scene"] in holdout_scenes]
    _require(len(holdout_entries) == 34, "Expected 34 internal17 holdout clips")

    per_seq_holdout = []
    with torch.no_grad():
        for e in holdout_entries:
            bundle = load_clip_bundle(e, holdout_scenes)
            low_d = bundle["low"].to(device)
            base_d = bundle["base"].to(device)
            gt_d = bundle["gt"].to(device)
            flows_d = bundle["flows"].to(device)
            geom_d = compute_cycle_geometry(flows_d)

            methods_scores = {}
            fixed_preds = fixed_baseline_controls(base_d, flows_d, geom_d)
            for cname, cpred in fixed_preds.items():
                methods_scores[cname] = score_clip_metrics(cpred, base_d, gt_d, lpips_metric, device)

            # Multi-seed predictions and 3-seed mean metric per sequence
            for variant_name in run_variants:
                seed_metrics = []
                for seed in run_seeds:
                    tag = f"{variant_name}_s{seed}"
                    pred_s = trained_models[tag](low_d, base_d, flows_d, geom=geom_d)["pred_pixels"]
                    sc = score_clip_metrics(pred_s, base_d, gt_d, lpips_metric, device)
                    methods_scores[tag] = sc
                    seed_metrics.append(sc)
                methods_scores[f"{variant_name}_mean3"] = {
                    mk: float(np.mean([sm[mk] for sm in seed_metrics])) for mk in seed_metrics[0]
                }

            per_seq_holdout.append(
                {
                    "name": bundle["name"],
                    "scene": bundle["scene"],
                    "methods": methods_scores,
                }
            )

    all_method_keys = list(per_seq_holdout[0]["methods"].keys())
    metric_names = ("psnr_y", "psnr_rgb", "decoupled_psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")
    means = {
        mk_method: {
            m: float(np.mean([r["methods"][mk_method][m] for r in per_seq_holdout])) for m in metric_names
        }
        for mk_method in all_method_keys
    }

    bootstrap_vs_anchor = {
        f"{v}_mean3": scene_bootstrap_ci(per_seq_holdout, f"{v}_mean3", "anchor") for v in run_variants
    }
    bootstrap_vs_fixed_mean = {
        f"{v}_mean3": scene_bootstrap_ci(per_seq_holdout, f"{v}_mean3", "fixed_temporal_mean_0.5")
        for v in run_variants
    }

    summary = {
        "schema_version": 1,
        "scope": "SDSD FIT51 training + SDSD internal17 holdout evaluation; zero target-domain access",
        "lineage": lineage,
        "hyperparameters": {
            "epochs": run_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seeds": run_seeds,
            "variants": run_variants,
            "parameters_per_model": int(next(iter(trained_models.values())).get_config()["parameters"]),
        },
        "total_elapsed_seconds": time.monotonic() - t_train_all,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_runs": training_logs,
        "internal17_n_clips": len(per_seq_holdout),
        "internal17_n_scenes": 17,
        "internal17_means": means,
        "internal17_bootstrap_vs_anchor": bootstrap_vs_anchor,
        "internal17_bootstrap_vs_fixed_temporal_mean": bootstrap_vs_fixed_mean,
        "internal17_per_sequence": per_seq_holdout,
    }
    write_new_json(args.out / "source_train_and_internal17_summary.json", summary)
    print(json.dumps({"internal17_means": means}, indent=2), flush=True)


if __name__ == "__main__":
    main()
