"""Source-only observation controls and two-step trajectory reconstruction.

Prepare immutable observable tensors once. Train every final head at a fixed
budget, without intermediate validation or target-dependent selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from train_correction_selector import digest, write_json, local_module, state_digest, fit_normalization
from train_source_selector import cache_manifest, make_scene_split
from cache_source_corrections import source_hashes, check_source_hashes, source_scene

VARIANTS = {
    "raw_node": ("raw_detail", "node_only"),
    "same_frame_node": ("same_frame_detail", "node_only"),
    "mismatched_node": ("mismatched_detail", "node_only"),
    "trajectory": ("raw_detail", "conditioned"),
    "trajectory_fixed_edge": ("raw_detail", "fixed_edge"),
    "trajectory_no_cross": ("raw_detail", "no_cross"),
    "trajectory_scrambled_cross": ("raw_detail", "scrambled_cross"),
}
CANDIDATES = ("raw_detail", "same_frame_detail", "mismatched_detail")
STEPS, CROP, CLIP_LEN, LOSS_FRAMES, LR = 1000, 128, 16, 8, 1e-3
EXTRA_SOURCES = ("tools/train_observation_v2.py", "models/observation_controls.py",
                 "models/trajectory_reconstruction.py", "tools/cache_adjacent_flows.py",
                 "tools/train_source_selector.py", "models/observation_candidates.py",
                 "models/correction_selector.py", "models/correction_risk.py")


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def modules():
    return {name: local_module("_observation_v2_" + name, CODE / "models" / filename)
            for name, filename in (("control", "observation_controls.py"),
                                   ("selector", "correction_selector.py"),
                                   ("risk", "correction_risk.py"),
                                   ("trajectory", "trajectory_reconstruction.py"))}


def new_local_directory(path):
    path = Path(path)
    if not path.resolve().is_relative_to(Path("/cache")):
        raise ValueError("Write experiment tensors under host-local /cache, not shared SFS")
    path.mkdir(parents=True, exist_ok=False)


def prepare(cache, adjacent, out, splits):
    import torch
    torch.set_num_threads(4)
    files, origin, rows = cache_manifest(cache, splits)
    adjacent, out = Path(adjacent), Path(out)
    ap = json.loads((adjacent / "provenance.json").read_text())
    summary = json.loads((adjacent / "cache_summary.json").read_text())
    adjacent_provenance_sha = digest(adjacent / "provenance.json")
    if (ap["source_cache"]["provenance"] != origin
            or ap["source_cache"]["summary"] != json.loads((Path(cache) / "cache_summary.json").read_text())
            or ap["source_cache"]["provenance_sha256"] != digest(Path(cache) / "provenance.json")
            or ap["source_cache"]["summary_sha256"] != digest(Path(cache) / "cache_summary.json")
            or ap["tool_sha256"] != digest(CODE / "tools/cache_adjacent_flows.py")
            or ap["args"]["offsets"] != [-1, 1] or ap["args"]["flow_steps"] != 12
            or ap["args"]["flow_batch"] != 2 or ap["raft_weights_sha256"] != origin["raft_weights_sha256"]
            or ap["raft_state_sha256_before"] != origin["raft_state_sha256"]
            or summary["provenance_sha256"] != adjacent_provenance_sha
            or summary["raft_state_sha256_before"] != origin["raft_state_sha256"]
            or summary["raft_state_sha256_after"] != origin["raft_state_sha256"]
            or summary["n"] != 136 or summary["raft_state_unchanged"] is not True
            or ap["gt_input"] is not False or ap["target_input"] is not False
            or ap["metric_computed"] is not False
            or len(summary["per_sequence"]) != 136
            or [r["name"] for r in summary["per_sequence"]] != origin["source_names"]):
        raise ValueError("Adjacent flows do not belong to the completed source cache")
    new_local_directory(out)
    m = modules()
    split = make_scene_split(origin["source_names"])
    fit = [i * 16 + t for i, n in enumerate(origin["source_names"])
           if source_scene(n) in split["fit"] for t in range(16)]
    arrays = {c: {k: [] for k in ("features", "a", "gram", "e0")} for c in CANDIDATES}
    recipes, output_rows = {}, []
    for i, (path, row, ar) in enumerate(zip(files, rows, summary["per_sequence"])):
        flow_path = adjacent / ar["cache_path"]
        if (ar["cache_path"] != f"adjacent_{i:03d}.pt" or ar["scene"] != row["scene"]
                or digest(flow_path) != ar["cache_sha256"] or ar["source_cache_sha256"] != row["cache_sha256"]):
            raise ValueError("Adjacent/source file hash differs")
        sample = torch.load(path, map_location="cpu", weights_only=True)
        adj = torch.load(flow_path, map_location="cpu", weights_only=True)
        t = sample["tensors"]
        base, low, gt = (t[k] for k in ("base", "lq", "gt"))
        if (gt.shape != base.shape or gt.dtype != torch.float32 or not bool(torch.isfinite(gt).all())
                or not bool(((gt >= 0) & (gt <= 1)).all())):
            raise ValueError("Invalid source GT loss labels")
        if (adj["name"] != row["name"] or adj["scene"] != row["scene"]
                or adj["provenance_sha256"] != adjacent_provenance_sha
                or adj["plan"] != sample["plan"] or adj["source_cache_sha256"] != row["cache_sha256"]
                or adj["base_tensor_sha256"] != tensor_digest(base)
                or ar["base_tensor_sha256"] != tensor_digest(base)):
            raise ValueError("Adjacent tensor lineage differs")
        candidates, features = {}, {}
        with torch.no_grad():
            e0 = m["selector"]._patch_pool((gt - base).square(), 8).mean(2)
            for c in CANDIDATES:
                d, mask, recipe = m["control"].build_observation_controls(
                    low, base, t["flow"], t["valid_mask"], mode=c)
                if c in recipes and recipes[c] != recipe:
                    raise ValueError("Candidate construction differs across clips")
                recipes[c] = recipe
                x, g = m["selector"].observable_patch_features(low, base, d, mask, 8)
                a, checked = m["risk"].patch_moments(gt - base, d, 8)
                torch.testing.assert_close(g, checked, atol=1e-8, rtol=1e-5)
                for key, value in (("features", x[0]), ("a", a[0]), ("gram", g[0]), ("e0", e0[0])):
                    arrays[c][key].append(value.numpy())
                candidates[c], features[c] = d.contiguous(), x.contiguous()
            lp = m["trajectory"].lowpass
            low_l, base_l = lp(low), lp(base)
            q = (base_l + (base_l / (low_l + 1e-4)).clamp(.25, 16) * (low - low_l)).clamp(0, 1)
            detail = (q - lp(q)).contiguous()
            # Validate adjacent geometry with the same CPU graph used in training.
            m["trajectory"].build_bilinear_graph(adj["tensors"]["flow"], adj["tensors"]["valid_mask"])
        payload = {"name": row["name"], "scene": row["scene"], "plan": sample["plan"],
                   "source_cache_sha256": row["cache_sha256"], "adjacent_cache_sha256": ar["cache_sha256"],
                   "base": base.contiguous(), "gt": gt.contiguous(), "corrections": candidates,
                   "features": features, "detail_observations": detail,
                   "adjacent_flow": adj["tensors"]["flow"].contiguous(),
                   "adjacent_mask": adj["tensors"]["valid_mask"].contiguous()}
        target = out / f"prepared_{i:03d}.pt"
        torch.save(payload, target)
        output_rows.append({"name": row["name"], "scene": row["scene"], "cache_path": target.name,
                            "cache_sha256": digest(target), "source_cache_sha256": row["cache_sha256"],
                            "adjacent_cache_sha256": ar["cache_sha256"]})
        print(json.dumps({"prepared": i + 1, "name": row["name"]}), flush=True)
    norms = {}
    for c in CANDIDATES:
        values = {k: np.concatenate(v) for k, v in arrays[c].items()}
        norms[c] = fit_normalization(values["features"], values["a"], values["gram"], fit, e0=values["e0"])
    metadata = {"schema_version": 1, "domain": "sdsd", "n": 136, "source_cache_provenance": origin,
                "adjacent_provenance": ap, "adjacent_summary_sha256": digest(adjacent / "cache_summary.json"),
                "source_cache_summary_sha256": digest(Path(cache) / "cache_summary.json"),
                "split": split, "candidate_recipes": recipes, "normalization": norms,
                "per_sequence": output_rows, "variants": {k: list(v) for k, v in VARIANTS.items()},
                "source_sha256": source_hashes(*EXTRA_SOURCES),
                "gt_firewall": "GT only supplies source reconstruction loss and fit loss-scale labels; never graph/flow/candidate/features",
                "oracle_computed": False, "target_frames_read": False}
    write_json(out / "prepared_manifest.json", metadata)


def make_plan(names, split, seed):
    fit = np.array([i for i, n in enumerate(names) if source_scene(n) in split["fit"]])
    rng = np.random.default_rng(seed)
    clips = rng.choice(fit, size=STEPS, replace=True)
    return {"seed": seed, "steps": STEPS, "crop": CROP, "loss_frames": LOSS_FRAMES,
            "source_names": names, "batches": [
                {"clip_index": int(i), "frames": rng.choice(16, size=LOSS_FRAMES, replace=False).tolist(),
                 "top": int(rng.integers(0, (256 - CROP) // 8 + 1)) * 8,
                 "left": int(rng.integers(0, (256 - CROP) // 8 + 1)) * 8} for i in clips]}


def crop_sample(sample, candidate, spec, normalization):
    import torch
    top, left = spec["top"], spec["left"]
    view = (..., slice(top, top + CROP), slice(left, left + CROP))
    base, gt, detail, flow, mask = (sample[k][view].clone() for k in
        ("base", "gt", "detail_observations", "adjacent_flow", "adjacent_mask"))
    d = sample["corrections"][candidate][view].clone()
    x = sample["features"][candidate][..., top // 8:(top + CROP) // 8,
                                             left // 8:(left + CROP) // 8].clone()
    yy, xx = torch.meshgrid(torch.arange(CROP), torch.arange(CROP), indexing="ij")
    inside = ((xx + flow[:, :, 0] >= 0) & (xx + flow[:, :, 0] <= CROP - 1)
              & (yy + flow[:, :, 1] >= 0) & (yy + flow[:, :, 1] <= CROP - 1))
    mask &= inside[:, :, None]
    mean = torch.tensor(normalization["feature_mean"])[None, None, :, None, None]
    std = torch.tensor(normalization["feature_std"])[None, None, :, None, None]
    return (base, d, (x - mean) / std, detail, flow, mask), gt


def train(prepared, out, variant, seed):
    import torch
    torch.set_num_threads(4)
    prepared, out = Path(prepared), Path(out)
    meta = json.loads((prepared / "prepared_manifest.json").read_text())
    check_source_hashes(meta["source_sha256"])
    names = [r["name"] for r in meta["per_sequence"]]
    if (meta["n"] != 136 or len(names) != 136 or len(set(names)) != 136
            or meta["oracle_computed"] is not False or meta["target_frames_read"] is not False
            or names != meta["source_cache_provenance"]["source_names"]
            or meta["split"] != make_scene_split(names)
            or meta["source_sha256"] != source_hashes(*EXTRA_SOURCES)
            or meta["variants"] != {k: list(v) for k, v in VARIANTS.items()}):
        raise ValueError("Require completed source-only prepared136 cache")
    for i, row in enumerate(meta["per_sequence"]):
        if row["cache_path"] != f"prepared_{i:03d}.pt" or digest(prepared / row["cache_path"]) != row["cache_sha256"]:
            raise ValueError("Prepared tensor SHA mismatch")
    new_local_directory(out)
    candidate, graph_mode = VARIANTS[variant]
    norm = meta["normalization"][candidate]
    plan = make_plan([r["name"] for r in meta["per_sequence"]], meta["split"], seed)
    write_json(out / "sample_plan.json", plan)
    torch.manual_seed(seed)
    graph = modules()["trajectory"]
    model = graph.TrajectoryReconstruction(mode=graph_mode).train()
    initial_sha = state_digest(model.state_dict())
    metadata = {"schema_version": 1, "variant": variant, "seed": seed, "candidate": candidate,
                "args": {"steps": STEPS, "lr": LR, "optimizer": "Adam default betas/eps; no weight decay",
                         "device": "cpu", "threads": 4, "clips_per_update": 1,
                         "clip_len": 16, "loss_frames": LOSS_FRAMES, "source_crop": CROP,
                         "target_crop": 256, "loss": "source RGB MSE / fit-only mean base MSE"},
                "model_config": model.get_config(), "initial_state_sha256": initial_sha,
                "sample_plan_sha256": digest(out / "sample_plan.json"), "normalization": norm,
                "candidate_recipe": meta["candidate_recipes"][candidate],
                "prepared_manifest_sha256": digest(prepared / "prepared_manifest.json"),
                "source_names": [r["name"] for r in meta["per_sequence"]],
                "source_scene_split": meta["split"],
                "base_checkpoint_sha256": meta["source_cache_provenance"]["base_checkpoint_sha256"],
                "source_split_sha256": meta["source_cache_provenance"]["split_sha256"],
                "source_sha256": source_hashes(*EXTRA_SOURCES),
                "allocated_parameters": sum(p.numel() for p in model.parameters()),
                "selection": "Final1000 only; no intermediate source holdout or target selection",
                "initial_function": "All graph modes begin with edge0.5; node-only is a distinct initial function",
                "runtime": {"torch": str(torch.__version__), "numpy": str(np.__version__)}}
    write_json(out / "provenance.json", metadata)
    torch.save({"model": copy.deepcopy(model.state_dict())}, out / "initial_state.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.monotonic()
    with (out / "curve.jsonl").open("w") as stream:
        for step, spec in enumerate(plan["batches"], 1):
            row = meta["per_sequence"][spec["clip_index"]]
            sample = torch.load(prepared / row["cache_path"], map_location="cpu", weights_only=True, mmap=True)
            inputs, gt = crop_sample(sample, candidate, spec, norm)
            optimizer.zero_grad(set_to_none=True)
            result = model(*inputs)
            frames = spec["frames"]
            mse = (result["pred_pixels"][:, :, frames] - gt[:, :, frames]).square().mean()
            loss = mse / norm["direct_mse_normalizer"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite source loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite source gradient")
            optimizer.step()
            entry = {"step": step, "loss": float(loss.detach()), "rgb_mse": float(mse.detach()),
                     "base_mse": float((inputs[0][:, :, frames] - gt[:, :, frames]).square().mean()),
                     "clip_index": spec["clip_index"], "elapsed_seconds": time.monotonic() - start,
                     "prediction_timing": "Before optimizer update", **result["trajectory_aux"]}
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            if step == 1 or step % 50 == 0:
                stream.flush(); print(json.dumps(entry), flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    checkpoint = out / "final.pt"
    torch.save({"model": state, "model_state_sha256": state_digest(state), "steps": STEPS,
                "provenance": metadata}, checkpoint)
    write_json(out / "final.sha256.json", {"checkpoint_sha256": digest(checkpoint),
               "model_state_sha256": state_digest(state), "initial_state_sha256": initial_sha,
               "sample_plan_sha256": metadata["sample_plan_sha256"]})
    write_json(out / "complete.json", {"steps": STEPS, "elapsed_seconds": time.monotonic() - start,
               "checkpoint_sha256": digest(checkpoint), "selection": metadata["selection"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--adjacent", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.cache, args.adjacent, args.out, args.splits)
    else:
        train(args.prepared, args.out, args.variant, args.seed)


if __name__ == "__main__":
    main()
