"""Fixed source FIT12 donor-value utility diagnostic; no learned RGB predictor.

Prediction helpers accept only RGB observables. Ground truth is opened only by
label_source(), after all four observable predictions have been constructed.
Outputs contain patch statistics, never image/GT arrays.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import io
import math as pymath
import sys
import zipfile
import json
from pathlib import Path
import time

import numpy as np

PHASE = Path(__file__).resolve().parent
GAMMA = .5
PATCH = 16
TARGET_FRAMES = (8, 9)
METHODS = ("base", "single1", "single2", "pair", "sameframe_zero", "temporal_no_dc")
RGB_RUNNER_SHA = "00b039e00a39812d2ae3e0d202c2436e6542ba6b964bf5c47cd78c78285d4fa2"
MATH_SHA = "c314d9b37f71009af0f4974cf59b3bcc41b0d48698184c9ec01de66a7caecd54"
COLOR_SHA = "7815887150b2e92ed7c2cf4c30f334a4a0c435d58b4ba017fd1b746a97dcab60"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def module_at(name, path, expected):
    require(digest(path) == expected, "Frozen dependency SHA differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def as_patches(value, patch=PATCH):
    """[C,T,H,W] -> [patch_y,patch_x,C,T,patch,patch]."""
    c, t, h, w = value.shape
    require(h % patch == w % patch == 0, "Require complete nonoverlapping patches")
    return value.reshape(c, t, h // patch, patch, w // patch, patch).permute(2, 4, 0, 1, 3, 5)


def from_patches(value):
    py, px, c, t, ph, pw = value.shape
    return value.permute(2, 3, 0, 4, 1, 5).reshape(c, t, py * ph, px * pw)


def patch_mean(value):
    """Per-patch scalar mean, including channels and both target frames."""
    return as_patches(value).double().mean(dim=(2, 3, 4, 5))


def center_directions(base, donors, valid):
    """Each actual donor residual loses only per-frame/channel support DC."""
    directions = []
    for a, m in zip(donors, valid):
        residual = as_patches(a - base).double()
        support = as_patches(m).double()
        n = support.sum(dim=(-2, -1), keepdim=True)
        mean = (residual * support).sum(dim=(-2, -1), keepdim=True) / n.clamp_min(1)
        directions.append(from_patches((residual - mean) * support).to(base.dtype))
    return directions


def project_box_fixed_sum(candidate, base, support):
    """Exact box/fixed-sum Euclidean projection in float64, cast once to RGB32.

    Solve y_i=clip(candidate_i+c,0,1), sum_valid(y)=sum_valid(base)
    independently for each patch, channel and frame. Support-excluded elements
    are returned from base with exact identity. Float32 representation leaves
    only the final rounding tolerance; no arbitrary postprojection redistribution.
    """
    import torch
    require(candidate.shape == base.shape and support.shape == (1, *base.shape[1:]), "Projection shape differs")
    if torch.equal(candidate, base):
        return base.clone()
    u, b = as_patches(candidate).double(), as_patches(base).double()
    m = as_patches(support).bool()
    goal = (b * m).sum(dim=(-2, -1), keepdim=True)
    lo = -u.amax(dim=(-2, -1), keepdim=True) - 1.
    hi = 1. - u.amin(dim=(-2, -1), keepdim=True) + 1.
    for _ in range(80):
        mid = (lo + hi) * .5
        actual = (torch.clamp(u + mid, 0., 1.) * m).sum(dim=(-2, -1), keepdim=True)
        lo = torch.where(actual < goal, mid, lo)
        hi = torch.where(actual < goal, hi, mid)
    projected = torch.clamp(u + (lo + hi) * .5, 0., 1.)
    # Empty supports and already-identical candidates remain exactly base.
    return from_patches(torch.where(m, projected, b)).to(base.dtype)


def warp_two(values, flows, masks, target_frames=TARGET_FRAMES):
    """Same pinned legacy grid math, retaining the two individual donors."""
    import torch
    from torch.nn import functional as F
    require(values.ndim == 5 and values.shape[0:2] == (1, 3) and values.dtype == torch.float32,
            "Require CPU B1 RGB float32 video")
    _, _, t, h, w = values.shape
    require(flows.shape == (1, 2, 2, t, h, w) and masks.shape == (1, 2, 1, t, h, w), "Flow/mask shape differs")
    require(all(x.device.type == "cpu" and bool(torch.isfinite(x).all()) for x in (values, flows, masks)),
            "Require finite CPU values/flow/mask")
    require(bool(((masks == 0) | (masks == 1)).all()), "Require binary masks")
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    times = torch.tensor(target_frames, dtype=torch.long)
    warped, validity = [], []
    for j, delta in enumerate((-2, 2)):
        st = times + delta
        require(bool(((st >= 0) & (st < t)).all()), "Donor frames outside clip")
        x = xx[None] + flows[0, j, 0].index_select(0, times)
        y = yy[None] + flows[0, j, 1].index_select(0, times)
        valid = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))[None] & masks[0, j].index_select(1, times).bool()
        grid = torch.stack((2 * (x + .5) / w - 1, 2 * (y + .5) / h - 1), -1)
        source = values[0].index_select(1, st).permute(1, 0, 2, 3)
        result = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped.append(result.permute(1, 0, 2, 3))
        validity.append(valid)
    return warped, validity


def observable_scores(base, low, donors, low_donors, valid):
    features = {}
    for d, (a, x, m) in enumerate(zip(donors, low_donors, valid), start=1):
        sp = as_patches(m).double()
        den = sp.sum(dim=(2, 3, 4, 5)) * 3
        for key, residual in (("lq", x - low), ("hvi", a - base)):
            numerator = (as_patches(residual).square().double() * sp).sum(dim=(2, 3, 4, 5))
            # Defined zero for no-valid-support patches; support fraction is saved.
            features[f"feature_{key}{d}"] = -numerator / den.clamp_min(1)
        features[f"feature_valid{d}"] = sp.mean(dim=(2, 3, 4, 5))
    return features


def prepare_values(base_rgb, low_rgb, flows, masks):
    """GT-free central-tubelet predictions, observable features and constraints."""
    import torch
    require(base_rgb.shape == low_rgb.shape and base_rgb.shape[2] == 16, "Require matched 16-frame observables")
    require(all(bool(((x >= 0) & (x <= 1) & torch.isfinite(x)).all()) for x in (base_rgb, low_rgb)),
            "Require canonical RGB unit-range inputs")
    donors, valid = warp_two(base_rgb, flows, masks)
    low_donors, low_valid = warp_two(low_rgb, flows, masks)
    require(all(torch.equal(a, b) for a, b in zip(valid, low_valid)), "LQ/HVI geometry differs")
    b, low = base_rgb[0, :, TARGET_FRAMES], low_rgb[0, :, TARGET_FRAMES]
    directions = center_directions(b, donors, valid)
    union = valid[0] | valid[1]
    count = valid[0].float() + valid[1].float()
    pair_direction = (directions[0] + directions[1]) / count.clamp_min(1)
    predictions = {"base": b}
    for i in range(2):
        predictions[f"single{i+1}"] = project_box_fixed_sum(b + GAMMA * directions[i], b, union)
    predictions["pair"] = project_box_fixed_sum(b + GAMMA * pair_direction, b, union)
    zero = center_directions(b, (b, b), valid)
    predictions["sameframe_zero"] = project_box_fixed_sum(b + GAMMA * (zero[0] + zero[1]), b, union)
    mean = (donors[0] * valid[0] + donors[1] * valid[1]) / count.clamp_min(1)
    predictions["temporal_no_dc"] = (b + GAMMA * (mean - b) * union).clamp(0, 1)
    require(torch.equal(predictions["sameframe_zero"], b), "Sameframe zero identity failed")
    max_dc = max(float(((as_patches(p).double() - as_patches(b).double()) * as_patches(union)).sum(
        dim=(-2, -1)).abs().max()) for k, p in predictions.items() if k != "temporal_no_dc")
    outside = all(torch.equal(p.masked_select(~union.expand_as(p)), b.masked_select(~union.expand_as(b)))
        for k, p in predictions.items() if k != "temporal_no_dc")
    require(max_dc <= 2e-5 and outside, "DC/support constraint failed")
    require(all(bool(((p >= 0) & (p <= 1) & torch.isfinite(p)).all()) for p in predictions.values()), "Invalid prediction range")
    return predictions, observable_scores(b, low, donors, low_donors, valid), {
        "max_abs_patch_frame_channel_dc_sum_error": max_dc,
        "support_outside_exact_anchor": outside, "sameframe_exact_anchor": True,
        "no_donor_fraction": float((count == 0).float().mean()),
        "one_donor_fraction": float((count == 1).float().mean()),
        "two_donor_fraction": float((count == 2).float().mean())}


def score_utilities(predictions, gt_rgb):
    """Label-only stage; patch RGB MSE averages both target frames equally."""
    require(tuple(predictions) == METHODS, "Fixed diagnostic methods changed")
    require(gt_rgb.shape == predictions["base"].shape, "Utility label shape differs")
    out = {"mse_" + k: patch_mean((p - gt_rgb).square()) for k, p in predictions.items()}
    out.update(u1=out["mse_base"] - out["mse_single1"],
               u2=out["mse_base"] - out["mse_single2"],
               u12=out["mse_base"] - out["mse_pair"],
               u1_given2=out["mse_single2"] - out["mse_pair"],
               u2_given1=out["mse_single1"] - out["mse_pair"])
    return out


def require_selected(row, selected_names, fit_scenes):
    require(row["name"] in set(selected_names) and row["scene"] in set(fit_scenes)
            and row["name"] == row["scene"], "Refusing excluded/non-pass1 source access")


def read_observables(path, reader, *, archive_class=zipfile.ZipFile, expected_shapes=None):
    """Read exactly LQ/flow/validity storages. GT and base stores stay closed."""
    import torch
    expected = {"lq": (1, 3, 16, 256, 256), "flow": (1, 2, 2, 16, 256, 256),
                "valid_mask": (1, 2, 1, 16, 256, 256)}
    expected = expected_shapes or expected
    opened, tensors = [], {}
    with archive_class(path) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if n.endswith("/data.pkl")]
        require(len(metadata_names) == 1, "Unexpected archive metadata")
        root = metadata_names[0][:-len("data.pkl")]
        def read(name):
            opened.append(name)
            return archive.read(name)
        meta = reader.MetadataOnlyUnpickler(io.BytesIO(read(metadata_names[0]))).load()
        require(isinstance(meta, dict) and isinstance(meta.get("tensors"), dict), "Source schema differs")
        order_name = root + "byteorder"
        require((read(order_name).decode() if order_name in names else "little") == "little", "Storage byteorder differs")
        allowed_paths = {( "tensors", key) for key in expected}
        specs = dict(reader.tensor_specs(meta))
        for key, shape in expected.items():
            spec = meta["tensors"][key]
            require(isinstance(spec, reader.TensorSpec) and spec.size == shape and spec.offset == 0
                    and spec.storage.dtype in ("float32", "bool") and spec.storage.numel == pymath.prod(shape), "Observable storage layout differs")
            if key in ("lq", "flow"):
                require(spec.storage.dtype == "float32", "LQ/flow dtype differs")
            require(all(path in allowed_paths or other.storage.key != spec.storage.key
                        for path, other in specs.items()), "Observable storage aliases excluded tensor")
            require(all(v >= 0 for v in spec.stride)
                    and sum((n-1)*v for n,v in zip(spec.size,spec.stride)) < spec.storage.numel,
                    "Observable stride escapes storage")
            raw = read(root + "data/" + spec.storage.key)
            dtype = np.dtype("<f4" if spec.storage.dtype == "float32" else "?")
            require(len(raw) == spec.storage.numel * dtype.itemsize, "Observable storage length differs")
            array = np.ndarray(spec.size, dtype=dtype, buffer=raw,
                strides=tuple(v*dtype.itemsize for v in spec.stride)).copy(order="C")
            tensors[key] = torch.from_numpy(array)
    return str(meta["name"]), str(meta["scene"]), tensors, {
        "archive_entries_read": opened, "allowed_tensor_fields": list(expected), "gt_storage_entries_read": 0,
        "excluded_storage_entries_read": 0}


def observable_geometry(flows, valid, height=256, width=256):
    """Mean source-pixel location over both frames' actual valid support."""
    import torch
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    target = torch.tensor(TARGET_FRAMES)
    coords, fractions = [], []
    for j, m in enumerate(valid):
        mp = as_patches(m).double()
        den = mp.sum(dim=(2, 3, 4, 5))
        xy = []
        for axis, grid in enumerate((xx, yy)):
            absolute = grid[None] + flows[0, j, axis].index_select(0, target)
            ap = as_patches(absolute[None]).double()
            mean = (ap*mp).sum(dim=(2,3,4,5)) / den.clamp_min(1)
            fallback = as_patches(grid[None,None].expand(1,2,height,width)).double().mean(dim=(2,3,4,5))
            xy.append(torch.where(den > 0, mean, fallback).reshape(-1).tolist())
        coords.append(np.stack(xy,axis=-1).tolist())
        fractions.append(mp.mean(dim=(2,3,4,5)).reshape(-1).tolist())
    return coords, fractions


def label_source(row, selected_names, fit_scenes, color):
    """The sole production GT access, guarded before archive hashing/loading."""
    require_selected(row, selected_names, fit_scenes)
    import torch
    require(digest(row["source_path"]) == row["source_sha256"], "Source label archive changed")
    data = torch.load(row["source_path"], map_location="cpu", weights_only=True, mmap=True)
    require(data["name"] == row["name"] and data["scene"] == row["scene"], "Label identity changed")
    return color.convert_video(data["tensors"]["gt"], source_order="BGR", target_order="RGB")[0, :, TARGET_FRAMES]


def selected_entries(entries, subset, split):
    """Metadata-only filter before any source archive is hashed or opened."""
    names = [r["name"] for r in subset["entries"]]
    require(len(names) == len(set(names)) == 12, "Require exactly twelve fixed pass1 clips")
    expected = sum((sorted(s for s in split["fit"] if s.startswith(domain + "/"))[:6]
                    for domain in ("SDSD_indoor", "SDSD_outdoor")), [])
    require(names == expected and subset["split"] == split and subset["target_frames"] == [8, 9]
            and subset["patch_size"] == 16, "Subset differs from fixed lexical FIT12 rule")
    selected = [next(r for r in entries if r["name"] == n) for n in names]
    for row, declared in zip(selected, subset["entries"]):
        require_selected(row, names, split["fit"])
        require(all(row[key] == declared[key] for key in ("name", "scene", "source_sha256", "hvi_sha256", "native_lq_tensor_sha256", "rgb_lq_tensor_sha256"))
                and row["base_tensor_sha256"] == declared["hvi_base_tensor_sha256"], "Declared subset tensor identity differs")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-cache", "hvi-cache", "geometry-manifest", "subset", "out", "rgb-runner", "color-source", "lq-reader", "math-source", "test-receipt"):
        parser.add_argument("--" + key, required=True, type=Path)
    for key in ("expected-hvi-manifest-sha256", "expected-subset-sha256", "expected-test-receipt-sha256", "expected-rgb-runner-sha256", "expected-lq-reader-sha256"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--stage", choices=("observables", "utilities"), required=True)
    parser.add_argument("--observable-sidecar", type=Path)
    parser.add_argument("--expected-observable-sidecar-sha256")
    args = parser.parse_args()
    require(args.out.is_absolute() and args.out.resolve().is_relative_to(Path("/cache")) and not args.out.exists(), "Use a fresh /cache output")
    require(digest(args.subset) == args.expected_subset_sha256, "Subset SHA differs")
    require(digest(args.test_receipt) == args.expected_test_receipt_sha256, "Test receipt differs")
    tests = json.loads(args.test_receipt.read_text())
    require(tests["status"] == "passed" and tests["skips"] == 0, "Require passing real CPU tests without skips")
    require(tests["code_sha256"] == digest(__file__) and tests["test_sha256"] == digest(PHASE / "test_values_probe.py")
            and tests["protocol_sha256"] == digest(PHASE / "values_protocol.json"), "Tests do not bind release")
    rgb = module_at("_credit_probe_rgb_runner", args.rgb_runner, args.expected_rgb_runner_sha256)
    color = module_at("_credit_probe_color", args.color_source, COLOR_SHA)
    reader = module_at("_credit_probe_lq_reader", args.lq_reader, args.expected_lq_reader_sha256)
    math = module_at("_credit_probe_math", args.math_source, MATH_SHA)
    entries, split, lineage = rgb.bound_inputs(args.source_cache, args.hvi_cache, args.geometry_manifest,
                                              args.expected_hvi_manifest_sha256)
    subset = json.loads(args.subset.read_text())
    rows = selected_entries(entries, subset, split)
    selected_names = [row["name"] for row in rows]
    prior_sidecar = None
    if args.stage == "utilities":
        require(args.observable_sidecar is not None and args.expected_observable_sidecar_sha256 is not None,
                "Utilities require independently frozen GT-free observable sidecar")
        require(digest(args.observable_sidecar) == args.expected_observable_sidecar_sha256, "Observable sidecar SHA differs")
        prior_sidecar = json.loads(args.observable_sidecar.read_text())
        require(prior_sidecar["subset_sha256"] == args.expected_subset_sha256
                and prior_sidecar["gt_storage_entries_read"] == 0, "Observable sidecar lineage differs")
    import torch
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), PHASE / "test_values_probe.py", PHASE / "values_protocol.json", args.subset,
             args.rgb_runner, args.color_source, args.lq_reader, args.math_source, args.test_receipt]
    frozen = {str(p.resolve()): digest(p) for p in files}
    manifest = {"schema_version": 1, "scope": "Source FIT12 fixed utility probe; no trained module or generalization score",
        "files": frozen, "lineage": lineage, "subset": subset, "selected_entries": rows,
        "source_archive_access": "Only selected12; metadata globals permitted; predictions precede GT-storage access",
        "prediction_has_gt_parameter": bool({"gt", "labels"} & set(inspect.signature(prepare_values).parameters)),
        "internal_holdout_tensor_archives_opened": 0, "target_inputs_read": 0,
        "optimizer_or_fitting": False, "rgb_or_gt_arrays_written": False,
        "threads": 4, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(args.out / "execution_manifest.json", manifest)
    began, all_arrays, clips, observable_clips = time.monotonic(), {}, [], []
    with torch.no_grad():
        for row in rows:
            require_selected(row, selected_names, split["fit"])
            require(digest(row["source_path"]) == row["source_sha256"] and digest(row["hvi_path"]) == row["hvi_sha256"], "Selected input changed")
            # Strict selective reader reads source metadata and LQ storage only.
            name, scene, native, access = read_observables(Path(row["source_path"]), reader)
            native_low = native["lq"]
            require((name, scene) == (row["name"], row["scene"]), "Observable identity differs")
            low = color.convert_video(native_low, source_order="BGR", target_order="RGB")
            hvi = torch.load(row["hvi_path"], map_location="cpu", weights_only=True, mmap=True)
            require(hvi["name"] == name and hvi["scene"] == scene, "Observable payload identity differs")
            require(rgb.tensor_digest(native_low) == row["native_lq_tensor_sha256"]
                    and rgb.tensor_digest(low) == row["rgb_lq_tensor_sha256"]
                    and rgb.tensor_digest(hvi["base"]) == row["base_tensor_sha256"], "Observable tensor identity differs")
            predictions, features, constraints = prepare_values(hvi["base"], low, native["flow"], native["valid_mask"])
            correction, _ = math.aligned_hvi_mean(hvi["base"], native["flow"], native["valid_mask"])
            pinned = (hvi["base"] + GAMMA * correction).clamp(0, 1)[0, :, TARGET_FRAMES]
            require(torch.equal(pinned, predictions["temporal_no_dc"]), "Pinned legacy temporal prediction parity differs")
            _, valid = warp_two(hvi["base"], native["flow"], native["valid_mask"])
            coords, fractions = observable_geometry(native["flow"], valid)
            observable = {"name": name, "scene": scene, "target_tubelet": 4,
                "patch_grid_xy": coords, "valid_fraction": fractions,
                "feature_lq": [features[f"feature_lq{d}"].reshape(-1).tolist() for d in (1,2)],
                "feature_hvi": [features[f"feature_hvi{d}"].reshape(-1).tolist() for d in (1,2)],
                "archive_access": access}
            observable_clips.append(observable)
            if args.stage == "observables":
                print(json.dumps({"completed_observable": len(observable_clips), "name": name}), flush=True)
                continue
            require(observable == prior_sidecar["clips"][len(clips)], "Observable values changed before utility labels")
            gt = label_source(row, selected_names, split["fit"], color)
            arrays = {**score_utilities(predictions, gt), **features}
            py, px = next(iter(arrays.values())).shape
            for key, value in arrays.items():
                all_arrays.setdefault(key, []).append(value.reshape(-1).numpy())
            all_arrays.setdefault("patch_y", []).append(np.repeat(np.arange(py), px))
            all_arrays.setdefault("patch_x", []).append(np.tile(np.arange(px), py))
            all_arrays.setdefault("clip_index", []).append(np.full(py * px, len(clips), dtype=np.int64))
            clips.append({"name": name, "scene": scene, "target_tubelet": 4, "target_frames": [8, 9], "patches": py * px,
                "methods": {k: {"mean_rgb_mse": float(arrays["mse_"+k].mean()),
                    "psnr_rgb_from_central_two_frame_mse": float(-10*np.log10(float(arrays["mse_"+k].mean())))} for k in METHODS},
                "constraints": constraints})
            print(json.dumps({"completed": len(clips), "name": name, "patches": py * px}), flush=True)
    if args.stage == "observables":
        require(len(observable_clips) == 12, "Incomplete observable coverage")
        require(all(digest(p) == sha for p, sha in frozen.items()), "Frozen sources changed")
        sidecar = {"schema": "seer.predictive_credit_probe.observables.v1", "subset_sha256": args.expected_subset_sha256,
            "execution_manifest_sha256": digest(args.out / "execution_manifest.json"),
            "gt_storage_entries_read": 0, "target_inputs_read": 0, "internal_holdout_inputs_read": 0,
            "patch_grid_xy_units": "source absolute pixel centers x,y; 0..255; valid support mean over target frames8,9; no-support target-patch center fallback",
            "clips": observable_clips, "elapsed_seconds": time.monotonic()-began}
        write_new(args.out / "observable_sidecar.json", sidecar)
        print(json.dumps({"observables_complete": True, "gt_storage_entries_read": 0,
            "sidecar_sha256": digest(args.out / "observable_sidecar.json")}), flush=True)
        return
    arrays = {k: np.concatenate(v) for k, v in all_arrays.items()}
    require(len(arrays["u1"]) == 3072 and len(clips) == 12, "Incomplete fixed utility coverage")
    with (args.out / "patch_statistics.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    require(all(digest(p) == sha for p, sha in frozen.items()), "Frozen code/input changed")
    require(digest(args.hvi_cache / "manifest.json") == args.expected_hvi_manifest_sha256, "HVI manifest changed")
    report = {"schema_version": 1, "scope": manifest["scope"], "execution_manifest_sha256": digest(args.out / "execution_manifest.json"),
        "patch_statistics_sha256": digest(args.out / "patch_statistics.npz"), "rows": 3072, "clips": clips,
        "mean_central_two_frame_clip_psnr_rgb": {k: float(np.mean([c["methods"][k]["psnr_rgb_from_central_two_frame_mse"] for c in clips])) for k in METHODS},
        "mean_rgb_mse": {k: float(arrays["mse_"+k].mean()) for k in METHODS},
        "utility_summary": {k: {"mean": float(arrays[k].mean()), "positive_fraction": float((arrays[k] > 0).mean())} for k in ("u1", "u2", "u12", "u1_given2", "u2_given1")},
        "target_scores_computed": False, "internal_holdout_scores_computed": False, "parameter_selection": False,
        "elapsed_seconds": time.monotonic()-began, "interpretation": "Fixed source-only utility labels and observable controls; no learned gain or novelty claim"}
    write_new(args.out / "values_results.json", report)
    print(json.dumps({"complete": True, "rows": 3072, "results_sha256": digest(args.out / "values_results.json")}), flush=True)


if __name__ == "__main__":
    main()
