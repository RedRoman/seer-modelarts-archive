#!/usr/bin/env python3
"""Score the frozen source-FIT predictive-credit probe without Torch or GT access.

The analysis operates only on scalar JSON/JSONL records. Higher scores must mean
greater expected utility. Every primary statistic is first computed per scene;
the only resampling unit is a complete scene. See checks/analysis_protocol.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Iterable, Sequence


SCHEMA = "seer.predictive_credit_probe.analysis.v1"
SUBSET_SCHEMA = "seer.predictive_credit_probe.subsets.v1"
SEED = 20261123
BOOTSTRAP_REPEATS = 20000
UTILITY_EPSILON = 1e-10
MIN_SCENES = 8
ELIGIBILITY = 0.5
CANDIDATE = "trained_conditional"
BASELINES = ("random_conditional", "wrong_time_conditional", "ema_cosine", "negative_lq_mse", "negative_hvi_mse")
COALITIONS = ("empty", "donor1", "donor2", "both")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label}: expected numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: non-finite")
    return result


def midranks(values: Sequence[float]) -> list[float]:
    """One-based average ranks; exact equal floating point values form a tie."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for i in order[start:stop]:
            ranks[i] = rank
        start = stop
    return ranks


def auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC with midranks. Single-class observations are missing."""
    if len(labels) != len(scores):
        raise ValueError("AUROC input lengths differ")
    if any(not isinstance(v, bool) for v in labels):
        raise ValueError("AUROC requires boolean labels")
    checked = [finite(v, "AUROC score") for v in scores]
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if not n_positive or not n_negative:
        return None
    positive_rank_sum = sum(r for r, y in zip(midranks(checked), labels) if y)
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("correlation input lengths differ")
    if len(left) < 2:
        return None
    left = [finite(v, "correlation left") for v in left]
    right = [finite(v, "correlation right") for v in right]
    lm, rm = fmean(left), fmean(right)
    ld = [v - lm for v in left]
    rd = [v - rm for v in right]
    denom = math.sqrt(sum(v * v for v in ld) * sum(v * v for v in rd))
    if denom == 0.0:
        return None
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(ld, rd)) / denom))


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman input lengths differ")
    left = [finite(v, "Spearman left") for v in left]
    right = [finite(v, "Spearman right") for v in right]
    return pearson(midranks(left), midranks(right))


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation at (n-1)*q, matching the registered CI recipe."""
    if not sorted_values or not 0.0 <= q <= 1.0:
        raise ValueError("invalid percentile")
    x = (len(sorted_values) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return sorted_values[lo] + (x - lo) * (sorted_values[hi] - sorted_values[lo])


def bootstrap_mean(values: Sequence[float], *, repeats: int = BOOTSTRAP_REPEATS,
                   seed: int = SEED) -> dict:
    """Paired complete-scene bootstrap of precomputed scene differences."""
    values = [finite(v, "scene difference") for v in values]
    if not values or repeats < 1:
        raise ValueError("bootstrap requires values and positive repeats")
    rng = random.Random(seed)
    n = len(values)
    draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats)]
    draws.sort()
    return {"mean": fmean(values), "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
            "scenes": n, "repeats": repeats, "seed": seed,
            "resampling_unit": "complete_scene", "percentile_method": "linear_(n-1)*q"}


def credit(losses: dict) -> dict:
    if set(losses) != set(COALITIONS):
        raise ValueError(f"loss coalition fields must be {COALITIONS}")
    empty, one, two, both = [finite(losses[k], f"loss {k}") for k in COALITIONS]
    if any(v < -1e-6 or v > 4.0 + 1e-4 for v in (empty, one, two, both)):
        raise ValueError("normalized squared L2 losses must lie in [0,4] within numeric tolerance")
    v1, v2, v12 = empty - one, empty - two, empty - both
    c1, c2 = two - both, one - both
    phi1, phi2 = (v1 + c1) / 2.0, (v2 + c2) / 2.0
    budget = min(1.0, max(v12, 0.0))
    caps = [min(1.0, max(v, 0.0)) for v in (phi1, phi2)]
    largest = max(phi1, phi2)
    exp_phi = [math.exp(v - largest) for v in (phi1, phi2)]
    proposal = exp_phi[0] / sum(exp_phi)
    lower, upper = max(0.0, budget - caps[1]), min(caps[0], budget)
    if lower > upper + 1e-12:
        raise ValueError("Shapley caps cannot fulfill the fixed total budget")
    alpha0 = min(upper, max(lower, budget * proposal))
    alpha = [alpha0, budget - alpha0]
    return {"v": [v1, v2, v12], "conditional": [c1, c2], "phi": [phi1, phi2],
            "interaction": v12 - v1 - v2, "tau": 1.0,
            "alpha": alpha, "anchor_alpha": 1.0 - sum(alpha),
            "budget": budget, "caps": caps,
            "positive_cap_touched": [cap > 0.0 and abs(weight - cap) <= 1e-10
                                     for weight, cap in zip(alpha, caps)],
            "zero_budget": budget == 0.0, "v12_ge_one": v12 >= 1.0,
            "phi_ge_one": [phi1 >= 1.0, phi2 >= 1.0],
            "single_negative_phi": (phi1 < 0.0) != (phi2 < 0.0),
            "nonnegative_uncapped_degeneracy": phi1 >= 0.0 and phi2 >= 0.0 and v12 <= 1.0,
            "phi_positive": [phi1 > 0.0, phi2 > 0.0]}


def utility(mse: dict) -> dict:
    if set(mse) != set(COALITIONS):
        raise ValueError(f"MSE coalition fields must be {COALITIONS}")
    empty, one, two, both = [finite(mse[k], f"MSE {k}") for k in COALITIONS]
    if any(v < 0.0 or v > 1.0 + 1e-6 for v in (empty, one, two, both)):
        raise ValueError("unit-range RGB MSE outside [0,1]")
    v1, v2, v12 = empty - one, empty - two, empty - both
    return {"v": [v1, v2, v12], "conditional": [two - both, one - both],
            "interaction_action_recipe": v12 - v1 - v2}


def validate_subset(subset: dict, *, expected_scenes: int = 12) -> list[dict]:
    if subset.get("schema") != SUBSET_SCHEMA:
        raise ValueError("unrecognized source subset schema")
    if subset.get("target_inputs_read") is not False or subset.get("internal_inputs_read") is not False:
        raise ValueError("probe must declare no target/internal input access")
    split = subset["split"]
    fit, internal = set(split["fit"]), set(split["internal_holdout"])
    if fit & internal:
        raise ValueError("FIT/internal scenes overlap")
    entries = subset["entries"]
    if len(entries) != expected_scenes:
        raise ValueError("incorrect number of selected FIT scenes")
    if len({e["scene"] for e in entries}) != len(entries):
        raise ValueError("probe must use one clip per unique scene")
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("duplicate source clip")
    for entry in entries:
        if entry["scene"] not in fit or entry["scene"] in internal:
            raise ValueError("selected input is outside source FIT")
        if not entry["scene"].startswith(("SDSD_indoor/", "SDSD_outdoor/")):
            raise ValueError("selected input is not a recognized source scene")
        if entry["name"] != entry["scene"]:
            raise ValueError("only fixed first pass per scene is allowed")
    return entries


def primary_gate(per_scene: Sequence[dict], *, repeats: int = BOOTSTRAP_REPEATS) -> dict:
    """All five comparisons use the same complete-scene intersection."""
    if len({row["scene"] for row in per_scene}) != len(per_scene):
        raise ValueError("duplicate scene statistics")
    common = [row for row in per_scene if all(row["auc"][name] is not None
                                             for name in (CANDIDATE,) + BASELINES)]
    candidate_values = [row["auc"][CANDIDATE] for row in common]
    candidate_macro = fmean(candidate_values) if candidate_values else None
    comparisons = {}
    for baseline in BASELINES:
        matched = common
        differences = [row["auc"][CANDIDATE] - row["auc"][baseline] for row in matched]
        stats = bootstrap_mean(differences, repeats=repeats) if differences else None
        evaluable = len(matched) >= MIN_SCENES
        comparisons[baseline] = {"matched_scene_names": [row["scene"] for row in matched],
                                 "bootstrap": stats, "evaluable": evaluable,
                                 "passes": bool(evaluable and stats["ci95"][0] > 0.0)}
    evaluable = len(candidate_values) >= MIN_SCENES and all(v["evaluable"] for v in comparisons.values())
    passes = bool(evaluable and candidate_macro >= 0.55 and all(v["passes"] for v in comparisons.values()))
    return {"status": "pass" if passes else ("fail" if evaluable else "not_evaluable"),
            "candidate_macro_auc": candidate_macro, "candidate_defined_scenes": len(candidate_values),
            "common_complete_scene_names": [row["scene"] for row in common],
            "defined_scene_counts_before_common_intersection": {
                name: sum(row["auc"][name] is not None for row in per_scene)
                for name in (CANDIDATE,) + BASELINES},
            "minimum_defined_scenes": MIN_SCENES, "candidate_auc_threshold": 0.55,
            "comparisons": comparisons, "passes": passes,
            "interpretation": "Source FIT screening only; no publication, SOTA, or generalization evidence."}


def scene_statistics(scene: str, patches: Sequence[dict]) -> dict:
    """Input is the strict normalized scalar schema produced by join_records."""
    eligible = [p for p in patches if min(p["valid_fraction"]) >= ELIGIBILITY]
    labels, utilities = [], []
    score_vectors = {name: [] for name in (CANDIDATE,) + BASELINES}
    ties = 0
    positive = negative = 0
    for patch in eligible:
        for donor in (0, 1):
            value = patch["utility"]["conditional"][donor]
            if abs(value) <= UTILITY_EPSILON:
                ties += 1
                continue
            labels.append(value > 0.0)
            positive += int(value > 0.0)
            negative += int(value < 0.0)
            utilities.append(value)
            score_vectors[CANDIDATE].append(patch["trained"]["conditional"][donor])
            score_vectors["random_conditional"].append(patch["random"]["conditional"][donor])
            score_vectors["wrong_time_conditional"].append(patch["wrong_time"]["conditional"][donor])
            for name in BASELINES[2:]:
                score_vectors[name].append(patch["controls"][name][donor])
    # Affine standardization of utility within each scene leaves rank correlation
    # unchanged, but is explicit here so no cross-scene scale can enter a summary.
    if utilities:
        mean = fmean(utilities)
        sigma = math.sqrt(fmean([(v - mean) ** 2 for v in utilities]))
        standardized = [(v - mean) / sigma for v in utilities] if sigma else [0.0] * len(utilities)
    else:
        standardized = []
    auc = {name: auroc(labels, scores) for name, scores in score_vectors.items()}
    rank_correlations = {name: spearman(standardized, scores) for name, scores in score_vectors.items()}
    interactions = {
        "predictive_vs_rgb_action_recipe_spearman": spearman(
            [p["trained"]["interaction"] for p in eligible],
            [p["utility"]["interaction_action_recipe"] for p in eligible]),
        "definition_note": "RGB contrast is specific to singleton versus averaged-pair actions; it is not physical interaction ground truth.",
    }
    denominator = len(eligible)
    budget = {
        "eligible_patches": denominator,
        "zero_budget_fraction": sum(p["trained"]["zero_budget"] for p in eligible) / denominator if denominator else None,
        "v12_ge_one_fraction_tau1": sum(p["trained"]["v12_ge_one"] for p in eligible) / denominator if denominator else None,
        "phi_ge_one_fraction_by_donor_tau1": [sum(p["trained"]["phi_ge_one"][d] for p in eligible) / denominator
                                               if denominator else None for d in (0, 1)],
        "allocation_touches_positive_cap_fraction_by_donor_tau1": [
            sum(p["trained"]["positive_cap_touched"][d] for p in eligible) / denominator
            if denominator else None for d in (0, 1)],
        "allocation_touches_any_positive_cap_fraction_tau1": sum(
            any(p["trained"]["positive_cap_touched"]) for p in eligible) / denominator if denominator else None,
        "single_negative_phi_fraction": sum(p["trained"]["single_negative_phi"] for p in eligible) / denominator if denominator else None,
        "nonnegative_uncapped_shapley_degeneracy_fraction": sum(p["trained"]["nonnegative_uncapped_degeneracy"] for p in eligible) / denominator if denominator else None,
        "positive_phi_fraction_by_donor": [sum(p["trained"]["phi_positive"][d] for p in eligible) / denominator
                                            if denominator else None for d in (0, 1)],
        "mean_anchor_weight_tau1": fmean(p["trained"]["anchor_alpha"] for p in eligible) if denominator else None,
        "v12_quantiles_eligible": {str(q): percentile(sorted(p["trained"]["v"][2] for p in eligible), q)
                                   if denominator else None for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
        "recipe": "B=min(1,max(v12,0)); caps_i=min(1,max(phi_i,0)); r=softmax(phi); w0=clip(B*r0,max(0,B-caps1),min(caps0,B)); w1=B-w0; anchor=1-B; tau fixed to 1",
    }
    return {"scene": scene, "patches": len(patches), "eligible_patches": denominator,
            "ineligible_patches": len(patches) - denominator, "donor_labels": len(labels),
            "positive_donor_labels": positive, "negative_donor_labels": negative,
            "excluded_zero_utility_donors": ties, "auc": auc,
            "spearman_scene_standardized_utility": rank_correlations,
            "interaction_diagnostic": interactions, "budget_diagnostic": budget}


def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError("scalar input must be JSONL, a JSON rows list, or an object with rows")


def load_critic_input(path: Path, subset: dict, *, observable_sha256: str | None = None) -> tuple[list[dict], dict]:
    """Read the frozen exporter's hashed, GT-free per-clip scalar JSON format."""
    if path.suffix == ".jsonl":
        return _load_rows(path), {"format": "canonical_jsonl", "sha256": sha256(path)}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "clips" not in payload:
        return _load_rows(path), {"format": "canonical_json", "sha256": sha256(path)}
    entries = validate_subset(subset)
    allowed = {(e["scene"], e["name"]): e for e in entries}
    if payload.get("schema_version") != 1 or payload.get("count") != len(entries) or len(payload["clips"]) != len(entries):
        raise ValueError("critic manifest schema/count mismatch")
    if payload.get("gt_storage_entries_read") != 0 or payload.get("quality_scores_computed") is not False:
        raise ValueError("critic manifest must declare zero GT storage access and no quality scoring")
    rows, receipts, seen = [], [], set()
    grid_y, grid_x = subset["spatial_grid"]
    patches = grid_y * grid_x
    expected_yx = [[y, x] for y in range(grid_y) for x in range(grid_x)]
    for item in payload["clips"]:
        identity = item.get("scene"), item.get("name")
        if identity not in allowed or identity in seen:
            raise ValueError("critic clip manifest is outside frozen FIT subset or duplicated")
        seen.add(identity)
        if item.get("rgb_lq_tensor_sha256") != allowed[identity]["rgb_lq_tensor_sha256"]:
            raise ValueError("critic input tensor hash differs from frozen RGB LQ")
        clip_path = (path.parent / item["path"]).resolve()
        if not clip_path.is_relative_to(path.parent.resolve()) or clip_path.suffix != ".json":
            raise ValueError("critic scalar path escapes its manifest directory")
        if sha256(clip_path) != item["sha256"]:
            raise ValueError("critic scalar file hash mismatch")
        clip = json.loads(clip_path.read_text())
        if (clip.get("scene"), clip.get("name")) != identity:
            raise ValueError("critic scalar identity mismatch")
        if clip.get("source_archive_sha256") != allowed[identity]["source_sha256"] or clip.get("rgb_lq_tensor_sha256") != allowed[identity]["rgb_lq_tensor_sha256"]:
            raise ValueError("critic scalar source/archive tensor digest differs from subset")
        if observable_sha256 is not None and clip.get("observable_sidecar_sha256") != observable_sha256:
            raise ValueError("critic flow-alignment sidecar differs from pinned GT-free observables")
        if clip.get("coalition_order") != list(COALITIONS):
            raise ValueError("critic coalition order mismatch")
        if clip.get("patch_yx") != expected_yx:
            raise ValueError("critic patch coordinates are not the complete row-major grid")
        if clip.get("target_frames") != subset["target_frames"] or clip.get("donor_frames") != subset["donor_frames"]:
            raise ValueError("critic target/donor time specification differs from frozen subset")
        cosine = clip["static_full_ema_flow_aligned_cosine"]
        if len(cosine) != 2 or any(len(vector) != patches for vector in cosine):
            raise ValueError("static EMA cosine must be 2 by patch count")
        for mode in ("trained", "random", "wrong_time"):
            losses = clip["losses"][mode]
            if len(losses) != 4 or any(len(vector) != patches for vector in losses):
                raise ValueError("critic losses must be 4 coalitions by patch count")
        for patch in range(patches):
            row = {"scene": identity[0], "name": identity[1], "patch": patch, "split_role": "source_fit",
                   "controls": {"ema_cosine": [cosine[d][patch] for d in (0, 1)]}}
            for mode in ("trained", "random", "wrong_time"):
                row[mode] = {coalition: clip["losses"][mode][i][patch]
                             for i, coalition in enumerate(COALITIONS)}
            rows.append(row)
        receipts.append({"path": str(clip_path), "sha256": item["sha256"], "name": identity[1]})
    return rows, {"format": "hashed_per_clip_scalar_json", "sha256": sha256(path), "clips": receipts}


def load_observable_sidecar(path: Path, expected_sha256: str, subset: dict, subset_sha256: str) -> dict:
    if sha256(path) != expected_sha256:
        raise ValueError("GT-free observable sidecar digest mismatch")
    payload = json.loads(path.read_text())
    if payload.get("schema") != "seer.predictive_credit_probe.observables.v1" or payload.get("subset_sha256") != subset_sha256:
        raise ValueError("observable sidecar schema/subset mismatch")
    for key in ("gt_storage_entries_read", "target_inputs_read", "internal_holdout_inputs_read"):
        if payload.get(key) != 0:
            raise ValueError("observable sidecar must have zero GT/target/internal access")
    allowed = {(e["scene"], e["name"]) for e in validate_subset(subset)}
    clips = {(clip.get("scene"), clip.get("name")): clip for clip in payload["clips"]}
    if len(payload["clips"]) != len(allowed) or set(clips) != allowed:
        raise ValueError("observable sidecar must cover the exact frozen FIT subset")
    return clips


def load_value_input(path: Path, subset: dict, *, observables: dict | None = None) -> tuple[list[dict], dict]:
    """Read scalar NPZ with allow_pickle=False and independently check utilities."""
    if path.suffix != ".npz":
        return _load_rows(path), {"format": "canonical_json", "sha256": sha256(path)}
    import numpy as np

    entries = validate_subset(subset)
    columns = ("clip_index", "patch_y", "patch_x", "mse_base", "mse_single1", "mse_single2",
               "mse_pair", "mse_sameframe_zero", "mse_temporal_no_dc", "u1", "u2", "u12",
               "u1_given2", "u2_given1", "feature_lq1", "feature_lq2", "feature_hvi1",
               "feature_hvi2", "feature_valid1", "feature_valid2")
    with np.load(path, allow_pickle=False) as archive:
        if not set(columns) <= set(archive.files):
            raise ValueError("utility NPZ missing required scalar columns")
        arrays = {name: archive[name].copy() for name in columns}
    expected = len(entries) * math.prod(subset["spatial_grid"])
    for name, values in arrays.items():
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(f"utility NPZ {name} has wrong shape or nonfinite values")
        if name in columns[:3]:
            if values.dtype != np.dtype("int64"):
                raise ValueError("utility NPZ coordinates must be int64")
        elif values.dtype != np.dtype("float64"):
            raise ValueError("utility NPZ scalar measurements must be float64")
    formulas = {"u1": arrays["mse_base"] - arrays["mse_single1"],
                "u2": arrays["mse_base"] - arrays["mse_single2"],
                "u12": arrays["mse_base"] - arrays["mse_pair"],
                "u1_given2": arrays["mse_single2"] - arrays["mse_pair"],
                "u2_given1": arrays["mse_single1"] - arrays["mse_pair"]}
    errors = {name: float(np.max(np.abs(arrays[name] - expected_values)))
              for name, expected_values in formulas.items()}
    if max(errors.values()) > 1e-12:
        raise ValueError("saved utility differs from independent MSE subtraction")
    if not np.array_equal(arrays["mse_base"], arrays["mse_sameframe_zero"]):
        raise ValueError("sameframe zero-action MSE must exactly equal base MSE")
    rows = []
    grid_y, grid_x = subset["spatial_grid"]
    for i in range(expected):
        clip, y, x = [int(arrays[name][i]) for name in columns[:3]]
        if not 0 <= clip < len(entries) or not 0 <= y < grid_y or not 0 <= x < grid_x:
            raise ValueError("utility NPZ scene or patch index outside frozen subset")
        entry = entries[clip]
        if observables is not None:
            prior = observables[(entry["scene"], entry["name"])]
            patch = y * grid_x + x
            for d in (1, 2):
                for column_prefix, sidecar_key in (("feature_lq", "feature_lq"), ("feature_hvi", "feature_hvi"),
                                                   ("feature_valid", "valid_fraction")):
                    if float(arrays[f"{column_prefix}{d}"][i]) != prior[sidecar_key][d - 1][patch]:
                        raise ValueError("post-GT utility feature differs from pinned GT-free observable scalar")
        rows.append({"scene": entry["scene"], "name": entry["name"], "patch": y * grid_x + x,
                     "split_role": "source_fit",
                     "valid_fraction": [float(arrays[f"feature_valid{d}"][i]) for d in (1, 2)],
                     "mse": {"empty": float(arrays["mse_base"][i]), "donor1": float(arrays["mse_single1"][i]),
                             "donor2": float(arrays["mse_single2"][i]), "both": float(arrays["mse_pair"][i])},
                     "controls": {"negative_lq_mse": [float(arrays[f"feature_lq{d}"][i]) for d in (1, 2)],
                                  "negative_hvi_mse": [float(arrays[f"feature_hvi{d}"][i]) for d in (1, 2)]}})
    return rows, {"format": "scalar_npz_allow_pickle_false", "sha256": sha256(path),
                  "independently_checked_utility_subtractions": expected * len(formulas),
                  "utility_max_absolute_errors": errors, "sameframe_zero_exact_equal": True,
                  "gt_free_observable_features_exact_checked": expected * 6 if observables is not None else 0,
                  "no_tensor_or_gt_pixel_access": True}


def join_records(subset: dict, critic_rows: list[dict], value_rows: list[dict]) -> dict[str, list[dict]]:
    """Validate identities before considering any scalar measurement."""
    entries = validate_subset(subset)
    allowed = {(e["scene"], e["name"]): e for e in entries}
    expected_patches = math.prod(subset["spatial_grid"])
    expected_keys = {(scene, name, patch) for scene, name in allowed for patch in range(expected_patches)}

    def indexed(rows: list[dict], label: str) -> dict:
        result = {}
        for row in rows:
            if not isinstance(row.get("patch"), int) or isinstance(row.get("patch"), bool):
                raise ValueError(f"{label}: patch must be an integer")
            key = row.get("scene"), row.get("name"), row["patch"]
            if key not in expected_keys:
                raise ValueError(f"{label}: unexpected scene/name/patch (outside frozen FIT subset)")
            if key in result:
                raise ValueError(f"{label}: duplicate patch")
            if row.get("split_role") != "source_fit":
                raise ValueError(f"{label}: only explicit source_fit scalar records are allowed")
            result[key] = row
        if set(result) != expected_keys:
            raise ValueError(f"{label}: incomplete frozen subset patch coverage")
        return result

    critics, values = indexed(critic_rows, "critic"), indexed(value_rows, "value")
    grouped = {entry["scene"]: [] for entry in entries}
    for key in sorted(expected_keys):
        cr, vr = critics[key], values[key]
        fractions = [finite(v, "valid fraction") for v in vr["valid_fraction"]]
        if len(fractions) != 2 or any(v < 0.0 or v > 1.0 for v in fractions):
            raise ValueError("valid_fraction requires two fractions in [0,1]")
        controls = {}
        for name in BASELINES[2:]:
            control_source = cr["controls"] if name in cr["controls"] else vr.get("controls", {})
            vector = [finite(v, f"{name} control") for v in control_source[name]]
            if len(vector) != 2:
                raise ValueError("observable controls need one score per donor")
            controls[name] = vector
        grouped[key[0]].append({"trained": credit(cr["trained"]), "random": credit(cr["random"]),
                                "wrong_time": credit(cr["wrong_time"]),
                                "controls": controls, "valid_fraction": fractions,
                                "utility": utility(vr["mse"]), "patch": key[2]})
    return grouped


def analyze(subset: dict, critic_rows: list[dict], value_rows: list[dict], *,
            repeats: int = BOOTSTRAP_REPEATS) -> dict:
    grouped = join_records(subset, critic_rows, value_rows)
    scenes = [scene_statistics(scene, grouped[scene]) for scene in sorted(grouped)]
    macro_auc = {}
    for name in (CANDIDATE,) + BASELINES:
        values = [row["auc"][name] for row in scenes if row["auc"][name] is not None]
        macro_auc[name] = {"mean": fmean(values) if values else None, "defined_scenes": len(values)}
    return {"schema": SCHEMA, "scope": "fixed_12_source_fit_scenes_no_training_no_tau_or_lambda_selection",
            "selected_scenes": [row["scene"] for row in scenes], "patches": sum(row["patches"] for row in scenes),
            "eligibility": "valid_fraction_donor1>=0.5 AND valid_fraction_donor2>=0.5",
            "utility_tie_epsilon": UTILITY_EPSILON, "primary_macro_auc": macro_auc,
            "primary_gate": primary_gate(scenes, repeats=repeats), "per_scene": scenes,
            "no_pooled_patch_auc": True, "no_patch_bootstrap": True,
            "limitations": ["Only a mechanism screen on source FIT; no heldout or target inference.",
                            "Gate pass permits a new source-only module pilot; it establishes no module innovation or performance claim.",
                            "Budget and action-recipe interaction are diagnostics, not model selection or physical interaction truth."]}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def freeze_sources(subset: Path, protocol: Path, sources: Iterable[Path], output: Path) -> dict:
    validate_subset(json.loads(subset.read_text()))
    paths = [Path(__file__).resolve(), Path(__file__).with_name("test_analyze_probe.py").resolve(),
             Path(__file__).parent.joinpath("checks/analysis_protocol.md").resolve(),
             subset.resolve(), protocol.resolve()] + [p.resolve() for p in sources]
    unique = list(dict.fromkeys(paths))
    root = output.parent.parent.resolve()
    manifest = {"schema": "seer.predictive_credit_probe.analysis_freeze.v1", "before_actual_scores_read": True,
                "files": [{"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                           "sha256": sha256(path)} for path in unique]}
    write_new(output, manifest)
    return manifest


def verify_freeze(path: Path, subset: Path, protocol: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "seer.predictive_credit_probe.analysis_freeze.v1" or manifest.get("before_actual_scores_read") is not True:
        raise ValueError("unrecognized or post-score analysis freeze")
    root = path.parent.parent.resolve()
    required = {Path(__file__).resolve(), Path(__file__).with_name("test_analyze_probe.py").resolve(),
                Path(__file__).parent.joinpath("checks/analysis_protocol.md").resolve(),
                subset.resolve(), protocol.resolve()}
    bound = set()
    for item in manifest["files"]:
        candidate = Path(item["path"])
        candidate = candidate if candidate.is_absolute() else root / candidate
        candidate = candidate.resolve()
        if sha256(candidate) != item["sha256"]:
            raise ValueError(f"frozen source changed: {candidate}")
        bound.add(candidate)
    if not required <= bound:
        raise ValueError("analysis freeze omits required code, test, subset, or protocol")
    return {"path": str(path), "sha256": sha256(path), "verified_bound_files": len(bound)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("freeze", "analyze"):
        p = subparsers.add_parser(mode)
        p.add_argument("--subset", type=Path, required=True)
        p.add_argument("--protocol", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if mode == "freeze":
            p.add_argument("--bind-source", type=Path, action="append", default=[])
        else:
            p.add_argument("--frozen-analysis-manifest", type=Path, required=True)
            p.add_argument("--critic", type=Path, required=True)
            p.add_argument("--values", type=Path, required=True)
            p.add_argument("--observable-sidecar", type=Path, required=True)
            p.add_argument("--expected-observable-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze_sources(args.subset, args.protocol, args.bind_source, args.output)
        print(json.dumps({"freeze_sha256": sha256(args.output), "bound_files": len(result["files"])}))
        return
    # All bound source bytes are checked before either scalar result file opens.
    receipt = verify_freeze(args.frozen_analysis_manifest, args.subset, args.protocol)
    subset = json.loads(args.subset.read_text())
    observables = load_observable_sidecar(args.observable_sidecar, args.expected_observable_sha256,
                                        subset, sha256(args.subset))
    critic_rows, critic_receipt = load_critic_input(args.critic, subset,
                                                  observable_sha256=args.expected_observable_sha256)
    value_rows, value_receipt = load_value_input(args.values, subset, observables=observables)
    result = analyze(subset, critic_rows, value_rows)
    result["provenance"] = {"analysis_freeze": receipt, "critic_sha256": sha256(args.critic),
                            "values_sha256": sha256(args.values), "subset_sha256": sha256(args.subset),
                            "protocol_sha256": sha256(args.protocol), "critic_input": critic_receipt,
                            "value_input": value_receipt, "observable_sidecar_sha256": args.expected_observable_sha256}
    write_new(args.output, result)
    print(json.dumps({"output_sha256": sha256(args.output), "gate": result["primary_gate"]["status"]}))


if __name__ == "__main__":
    main()
