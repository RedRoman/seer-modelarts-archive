"""Audit H17 temporal-gap macros against untouched archived per-sequence rows.

Run from any directory: python paper/tools/check_temporal_metric.py
Requires NumPy; performs no model inference and changes no files.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import statistics

import numpy as np

CVPR = Path(__file__).resolve().parents[2]
ARCHIVE = CVPR / "results" / "cvpr_converged"
MIRROR = CVPR / "paper" / "results" / "cvpr"
MACROS = CVPR / "paper" / "tables" / "result_macros_cvpr.tex"
NBOOT, RNG = 10000, 0


def load(relative):
    archived = json.loads((ARCHIVE / relative).read_text())
    assert archived == json.loads((MIRROR / relative).read_text()), relative
    return {row["name"]: row for row in archived["per_sequence"]}


def gap(row):
    return row["tlp"] - row["gt_tlp"]


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    indices = np.random.default_rng(RNG).integers(
        0, len(values), size=(NBOOT, len(values)))
    return np.percentile(values[indices].mean(1), [2.5, 97.5])


def macro(key):
    pattern = r"\\csname res@" + re.escape(key) + r"\\endcsname\{([^}]+)\}"
    matches = re.findall(pattern, MACROS.read_text())
    assert len(matches) == 1, (key, matches)
    return matches[0]


def main():
    baselines = [load(f"indomain/e17_small_pretrained_s{s}.json") for s in (0, 1)]
    names = sorted(baselines[0])
    assert len(names) == 39 and set(names) == set(baselines[1])
    report = {"bootstrap_samples": NBOOT, "numpy_rng_seed": RNG, "n": len(names)}
    for tag, key in (("e22_tdelta", "tdrel"), ("e24_tdcos", "tdcos")):
        variants = [load(f"h17/{tag}_s{s}.json") for s in (0, 1)]
        assert all(set(rows) == set(names) for rows in variants)
        # Independent stdlib calculation: average signed gaps over each arm's
        # seeds, then take absolute value per sequence, matching aggregate.py.
        old, new = [], []
        for name in names:
            variant_gap = statistics.mean(gap(rows[name]) for rows in variants)
            baseline_gap = statistics.mean(gap(rows[name]) for rows in baselines)
            old.append(variant_gap - baseline_gap)
            new.append(abs(variant_gap) - abs(baseline_gap))
        per_seed = [
            statistics.mean(abs(gap(v[name])) - abs(gap(b[name])) for name in names)
            for v, b in zip(variants, baselines)
        ]
        ci = bootstrap(new)
        assert all(effect > 0 for effect in per_seed), (tag, per_seed)
        assert ci[0] > 0, (tag, ci)
        mean = statistics.mean(new)
        expected_ci = "[%+.4f, %+.4f]" % tuple(ci)
        assert macro(f"h17@{key}@tlp") == "%+.4f" % mean
        assert macro(f"h17@{key}@tlp@ci") == expected_ci
        report[tag] = {
            "old_signed_difference": statistics.mean(old),
            "old_signed_ci": bootstrap(old).tolist(),
            "new_absolute_gap_difference": mean,
            "new_absolute_gap_ci": ci.tolist(),
            "per_seed_absolute_gap_differences": per_seed,
            "both_seeds_worse": True,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
