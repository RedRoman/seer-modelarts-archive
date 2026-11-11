"""CPU-only regression checks for sequence pairing and temporal decision signs."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "aggregate.py"
SPEC = importlib.util.spec_from_file_location("seer_aggregate", TOOL)
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def run(arm, rows, seed=0):
    return {"heldout": "sdsd", "arm": arm, "seed": seed, "per_sequence": rows}


class AggregationTests(unittest.TestCase):
    def test_temporal_seed_sign_measures_distance_from_gt(self):
        # A has lower raw tLP than B, but is farther from the target activity:
        # raw A-B = -0.1, absolute-gap A-B = +0.1.
        runs = [
            run("A", [{"name": "clip", "tlp": 0.1, "gt_tlp": 0.3}]),
            run("B", [{"name": "clip", "tlp": 0.2, "gt_tlp": 0.3}]),
        ]
        self.assertEqual(aggregate.per_seed_sign(runs, "sdsd", "A", "B", "tlp"), [-1.0])
        self.assertEqual(aggregate.per_seed_sign(runs, "sdsd", "A", "B", "d_tlp"), [1.0])

    def test_temporal_seed_sign_requires_gt_on_both_arms(self):
        runs = [
            run("A", [{"name": "clip", "tlp": 0.1, "gt_tlp": 0.3}]),
            run("B", [{"name": "clip", "tlp": 0.2}]),
        ]
        self.assertEqual(aggregate.per_seed_sign(runs, "sdsd", "A", "B", "d_tlp"), [])

    def test_missing_scores_are_filtered_jointly(self):
        sa = {"left": {"lpips": 100.0}, "right": {}, "shared": {"lpips": 3.0}}
        sb = {"left": {}, "right": {"lpips": -100.0}, "shared": {"lpips": 1.0}}
        names, va, vb = aggregate.paired_metric_values(sa, sb, "lpips")
        self.assertEqual(names, ["shared"])
        np.testing.assert_array_equal(va - vb, [2.0])
        self.assertEqual(aggregate.paired_metric_values(
            {"left": {"lpips": 100.0}, "right": {}},
            {"left": {}, "right": {"lpips": -100.0}}, "lpips")[0], [])

    def test_main_does_not_compare_different_missing_sequences(self):
        runs = [
            run("A", [{"name": "left", "psnr_y": 1.0, "lpips": 100.0},
                      {"name": "right", "psnr_y": 1.0}]),
            run("B", [{"name": "left", "psnr_y": 1.0},
                      {"name": "right", "psnr_y": 1.0, "lpips": -100.0}]),
        ]
        output = io.StringIO()
        with patch.object(aggregate, "load", return_value=runs), \
                patch.object(sys, "argv", ["aggregate.py", "--dir", "unused",
                                           "--compare", "A", "B", "--metrics", "lpips"]), \
                redirect_stdout(output):
            aggregate.main()
        self.assertNotIn("lpips", output.getvalue())

    def test_full_data_pooling_and_temporal_absolute_order_are_unchanged(self):
        runs = [
            run("A", [{"name": "one", "psnr_y": 1.0, "tlp": 0.1, "gt_tlp": 0.3}], 0),
            run("A", [{"name": "one", "psnr_y": 5.0, "tlp": 0.5, "gt_tlp": 0.3}], 1),
            run("B", [{"name": "one", "psnr_y": 1.0, "tlp": 0.2, "gt_tlp": 0.3}], 0),
            run("B", [{"name": "one", "psnr_y": 3.0, "tlp": 0.2, "gt_tlp": 0.3}], 1),
        ]
        averages, _ = aggregate.seed_average(runs)
        _, va, vb = aggregate.paired_metric_values(
            averages[("sdsd", "A")], averages[("sdsd", "B")], "psnr_y")
        np.testing.assert_array_equal(va - vb, [1.0])
        _, va, vb = aggregate.paired_metric_values(
            averages[("sdsd", "A")], averages[("sdsd", "B")], "d_tlp")
        # abs is still taken AFTER averaging across seeds in each arm.
        np.testing.assert_allclose(va - vb, [-0.1], atol=1e-15)


if __name__ == "__main__":
    unittest.main()
