"""Pure-NumPy unit tests for verify_eid_results.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_eid_results import recompute_scene_bootstrap, verify_target_report  # noqa: E402


class TestVerifyEIDResults(unittest.TestCase):
    def test_synthetic_report_verification(self) -> None:
        rows = []
        for idx, (seq, scene) in enumerate(
            [
                ("BVI/S01__10fps", "BVI/S01"),
                ("BVI/S01__20fps", "BVI/S01"),
                ("BVI/S02__10fps", "BVI/S02"),
                ("BVI/S02__20fps", "BVI/S02"),
            ]
        ):
            gt_tlp = 0.05 + 0.001 * idx
            anc = {
                "psnr_y": 24.0 + 0.1 * idx,
                "psnr_rgb": 23.0 + 0.1 * idx,
                "lpips": 0.15,
                "tlp": gt_tlp + 0.03,
                "gt_tlp": gt_tlp,
                "tlp_gap": 0.03,
            }
            seeds = {}
            for s in (0, 1, 2):
                tlp_s = gt_tlp + 0.01 + 0.001 * s
                seeds[f"eid_full_s{s}"] = {
                    "psnr_y": anc["psnr_y"] + 0.25 + 0.01 * s,
                    "psnr_rgb": anc["psnr_rgb"] + 0.22 + 0.01 * s,
                    "lpips": anc["lpips"] - 0.006,
                    "tlp": tlp_s,
                    "gt_tlp": gt_tlp,
                    "tlp_gap": abs(tlp_s - gt_tlp),
                }
            mean3 = {
                mk: sum(seeds[f"eid_full_s{s}"][mk] for s in (0, 1, 2)) / 3.0
                for mk in ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")
            }
            rows.append(
                {
                    "name": seq,
                    "scene": scene,
                    "lq_rgb_u8_sha256": "a" * 64,
                    "gt_rgb_u8_sha256": "b" * 64,
                    "methods": {"anchor": anc, **seeds, "eid_full_mean3": mean3},
                }
            )

        all_m = list(rows[0]["methods"].keys())
        means = {
            m: {k: sum(r["methods"][m][k] for r in rows) / len(rows) for k in ("psnr_y", "psnr_rgb", "lpips", "tlp", "gt_tlp", "tlp_gap")}
            for m in all_m
        }
        boot = {"eid_full_mean3": recompute_scene_bootstrap(rows, "eid_full_mean3", "anchor", ("psnr_y", "psnr_rgb", "lpips", "tlp_gap"))}
        payload = {
            "domain": "bvi",
            "split": "val",
            "backbone": "hvi_cidnet",
            "n_sequences": 4,
            "n_scenes": 2,
            "all_states_unchanged": True,
            "means": means,
            "scene_bootstrap_vs_anchor": boot,
            "per_sequence": rows,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "eval_synthetic.json"
            p.write_text(json.dumps(payload))
            res = verify_target_report(p, None)
            self.assertEqual(res["n_sequences"], 4)
            self.assertGreater(res["checks_passed"], 40)


if __name__ == "__main__":
    unittest.main()
