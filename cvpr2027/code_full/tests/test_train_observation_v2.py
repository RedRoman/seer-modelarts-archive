"""Sampling isolation and crop-coordinate guards for the source-only runner."""
import importlib.util
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "tools"))
spec = importlib.util.spec_from_file_location("_train_observation_v2_test", CODE / "tools/train_observation_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
try:
    import torch
except ImportError:
    torch = None


class SourcePlanTests(unittest.TestCase):
    def test_fit_only_clip_plan_and_valid_frame_crop_budget(self):
        names = [f"SDSD_indoor/pair{s}{suffix}" for s in range(6) for suffix in ("", "_2")]
        split = {"fit": [f"SDSD_indoor/pair{s}" for s in range(4)]}
        plan = runner.make_plan(names, split, 0)
        self.assertEqual(plan, runner.make_plan(names, split, 0))
        self.assertNotEqual(plan, runner.make_plan(names, split, 1))
        self.assertEqual(len(plan["batches"]), 1000)
        for row in plan["batches"]:
            self.assertLess(row["clip_index"], 8)
            self.assertEqual(len(set(row["frames"])), 8)
            self.assertTrue(all(0 <= t < 16 for t in row["frames"]))
            self.assertTrue(all(0 <= row[k] <= 128 and row[k] % 8 == 0 for k in ("top", "left")))

    def test_scene_prefixes_prevent_accidental_source_mix(self):
        names = ["SDSD_indoor/pair1", "SDSD_indoor/pair1_2", "SDSD_outdoor/pair1"]
        plan = runner.make_plan(names, {"fit": ["SDSD_indoor/pair1"]}, 2)
        self.assertEqual({row["clip_index"] for row in plan["batches"]}, {0, 1})


@unittest.skipIf(torch is None, "Torch required; execute actual tensor guards remotely")
class SourceCropTests(unittest.TestCase):
    def test_crop_masks_reject_donors_outside_available_state(self):
        torch.set_num_threads(1)
        def zeros(shape):
            return torch.zeros(1).expand(shape)
        flow = torch.zeros(1, 2, 2, 16, 256, 256)
        flow[:, :, 0] = 8
        mask = torch.ones(1, 2, 1, 16, 256, 256, dtype=torch.bool)
        mask[:, 0, :, 0] = False
        mask[:, 1, :, -1] = False
        sample = {"base": zeros((1, 3, 16, 256, 256)), "gt": zeros((1, 3, 16, 256, 256)),
                  "detail_observations": zeros((1, 3, 16, 256, 256)),
                  "adjacent_flow": flow, "adjacent_mask": mask,
                  "corrections": {"raw_detail": zeros((1, 2, 3, 16, 256, 256))},
                  "features": {"raw_detail": zeros((1, 16, 19, 32, 32))}}
        norm = {"feature_mean": [0.] * 19, "feature_std": [1.] * 19}
        inputs, gt = runner.crop_sample(sample, "raw_detail", {"top": 64, "left": 64}, norm)
        self.assertEqual(tuple(gt.shape), (1, 3, 16, 128, 128))
        self.assertFalse(bool(inputs[-1][..., -8:].any()))
        self.assertTrue(bool(inputs[-1][:, 0, :, 1:, :, :-8].all()))
        self.assertFalse(bool(inputs[-1][:, 0, :, 0].any()))
        self.assertTrue(bool(mask[:, 0, :, 1:, :, :-8].all()), "Cropping must not mutate source cache")
        sample["gt"] = torch.ones(1).expand(1, 3, 16, 256, 256)
        changed, labels = runner.crop_sample(sample, "raw_detail", {"top": 64, "left": 64}, norm)
        for a, b in zip(inputs, changed):
            self.assertTrue(torch.equal(a, b), "GT cannot enter observable model inputs")
        self.assertTrue(bool(labels.eq(1).all()))


if __name__ == "__main__":
    unittest.main()
