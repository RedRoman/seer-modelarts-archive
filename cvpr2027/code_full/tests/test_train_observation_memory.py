"""Protocol/data alignment checks for the source memory screen."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "tools"))
spec = importlib.util.spec_from_file_location("_memory_trainer_test", CODE / "tools/train_observation_memory.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class SourcePlanTests(unittest.TestCase):
    def test_fit_only_plan_and_repeatability(self):
        names = [f"SDSD_indoor/pair{i}{suffix}" for i in range(1, 9) for suffix in ("", "_2")]
        split = trainer.make_scene_split(names)
        a = trainer.make_plan(names, split, 0)
        self.assertEqual(a, trainer.make_plan(names, split, 0))
        self.assertNotEqual(a, trainer.make_plan(names, split, 1))
        for batch in a["batches"]:
            self.assertIn(trainer.source_scene(names[batch["clip_index"]]), split["fit"])
            self.assertEqual(len(set(batch["frames"])), 8)
            self.assertTrue(all(0 <= t < 16 for t in batch["frames"]))
            self.assertTrue(all(batch[k] % 16 == 0 and 0 <= batch[k] <= 128 for k in ("top", "left")))

    def test_split_tampering_rejected(self):
        names = [f"SDSD_outdoor/pair{i}" for i in range(8)]
        split = trainer.make_scene_split(names)
        split["fit"].append(split["internal_holdout"][0])
        with self.assertRaises(ValueError):
            trainer.make_plan(names, split, 0)


class PatchCropTests(unittest.TestCase):
    def test_packed_fields_and_rgb_crop_have_identical_positions(self):
        import torch
        # Pixel coordinates and patch IDs catch transposed/offset row-major slices.
        grid = torch.arange(256*256).reshape(1,1,1,256,256).float().expand(1,3,16,256,256)
        packed = torch.arange(256).reshape(1,1,256,1).float().expand(1,16,256,64)
        sample = {"base":grid,"gt":grid+1,"prior_address":packed,"pixel_address":packed+1000,
                  "detail_values":packed[:,:,:,:1].expand(1,16,256,768)}
        inputs, gt = trainer.crop_inputs(sample,{"top":32,"left":64})
        expected = torch.arange(256).reshape(16,16)[2:10,4:12].reshape(-1).float()
        torch.testing.assert_close(inputs[2][0,0,:,0],expected)
        torch.testing.assert_close(inputs[3][0,0,:,0],expected+1000)
        torch.testing.assert_close(inputs[1][0,0,:,0],expected)
        torch.testing.assert_close(inputs[0],grid[...,32:160,64:192])
        torch.testing.assert_close(gt,grid[...,32:160,64:192]+1)

    def test_partial_patch_crop_rejected(self):
        with self.assertRaises(ValueError):
            trainer.crop_inputs({}, {"top":8,"left":0})


if __name__ == "__main__":
    unittest.main()
