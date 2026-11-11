"""Small paired PNG fixtures; no dataset/model weights or test frames are read."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is required for BVI CPU checks")
class PlannedBVITests(unittest.TestCase):
    def setUp(self):
        from data import planned_bvi
        self.module = planned_bvi
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "datasets/BVI-Lowlight-videos"
        self.split = self.root / "splits.json"
        self.entry = {"root": "/temp/zihaohe/seer_cvpr/datasets/BVI-Lowlight-videos",
                      "train": ["train_scene__10fps", "train_scene__20fps"],
                      "val": ["val_scene__10fps", "val_scene__20fps"],
                      "test": ["test_scene__10fps", "test_scene__20fps"]}
        self.counts = dict(train=2, val=2, test=2)
        self.write_split()
        # Deliberately create no VAL/test directories. Training construction
        # and sampling must succeed without enumerating/opening either split.
        for condition in (10, 20):
            for kind, offset in (("low", 0), ("normal", 20)):
                directory = self.base / "train_scene" / f"{kind}_light_{condition}"
                directory.mkdir(parents=True)
                for frame in (2, 1):  # filesystem creation order is reversed
                    array = (np.arange(6 * 7 * 3).reshape(6, 7, 3) + frame * 10 + offset).astype(np.uint8)
                    Image.fromarray(array).save(directory / f"{frame:04d}.png")

    def write_split(self):
        self.split.write_text(json.dumps({"BVI_RLV": self.entry}))

    def dataset(self, **kwargs):
        params = dict(samples=8, seed=123, crop=4, clip_len=4, expected_counts=self.counts)
        params.update(kwargs)
        return self.module.PlannedBVI(self.split, self.root, **params)

    def test_plan_and_paired_augmentation_are_order_and_global_rng_independent(self):
        first, second = self.dataset(), self.dataset()
        self.assertEqual(first.plan, second.plan)
        self.assertNotEqual(first.plan, self.dataset(seed=124).plan)
        examples = [first[i] for i in range(len(first))]
        np.random.seed(987)
        torch.manual_seed(987)
        for index in reversed(range(len(second))):
            actual, expected = second[index], examples[index]
            self.assertEqual(actual["name"], expected["name"])
            self.assertEqual(actual["sample_index"], index)
            self.assertTrue(torch.equal(actual["lq"], expected["lq"]))
            self.assertTrue(torch.equal(actual["gt"], expected["gt"]))
            torch.testing.assert_close(actual["gt"] - actual["lq"],
                                       torch.full_like(actual["lq"], 20 / 255))
        self.assertFalse((self.base / "val_scene").exists())
        self.assertFalse((self.base / "test_scene").exists())

    def test_sorted_index_pairing_short_clip_padding_and_tensor_contract(self):
        dataset = self.dataset(crop=8)
        self.assertEqual([Path(p).name for p in dataset.items[0][1]], ["0001.png", "0002.png"])
        sample = dataset[0]
        self.assertEqual(tuple(sample["lq"].shape), (3, 4, 8, 8))
        self.assertEqual(sample["lq"].dtype, torch.float32)
        self.assertTrue(torch.equal(sample["lq"][:, 1], sample["lq"][:, 2]))
        self.assertTrue(torch.equal(sample["gt"][:, 1], sample["gt"][:, 3]))
        torch.testing.assert_close(sample["lq"][:, 1] - sample["lq"][:, 0],
                                   torch.full_like(sample["lq"][:, 0], 10 / 255))
        self.assertGreaterEqual(float(sample["lq"].min()), 0.)
        self.assertLessEqual(float(sample["gt"].max()), 1.)

    def test_counts_scene_leakage_alias_duplicates_and_bad_identifiers_reject_before_io(self):
        with self.assertRaisesRegex(ValueError, "expected 48"):
            self.dataset(expected_counts=None)
        for change, message in (
            ({"val": list(self.entry["train"])}, "scene leakage"),
            ({"train": ["train_scene__10fps", "train_scene__fps10"]}, "duplicate scene/condition"),
            ({"val": ["val_scene__10fps", "another_scene__20fps"]}, "both conditions"),
            ({"test": ["../unsafe__10fps", "../unsafe__20fps"]}, "Invalid BVI"),
            ({"train_scenes": ["wrong_scene"]}, "declared scene list"),
        ):
            with self.subTest(message=message):
                entry = dict(self.entry)
                entry.update(change)
                self.split.write_text(json.dumps({"BVI_RLV": entry}))
                with mock.patch.object(self.module, "_png_frames", side_effect=AssertionError("Unexpected frame IO")):
                    with self.assertRaisesRegex(ValueError, message):
                        self.dataset()
        self.write_split()

    def test_empty_and_unequal_png_frame_counts_are_not_silently_truncated(self):
        directory = self.base / "train_scene/normal_light_10"
        (directory / "0002.png").unlink()
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            self.dataset()
        (directory / "0001.png").unlink()
        with self.assertRaisesRegex(ValueError, "Empty BVI"):
            self.dataset()

    def test_equal_counts_with_shifted_frame_names_are_rejected(self):
        directory = self.base / "train_scene/normal_light_10"
        (directory / "0001.png").rename(directory / "0003.png")
        with self.assertRaisesRegex(ValueError, "PNG basename mismatch"):
            self.dataset()

    def test_pair_and_within_clip_shapes_are_validated_before_augmentation(self):
        dataset = self.dataset()
        index = next(i for i, p in enumerate(dataset.plan) if p["sequence_index"] == 0)
        gt_path = self.base / "train_scene/normal_light_10/0002.png"
        Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(gt_path)
        with self.assertRaisesRegex(ValueError, "paired frame shape mismatch"):
            dataset[index]
        lq_path = self.base / "train_scene/low_light_10/0002.png"
        Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(lq_path)
        with self.assertRaisesRegex(ValueError, "shape changes within sampled clip"):
            dataset[index]

    def test_finite_range_and_rgb_shape_guards(self):
        dataset = self.dataset()
        for invalid in (np.full((6, 7, 3), np.nan), np.full((6, 7, 3), -.1),
                        np.full((6, 7, 3), 1.1), np.zeros((6, 7, 2))):
            with self.subTest(shape=invalid.shape, value=float(invalid.flat[0])):
                with mock.patch.object(self.module, "read_image_01", return_value=invalid):
                    with self.assertRaisesRegex(ValueError, "finite and in|Invalid RGB"):
                        dataset[0]

    def test_legacy_identifier_selects_condition_without_temporal_resampling(self):
        self.entry["train"] = ["train_scene__fps10", "train_scene__fps20"]
        self.write_split()
        dataset = self.dataset()
        self.assertIn("low_light_10", dataset.items[0][1][0])
        self.assertEqual(len(dataset.items[0][1]), 2)
        self.assertEqual(dataset[0]["lq"].shape[1], 4)
        self.assertEqual(self.module.parse_sequence("scene__20fps"), ("scene", "20"))


if __name__ == "__main__":
    unittest.main()
