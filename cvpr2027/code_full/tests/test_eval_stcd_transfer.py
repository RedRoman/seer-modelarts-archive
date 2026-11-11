"""Data-free STCD protocol guards; tensor checks require an existing torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_stcd_transfer_test", CODE / "tools/eval_stcd_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
try:
    import torch
except ImportError:
    torch = None


def fixture_split(directory):
    entry = {"root": str(directory / "dataset")}
    offset = 0
    for split, scenes in (("train", 24), ("val", 8), ("test", 8)):
        entry[split] = [f"scene{i:02d}__{condition}fps"
                        for i in range(offset, offset + scenes) for condition in (10, 20)]
        offset += scenes
    for name in entry["val"]:
        scene, condition = transfer._scene_condition(name)
        for kind in ("low", "normal"):
            path = Path(entry["root"]) / scene / f"{kind}_light_{condition}"
            path.mkdir(parents=True)
            for index in range(20):
                (path / f"{index:04d}.png").touch()
    path = directory / "split.json"
    path.write_text(json.dumps({"BVI_RLV": entry}))
    return path, entry


class ProtocolTests(unittest.TestCase):
    def test_pinned_source_verification_precedes_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "arch.py").write_text("raise AssertionError('must never execute during verification')\n")
            expected = {"arch.py": transfer.digest(root / "arch.py")}
            self.assertIn("arch.py", transfer.verified_sources(root, expected))
            (root / "arch.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                transfer.verified_sources(root, expected)

    def test_only_two_forced_moves_and_isolated_import_are_adapted(self):
        original = "import models.archs.arch_util as arch_util\na=f().to('cuda')\nb=g().to('cuda')\ny=a+b\n"
        adapted = transfer.adapt_architecture(original)
        self.assertNotIn(".to('cuda')", adapted)
        self.assertIn("a=f()\nb=g()\ny=a+b\n", adapted)
        for bad in (original.replace("b=g().to('cuda')", "b=g()"), original + "c=h().to('cuda')", original.replace("import models.archs.arch_util as arch_util", "")):
            with self.assertRaises(ValueError):
                transfer.adapt_architecture(bad)

    def test_inventory_checks_complete_pairs_scene_split_and_centre_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, entry = fixture_split(Path(tmp))
            records = transfer.validation_inventory(path, tmp)
            self.assertEqual(len(records), 16)
            self.assertEqual(records[0]["centre_start"], 2)
            self.assertEqual(records[0]["selected_frame_names"], [f"{i:04d}.png" for i in range(2, 18)])
            first = records[0]
            high = Path(first["directories"][1])
            (high / "0000.png").rename(high / "0099.png")
            with self.assertRaisesRegex(ValueError, "paired frame lists"):
                transfer.validation_inventory(path, tmp)
            (high / "0099.png").rename(high / "0000.png")
            entry["test"][0] = entry["val"][0]
            entry["test"][1] = entry["val"][1]
            path.write_text(json.dumps({"BVI_RLV": entry}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                transfer.validation_inventory(path, tmp)

    def test_inventory_rejects_missing_frames_or_duplicate_validation_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, entry = fixture_split(Path(tmp))
            scene, condition = transfer._scene_condition(entry["val"][0])
            (Path(entry["root"]) / scene / f"low_light_{condition}" / "0000.png").unlink()
            with self.assertRaises(ValueError):
                transfer.validation_inventory(path, tmp)
            entry["val"][0] = entry["val"][1]
            path.write_text(json.dumps({"BVI_RLV": entry}))
            with self.assertRaisesRegex(ValueError, "val16"):
                transfer.validation_inventory(path, tmp)

    def test_prediction_receives_only_low_and_scoring_is_later(self):
        low = np.zeros((16, 256, 256, 3), np.uint8)
        gt = np.full_like(low, 99)
        events = []
        def predict(value):
            self.assertIs(value, low)
            events.append("predict")
            return value.copy()
        def score(prediction, target):
            self.assertEqual(events[-1], "predict")
            self.assertIs(target, gt)
            events.append("score")
            return {"metric": float(target.mean())}
        with mock.patch("builtins.print"):
            rows = transfer.evaluate_sequences([("BVI/example", low, gt, "video")], predict, score, ["BVI/example"])
        self.assertEqual(events, ["predict", "score"])
        self.assertEqual(rows[0]["metric"], 99)
        with self.assertRaisesRegex(ValueError, "complete"):
            transfer.evaluate_sequences([], predict, score, ["BVI/example"])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            transfer.evaluate_sequences([("wrong", low, gt, "video")], predict, score, ["BVI/example"])


@unittest.skipIf(torch is None, "torch required for CPU tensor checks")
class TensorTests(unittest.TestCase):
    def test_loader_is_strict_and_never_requires_cuda_or_training_wrapper(self):
        source = """import torch
import torch.nn as nn
import models.archs.arch_util as arch_util
class low_light_transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.first = nn.Identity().to('cuda')
        self.second = nn.Identity().to('cuda')
    def forward(self, a, b):
        return (a * self.weight,) * 6
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, value in (("models/archs/arch_util.py", "# tiny verified fixture\n"),
                                ("models/archs/low_light_transformer42.py", source)):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            checkpoint = root / "fixture.pt"
            torch.save({"weight": torch.tensor([.5])}, checkpoint)
            hashes = {name: transfer.digest(root / name) for name in transfer.SOURCE_SHA256}
            with mock.patch.object(transfer, "SOURCE_SHA256", hashes), mock.patch.object(
                    transfer, "CHECKPOINT_SHA256", transfer.digest(checkpoint)):
                model, provenance = transfer.load_stcd(root, checkpoint, "cpu")
                self.assertTrue(provenance["load_state_dict_strict"])
                self.assertEqual(model.weight.item(), .5)
                self.assertFalse(model.weight.requires_grad)
            torch.save({"wrong_name": torch.tensor([.5])}, checkpoint)
            with mock.patch.object(transfer, "SOURCE_SHA256", hashes), mock.patch.object(
                    transfer, "CHECKPOINT_SHA256", transfer.digest(checkpoint)):
                with self.assertRaises(RuntimeError):
                    transfer.load_stcd(root, checkpoint, "cpu")

    def test_two_frame_context_channel_swap_output_index_and_last_duplication(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, first, second):
                self.calls.append((first.clone(), second.clone()))
                # Only index2 is a valid enhanced image, with asymmetric weights.
                result = .25 * first + .75 * second
                return (torch.zeros_like(result), torch.zeros_like(result), result,
                        torch.zeros_like(result), torch.zeros_like(result), torch.zeros_like(result))
        low = torch.arange(3 * 3 * 8 * 8).reshape(3, 3, 8, 8).float() / 1000
        model = Spy()
        output = transfer.enhance_clip(model, low, chunk=2)
        expected = .25 * low + .75 * torch.cat((low[1:], low[-1:]))
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(model.calls[0][0], low[:2, [2, 1, 0]])
        torch.testing.assert_close(model.calls[-1][1], low[-1:, [2, 1, 0]])

    def test_prediction_rounds_uint8_and_rejects_bad_input_or_output(self):
        class Identity(torch.nn.Module):
            def forward(self, first, second):
                return (first,) * 6
        x = np.arange(3 * 8 * 8 * 3, dtype=np.uint16).reshape(3, 8, 8, 3).astype(np.uint8)
        np.testing.assert_array_equal(transfer.predict_uint8(Identity(), x, "cpu", 1), x)
        for bad in (torch.zeros(1, 3, 8, 8), torch.zeros(3, 3, 7, 8), torch.full((3, 3, 8, 8), float("nan")), torch.ones(3, 3, 8, 8) * 1.1):
            with self.assertRaises(ValueError):
                transfer.enhance_clip(Identity(), bad)
        with self.assertRaises(ValueError):
            transfer.enhance_clip(lambda a, b: (a,), torch.zeros(3, 3, 8, 8))

    def test_metrics_use_quantized_pixels_and_absolute_temporal_gap(self):
        p = np.full((16, 4, 4, 3), 40, np.uint8)
        g = np.arange(16, dtype=np.uint8)[:, None, None, None] + np.full_like(p, 50)
        fake = types.SimpleNamespace(
            lpips_of=lambda a, b, dev: float((a - b).abs().mean()),
            psnr_y_uint8=lambda a, b: float(np.abs(a.astype(float) - b).mean()),
            rgb_psnr_uint8=lambda a, b: float(np.square(a.astype(float) - b).mean()))
        with mock.patch.dict("sys.modules", {"eval_e4": fake}):
            result = transfer.score_uint8(p, g, "cpu")
        self.assertEqual(result["tlp"], 0)
        self.assertAlmostEqual(result["gt_tlp"], 1 / 255, places=7)
        self.assertAlmostEqual(result["tlp_gap"], 1 / 255, places=7)
        self.assertAlmostEqual(result["lpips"], 17.5 / 255, places=7)
        self.assertEqual(result["psnr_y"], 17.5)


if __name__ == "__main__":
    unittest.main()
