"""Tiny CPU tensor checks for the post-hoc validation mean/basis control."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is required for CPU tensor checks")
class MeanBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mean_basis_control
        cls.control = mean_basis_control

    def setUp(self):
        self.field = torch.diag(torch.arange(1., 7.))[None] + torch.arange(6.)
        self.training_mean = torch.arange(6.) - 3
        self.training_basis = torch.eye(6)[:4]
        self.statistics = self.control.prepare_token_statistics(self.field)

    def test_identity_mean_only_and_all_four_crossed_conditions(self):
        outputs = {name: self.control.apply_token_condition(
            self.statistics, self.training_mean, self.training_basis, name)
            for name in self.control.CONDITIONS}
        self.assertEqual(len(outputs), 6)
        self.assertTrue(torch.equal(outputs["identity"], self.field))
        mean = self.field.mean(1, keepdim=True)
        torch.testing.assert_close(outputs["clip_mean_only"], mean.expand_as(self.field))
        self.assertFalse(torch.allclose(outputs["clip_mean_only"], outputs["clip_mean_clip_basis"]))
        # A canonical training basis must retain exactly the first four
        # coordinates; discarded coordinates equal the chosen mean.
        for source, selected_mean in (("clip", mean[0, 0]), ("training", self.training_mean)):
            actual = outputs[f"{source}_mean_training_basis"]
            torch.testing.assert_close(actual[..., :4], self.field[..., :4])
            torch.testing.assert_close(actual[..., 4:], selected_mean[4:].reshape(1, 1, 2).expand(1, 6, 2))
        for key in ("clip_mean_clip_basis", "clip_mean_training_basis"):
            torch.testing.assert_close(outputs[key].mean(1, keepdim=True), mean)
        # Changing the mean must not recompute or rotate the adaptive basis.
        basis = self.statistics["clip_basis"]
        projector = basis.T @ basis
        expected_difference = (self.training_mean - mean[0, 0]) @ (torch.eye(6) - projector)
        torch.testing.assert_close(outputs["training_mean_clip_basis"] - outputs["clip_mean_clip_basis"],
                                   expected_difference.reshape(1, 1, 6).expand_as(self.field), atol=2e-6, rtol=1e-5)

    def test_svd_runs_once_and_is_clip_centered_float32(self):
        with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                statistics = self.control.prepare_token_statistics(self.field)
                for condition in self.control.CONDITIONS:
                    output = self.control.apply_token_condition(statistics, self.training_mean,
                                                                 self.training_basis, condition)
                    self.assertEqual(output.dtype, torch.float32)
            self.assertEqual(svd.call_count, 1)
            centered = svd.call_args.args[0]
            self.assertEqual(centered.device.type, "cpu")
            torch.testing.assert_close(centered.mean(0), torch.zeros(6), atol=1e-6, rtol=0)

    def test_invalid_selection_and_projection_dimensions_fail(self):
        names = [f"SDSD_indoor/pair{i}" for i in range(16)]
        self.control.validate_selection(names, names)
        for actual in (names[:-1], names[:-1] + [names[0]]):
            with self.assertRaises(ValueError):
                self.control.validate_selection(actual, names)
        with self.assertRaises(ValueError):
            self.control.apply_token_condition(self.statistics, self.training_mean[:-1], self.training_basis, "identity")
        with self.assertRaises(ValueError):
            self.control.apply_token_condition(self.statistics, self.training_mean, self.training_basis, "unknown")

    def test_backbone_and_svd_are_cached_per_clip_across_conditions(self):
        field = self.field

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, video):
                self.calls += 1
                return {"feats": [field.clone() for _ in range(4)], "token_grid": (1, 2, 3)}

        backbone = Backbone()
        controlled = self.control.ControlledBackbone(
            backbone, self.training_mean.repeat(4, 1), self.training_basis.repeat(4, 1, 1))
        video = torch.zeros(1, 3, 16, 2, 2)
        with patch.object(torch.linalg, "svd", wraps=torch.linalg.svd) as svd:
            controlled.begin_clip("SDSD_indoor/test")
            for condition in self.control.CONDITIONS:
                controlled.condition = condition
                result = controlled(video)
                if condition == "identity":
                    torch.testing.assert_close(result["feats"][0], field)
            self.assertEqual(backbone.calls, 1)
            self.assertEqual(svd.call_count, 4)
            controlled.begin_clip("SDSD_indoor/next")
            controlled(video)
            self.assertEqual(backbone.calls, 2)
            self.assertEqual(svd.call_count, 8)


if __name__ == "__main__":
    unittest.main()
