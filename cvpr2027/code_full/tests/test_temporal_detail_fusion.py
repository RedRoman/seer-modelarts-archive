"""Tiny CPU checks for experimental neighbour-feature transport, no model weights."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    import einops  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch and einops are required for CPU checks")
class TemporalDetailFusionTests(unittest.TestCase):
    def setUp(self):
        from models.temporal_detail_fusion import TemporalDetailFusion
        self.fusion = TemporalDetailFusion
        torch.manual_seed(7)

    def test_zero_initialization_is_identity_and_output_projection_can_learn(self):
        module = self.fusion(4, key_dim=3, radius=1, offset=2, guidance=1,
                             token_grid=(2, 2, 2))
        features = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        tokens = torch.randn(1, 8, 6)
        output, aux = module(features, tokens)
        self.assertTrue(torch.equal(output, features))
        self.assertTrue(all(value.ndim == 0 and not value.requires_grad for value in aux.values()))
        (output - torch.randn_like(output)).square().mean().backward()
        self.assertGreater(module.output_proj.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(module.output_proj.weight.grad).all())

    def test_values_really_come_from_neighbour_frames_and_bypass_restores_input(self):
        module = self.fusion(2, key_dim=2, radius=0, offset=2, guidance=0,
                             token_grid=(2, 1, 1))
        with torch.no_grad():
            module.output_proj.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            module.confidence.weight.zero_()
            module.confidence.bias.fill_(30)
        frames = torch.arange(1., 5.).reshape(1, 1, 4, 1, 1)
        features = frames.expand(1, 2, 4, 2, 2).clone().requires_grad_(True)
        output, aux = module(features, return_aux=True)
        torch.testing.assert_close(output, features[:, :, [2, 3, 0, 1]])
        self.assertEqual(aux["confidence_map"].shape, (1, 1, 4, 2, 2))
        self.assertTrue(aux["confidence_logits"].requires_grad)
        output[:, :, 0].sum().backward()
        self.assertGreater(features.grad[:, :, 2].abs().sum().item(), 0)
        bypassed, bypass_aux = module(features, bypass=True)
        self.assertTrue(torch.equal(bypassed, features))
        self.assertEqual(bypass_aux["bypassed"].item(), 1)

    def test_centered_semantics_ignore_clip_dc_and_teacher_is_loss_only(self):
        module = self.fusion(4, key_dim=3, radius=1, offset=2, guidance=1,
                             token_grid=(2, 2, 2))
        features = torch.randn(1, 4, 4, 4, 4)
        tokens = torch.randn(1, 8, 6)
        dc = torch.randn(1, 1, 6) * 3
        scores = module._semantic_scores(tokens, 1, 4, 4, 4)
        shifted = module._semantic_scores(tokens + dc, 1, 4, 4, 4)
        torch.testing.assert_close(scores, shifted, atol=2e-6, rtol=2e-6)
        self.assertGreater(scores.std().item(), 0)
        teacher = torch.randn(1, 8, 6, requires_grad=True)
        plain, _ = module(features, tokens)
        supervised, aux = module(features, tokens, teacher_tokens=teacher)
        self.assertTrue(torch.equal(plain, supervised))
        self.assertGreater(aux["correspondence_loss"].item(), 0)
        aux["correspondence_loss"].backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(module.match_proj.weight.grad.abs().sum().item(), 0)

    def test_no_valid_temporal_neighbour_is_finite_identity(self):
        module = self.fusion(3, key_dim=2, radius=2, offset=2, guidance=0,
                             token_grid=(1, 1, 1))
        with torch.no_grad():
            module.output_proj.weight.normal_()
        features = torch.randn(1, 3, 1, 2, 2)
        output, aux = module(features)
        self.assertTrue(torch.equal(output, features))
        self.assertEqual(aux["valid_query_fraction"].item(), 0)
        self.assertTrue(all(torch.isfinite(value).all() for value in aux.values()))

    def test_subclass_loads_base_and_local_prior_have_identical_state(self):
        from models.pixel_branch_v2 import SeerV2PixelBranch
        from models.pixel_branch_fusion import SeerFusionPixelBranch
        kwargs = dict(dims=(8, 16, 32, 64, 128), num_blocks=(0, 0, 0, 0, 0),
                      heads=(1, 1, 1, 1, 1), vjepa_dim=8, vjepa_grid=(2, 2, 2),
                      use_cfim_bottleneck=False, use_illumination_cues=False,
                      use_gradient_checkpoint=False)
        base = SeerV2PixelBranch(**kwargs).eval()
        # Nonzero head makes parity check meaningful beyond the residual init.
        with torch.no_grad():
            base.residual_head.weight.normal_(std=0.01)
        torch.manual_seed(123)
        local = SeerFusionPixelBranch(**kwargs, fusion_mode="local").eval()
        torch.manual_seed(123)
        prior = SeerFusionPixelBranch(**kwargs, fusion_mode="prior").eval()
        self.assertEqual(list(local.state_dict()), list(prior.state_dict()))
        for key, value in local.state_dict().items():
            self.assertTrue(torch.equal(value, prior.state_dict()[key]), key)
        for model in (local, prior):
            missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
            self.assertTrue(missing and all(key.startswith("temporal_fusion.") for key in missing))
            self.assertEqual(unexpected, [])
            video = torch.rand(1, 3, 4, 32, 32)
            tokens = [torch.randn(1, 8, 8) for _ in range(4)]
            with torch.no_grad():
                reference, actual = base(video, tokens), model(video, tokens)
            for key in ("pred_pixels", "residual", "R_hat", "L_hat"):
                torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)
        self.assertEqual(prior.get_fusion_config()["fusion_mode"], "prior")
        restored = SeerFusionPixelBranch(**kwargs, **prior.get_fusion_config())
        restored.load_state_dict(prior.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
