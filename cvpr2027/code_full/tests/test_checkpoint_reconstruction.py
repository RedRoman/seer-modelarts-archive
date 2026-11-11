"""CPU-only reconstruction checks; no datasets, pretrained weights or accelerators.

Run: python -m unittest discover -s tests -p 'test_checkpoint_reconstruction.py' -v
Configuration tests also run without torch; tensor tests are explicitly skipped.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / "tools")]
from checkpoint_config import (encoder_initialization_seed, resolve_basis_path,
                               resolve_config)

try:
    import torch
    import einops  # noqa: F401
except ImportError:
    torch = None


class ConfigTests(unittest.TestCase):
    def test_baseline_defaults_have_same_conditioning(self):
        saved, legacy = resolve_config({}), resolve_config({}, mode="legacy")
        for key in ("arch", "arm", "branch", "xattn", "kv", "layers", "taps", "token_proj"):
            self.assertEqual(saved[key], legacy[key], key)

    def test_saved_variants_and_explicit_legacy(self):
        args = dict(branch="small", xattn="gated", kv="hybrid", layers="single",
                    taps="17,17,23,23", token_proj="clip_rank", token_proj_k=4)
        saved = resolve_config(args)
        self.assertEqual(saved["taps"], [17, 17, 23, 23])
        self.assertEqual(saved["kv"], "hybrid")
        self.assertEqual(saved["token_proj"], "clip_rank")
        legacy = resolve_config(args, mode="legacy")
        self.assertEqual(legacy["taps"], [5, 11, 17, 23])
        self.assertEqual(legacy["kv"], "per_tubelet")
        self.assertEqual(legacy["token_proj"], "none")
        self.assertEqual(resolve_config({"layers": "all"})["layers"], "all")

    def test_random_seed_does_not_change_pretrained_arm(self):
        self.assertIsNone(resolve_config({"vjepa": "pretrained"})["random_encoder_seed"])
        self.assertEqual(resolve_config({"vjepa": "random"})["random_encoder_seed"], 1234)
        forced = resolve_config({"vjepa": "pretrained"}, force_arm="random")
        self.assertEqual(forced["trained_arm"], "pretrained")
        self.assertEqual(forced["arm"], "random")
        self.assertEqual(forced["random_encoder_seed"], 1234)
        self.assertIsNone(resolve_config({"vjepa": "random"}, mode="legacy")["random_encoder_seed"])

    def test_invalid_configuration_is_rejected_before_loading(self):
        for args in ({"arch": "typo"}, {"branch": "tiny"}, {"xattn": "typo"},
                     {"taps": "23"}, {"taps": "5,11,17,24"}, {"clip_len": 15},
                     {"kv": "full", "layers": "all"},
                     {"token_proj": "global_rank", "token_proj_k": 0}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_config(args)
        with self.assertRaises(ValueError):
            resolve_config({"vjepa": "none"}, force_arm="random")

    def test_basis_relocation_requires_the_saved_file_or_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basis = root / "results" / "basis.json"
            basis.parent.mkdir()
            basis.write_text("{}")
            self.assertEqual(resolve_basis_path("/cache/zihaohe/seer_cvpr/results/basis.json",
                                               root, root / "final.pt"), basis.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_basis_path("/unknown/basis.json", root, root / "final.pt")
            self.assertEqual(resolve_basis_path("/unknown/basis.json", root, root / "final.pt",
                                               override=basis), basis.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_basis_path(str(basis), root, root / "final.pt", override=root / "missing")


@unittest.skipIf(torch is None, "torch and einops are required for CPU tensor checks")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # build_model does not call baseline metrics. Stub only that external
        # import so these CPU tests don't require the unvendored baseline tree.
        stub = types.ModuleType("common.eval_lib")
        stub._load_img_any = stub.center_crop = stub.rgb_psnr_uint8 = None
        with patch.dict(sys.modules, {"common.eval_lib": stub}):
            cls.evaluator = importlib.import_module("eval_e4")

    def test_cpu_encoder_seed_matches_training_and_restores_rng(self):
        torch.manual_seed(99)
        state = torch.random.get_rng_state().clone()
        with encoder_initialization_seed(1234):
            first = torch.rand(12)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.manual_seed(1234)
        self.assertTrue(torch.equal(first, torch.rand(12)))
        torch.manual_seed(2048)
        with encoder_initialization_seed(1234):
            self.assertTrue(torch.equal(first, torch.rand(12)))

    def test_gated_branch_constructs_and_preserves_initial_residual_identity(self):
        from models.pixel_branch_v2 import SeerV2PixelBranch
        from models.xattn_gated import VJepaCrossAttnGated
        branch = SeerV2PixelBranch(
            dims=(8, 16, 32, 64, 128), num_blocks=(0, 0, 0, 0, 0),
            heads=(1, 1, 1, 1, 1), vjepa_dim=8, vjepa_grid=(1, 2, 2),
            use_cfim_bottleneck=False, use_illumination_cues=False,
            use_gradient_checkpoint=False, xattn="gated").eval()
        self.assertIsInstance(branch.enc_xattn[1], VJepaCrossAttnGated)
        self.assertIsInstance(branch.dec_xattn[0], VJepaCrossAttnGated)
        x = torch.rand(1, 3, 2, 32, 32)
        with torch.no_grad():
            out = branch(x, [torch.randn(1, 4, 8) for _ in range(4)])
        self.assertTrue(torch.equal(out["pred_pixels"], x))

    def test_projection_math_and_float32_autocast_boundary(self):
        from models.token_projection import ProjectedBackbone, project_tokens
        z = torch.tensor([[[1., 2., 3.], [3., 4., 5.], [5., 6., 7.]]])
        basis = {"mean": torch.zeros(1, 3), "comp": torch.eye(3)[None]}
        projected = project_tokens(z, "global_rank", 1, basis, 0)
        self.assertTrue(torch.equal(projected[..., 0], z[..., 0]))
        self.assertEqual(projected[..., 1:].count_nonzero().item(), 0)
        torch.testing.assert_close(project_tokens(z, "clip_rank", 1, None, 0), z)

        class Backbone(torch.nn.Module):
            def forward(self, video):
                return {"feats": [z.to(torch.bfloat16)], "token_grid": (1, 1, 3)}

        wrapper = ProjectedBackbone(Backbone(), "global_rank", 1, basis)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = wrapper(torch.zeros(1))
        self.assertEqual(result["feats"][0].dtype, torch.float32)
        torch.testing.assert_close(result["feats"][0], projected)
        self.assertEqual(result["token_grid"], (1, 1, 3))

    def _build(self, args, state=None, **kwargs):
        # Real torch state serialization + fake tiny constructors exercise the
        # evaluator's config and strictness, without allocating a 361M model.
        class Pixel(torch.nn.Module):
            def __init__(self, **config):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = config

        class Backbone(torch.nn.Module):
            def __init__(self, **config):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(8), requires_grad=False)
                self.config = config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "final.pt"
            torch.save({"args": args, "pixel_branch": state or {"weight": torch.ones(1)}}, ckpt)
            with patch.object(self.evaluator, "SeerV2PixelBranch", Pixel), \
                 patch.object(self.evaluator, "VJepaBackboneLocal21", Backbone):
                return self.evaluator.build_model(str(ckpt), torch.device("cpu"), **kwargs)

    def test_evaluator_restores_configuration_and_random_weights(self):
        args = dict(arch="v2", branch="small", vjepa="random", xattn="gated",
                    kv="hybrid", taps="17,17,23,23", token_proj="clip_rank", token_proj_k=4)
        torch.manual_seed(7)
        p1, b1, *_ = self._build(args)
        torch.manual_seed(81)
        p2, b2, *_ = self._build(args)
        self.assertEqual(p1.config["xattn"], "gated")
        self.assertEqual(p1.config["kv_mode"], "hybrid")
        self.assertEqual(b1.backbone.config["feature_layers"], (17, 17, 23, 23))
        self.assertEqual(b1.mode, "clip_rank")
        self.assertEqual(b1.backbone.config["local_ckpt"], "")
        self.assertTrue(torch.equal(b1.backbone.weight, b2.backbone.weight))
        with encoder_initialization_seed(1234):
            self.assertTrue(torch.equal(b1.backbone.weight, torch.rand(8)))
        _, pretrained, *_ = self._build(dict(vjepa="pretrained"))
        self.assertEqual(pretrained.config["local_ckpt"], self.evaluator.VJEPA_CKPT)
        p_all, *_ = self._build(dict(vjepa="none", layers="all"))
        self.assertEqual(p_all.config["layer_mode"], "all")

    def test_evaluator_rejects_checkpoint_keys_that_do_not_match(self):
        for state in ({"wrong": torch.ones(1)},
                      {"weight": torch.ones(1), "unexpected": torch.ones(1)}):
            with self.subTest(keys=list(state)), self.assertRaisesRegex(RuntimeError, "missing=.*unexpected="):
                self._build({"vjepa": "none"}, state=state)

    def test_evaluator_rejects_basis_tap_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            basis = Path(tmp) / "basis.json"
            basis.write_text(json.dumps(dict(taps=[23, 23, 23, 23], rank=64)))
            with self.assertRaisesRegex(ValueError, "basis taps"):
                self._build(dict(token_proj="global_rank", token_basis=str(basis)))


if __name__ == "__main__":
    unittest.main()
