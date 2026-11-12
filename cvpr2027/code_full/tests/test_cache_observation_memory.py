"""Exporter identity, strict state loading and LQ-only input guards."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "tools"))
spec = importlib.util.spec_from_file_location("memory_export_test", CODE / "tools/cache_observation_memory.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExportGuards(unittest.TestCase):
    def origin(self):
        return {"base_saved_args": {"tag": "e17_small_pretrained_s0", "seed": 0,
                "iters": 20000, "taps": "5,11,17,23", "vjepa": "pretrained"},
                "encoder_weights_sha256": m.ENCODER_SHA,
                "args": {"data_seed": 20261113, "crop": 256, "clip_len": 16}}

    def test_origin_rejects_wrong_arm_tap_and_encoder(self):
        m.check_origin(self.origin())
        for key, value in (("taps", "5,11,23,23"), ("tag", "e17_small_none_s0"),
                           ("vjepa", "random"), ("iters", 19999)):
            p = self.origin(); p["base_saved_args"][key] = value
            with self.assertRaises(ValueError): m.check_origin(p)
        p = self.origin(); p["encoder_weights_sha256"] = "0"*64
        with self.assertRaises(ValueError): m.check_origin(p)

    def state(self):
        return {"encoder": {"module.backbone.live": torch.ones(2),
                "module.backbone.patch_embed_img.proj.weight": torch.empty(1024, 3, 1, 16, 16),
                "module.backbone.patch_embed_img.proj.bias": torch.empty(1024)}}

    def test_only_exact_unused_image_keys_are_dropped(self):
        clean, ignored = m.clean_encoder_state(self.state(), {"live": torch.ones(2)})
        self.assertEqual(list(clean), ["live"])
        self.assertEqual(len(ignored), 2)
        for key in ("module.backbone.extra", "live"):
            state = self.state(); state["encoder"][key] = torch.ones(1)
            with self.assertRaises(ValueError): m.clean_encoder_state(state, {"live": torch.ones(2)})
        with self.assertRaises(ValueError): m.clean_encoder_state(self.state(), {"live": torch.ones(3)})

    def test_project_real_helper_is_cpu_deterministic_and_rng_preserving(self):
        helper = m.local_module("memory_projection_test", CODE / "models/observation_memory.py")
        tokens = torch.randn(1, 2048, 1024, generator=torch.Generator().manual_seed(5))
        encoded = {"token_grid": (8,16,16), "hidden_dim": 1024, "feats": [tokens]}
        rng = torch.random.get_rng_state().clone()
        p = m.project_clip(encoded, helper)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(p.shape, (1,2048,64))
        self.assertTrue(torch.equal(p, m.project_clip(encoded, helper)))
        self.assertLessEqual(float(p.norm(dim=-1).max()), 1+2e-5)
        encoded["token_grid"] = (16,8,16)
        with self.assertRaises(ValueError): m.project_clip(encoded, helper)

    def test_source_lq_loaded_without_using_gt_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"clip.pt"
            plan = {"sequence_index": 0, "augmentation_seed": 1}
            low = torch.full((1,3,16,256,256), .2)
            # GT is intentionally invalid: exporter must not consult it.
            torch.save({"name": "SDSD_indoor/pair1", "scene": "SDSD_indoor/pair1",
                        "plan": plan, "tensors": {"lq": low, "gt": "forbidden"}}, path)
            row = {"name": "SDSD_indoor/pair1", "scene": "SDSD_indoor/pair1", "cache_sha256": m.digest(path)}
            self.assertTrue(torch.equal(m.load_lq(path, row, plan), low))
            with self.assertRaises(ValueError): m.load_lq(path, row, {"sequence_index": 1})
            row["cache_sha256"] = "0"*64
            with self.assertRaises(ValueError): m.load_lq(path, row, plan)

    def test_state_hash_covers_parameter_and_buffer_changes(self):
        model = torch.nn.BatchNorm1d(3)
        old = m.state_sha(model)
        model.running_mean[1] += 1
        self.assertNotEqual(old, m.state_sha(model))

    def test_json_outputs_are_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"a.json"
            m.write_json_new(path, {"ok": True})
            with self.assertRaises(FileExistsError): m.write_json_new(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
