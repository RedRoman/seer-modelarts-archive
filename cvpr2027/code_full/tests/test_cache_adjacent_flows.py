"""Complete-manifest guards and synthetic CPU fake-RAFT tests; no real data."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_adjacent_cache_test", CODE / "tools/cache_adjacent_flows.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
try:
    import torch
except ImportError:
    torch = None


def metadata(directory, names, *, payloads=None):
    directory.mkdir()
    plan = [{"sequence_index": i, "augmentation_seed": i + 10} for i in range(len(names))]
    rows = []
    for i, name in enumerate(names):
        path = directory / f"clip_{i:03d}.pt"
        if payloads is None:
            path.write_bytes(f"fixture{i}".encode())
        else:
            torch.save({"name": name, "scene": tool._scene(name), "plan": plan[i], "tensors": payloads[i]}, path)
        rows.append({"name": name, "scene": tool._scene(name), "cache_path": path.name,
                     "cache_sha256": tool.digest(path)})
    p = {"domain": "sdsd", "source_names": names, "plan": plan,
         "split_sha256": "1" * 64, "base_checkpoint_sha256": "2" * 64,
         "base_state_sha256": "3" * 64, "raft_weights_sha256": tool.RAFT_WEIGHTS_SHA,
         "raft_state_sha256": "4" * 64,
         "base_saved_args": {"tag": "e17_small_pretrained_s0", "seed": 0, "iters": 20000},
         "args": {"clip_len": tool.BASE_SHAPE[2], "crop": tool.BASE_SHAPE[-1],
                  "offset": 2, "flow_steps": 12, "flow_batch": 2}}
    s = {"n": len(names), "per_sequence": rows, "oracle_computed": False,
         "base_state_unchanged": True, "raft_state_unchanged": True}
    (directory / "provenance.json").write_text(json.dumps(p))
    (directory / "cache_summary.json").write_text(json.dumps(s))
    return p, s


class ManifestTests(unittest.TestCase):
    def names(self):
        return [f"SDSD_indoor/a{i:03d}" for i in range(136)]

    def test_full_136_manifest_requires_every_original_sha_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "cache"
            _, summary = metadata(cache, self.names())
            manifest = tool.base_manifest(cache)
            self.assertEqual(len(manifest["rows"]), 136)
            self.assertEqual(manifest["kind"], "full")
            (cache / "clip_135.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                tool.base_manifest(cache)
            # Even replacing the modified file's SHA cannot legalize a reorder.
            summary["per_sequence"][135]["cache_sha256"] = tool.digest(cache / "clip_135.pt")
            summary["per_sequence"][0], summary["per_sequence"][1] = summary["per_sequence"][1], summary["per_sequence"][0]
            (cache / "cache_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "order/name/scene/plan"):
                tool.base_manifest(cache)

    def test_missing_file_or_incomplete_state_fails_before_export(self):
        with tempfile.TemporaryDirectory() as td:
            cache, out = Path(td) / "cache", Path(td) / "out"
            _, summary = metadata(cache, self.names())
            (cache / "clip_050.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                tool.export_slim(cache, out)
            self.assertFalse(out.exists())
            summary["base_state_unchanged"] = False
            (cache / "cache_summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "unmodified"):
                tool.base_manifest(cache)

    def test_output_path_is_exclusive_and_resolved_under_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "allowed"; root.mkdir()
            outside = Path(td) / "sfs"; outside.mkdir()
            with patch.object(tool, "CACHE_ROOT", root):
                with self.assertRaisesRegex(ValueError, "under /cache"):
                    tool._new_output(outside / "out")
                (root / "link").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "resolves outside"):
                    tool._new_output(root / "link/out")
                tool._new_output(root / "out")
                with self.assertRaisesRegex(ValueError, "already exists"):
                    tool._new_output(root / "out")
                tool._atomic_json(root / "out/a.json", {"a": 1})
                with self.assertRaisesRegex(ValueError, "existing output"):
                    tool._atomic_json(root / "out/a.json", {"a": 2})
                self.assertEqual(json.loads((root / "out/a.json").read_text()), {"a": 1})

    def test_helper_signature_and_pinned_function_source(self):
        self.assertEqual(list(inspect.signature(tool.adjacent_flows).parameters), ["base", "raft"])
        path = CODE / "tools/cache_correction_oracle.py"
        source = path.read_text()
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "donor_correction"]
        self.assertEqual(len(nodes), 1)
        self.assertEqual(tool.digest(path), tool.DONOR_SOURCE_SHA)
        self.assertEqual(hashlib.sha256(ast.get_source_segment(source, nodes[0]).encode()).hexdigest(),
                         tool.DONOR_FUNCTION_SHA)


if torch is not None:
    class FakeRAFT(torch.nn.Module):
        def __init__(self, displacement=0., mutate=False):
            super().__init__()
            self.register_buffer("fixed", torch.tensor(displacement, dtype=torch.float32))
            self.calls, self.mutate = [], mutate
            self.eval().requires_grad_(False)

        def forward(self, target, source, *, num_flow_updates):
            self.calls.append((target.detach().clone(), source.detach().clone(), num_flow_updates))
            if self.mutate:
                self.fixed.add_(1)
            result = target.new_zeros(target.shape[0], 2, *target.shape[-2:])
            result[:, 0] = self.fixed
            return [result]


@unittest.skipIf(torch is None, "Torch CPU runtime required")
class TensorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.weight = self.root / "fake_weights.pt"; self.weight.write_bytes(b"CPU-only fake weights")
        self.patches = [patch.object(tool, "EXPECTED_CLIPS", 4),
                        patch.object(tool, "BASE_SHAPE", (1, 3, 4, 8, 8)),
                        patch.object(tool, "CACHE_ROOT", self.root),
                        patch.object(tool, "RAFT_WEIGHTS_SHA", tool.digest(self.weight))]
        for item in self.patches: item.start()
        self.names = ["SDSD_indoor/a", "SDSD_indoor/a_2", "SDSD_outdoor/a", "SDSD_outdoor/a_2"]
        self.base = torch.stack([torch.full((1, 3, 8, 8), .1 + t * .1) for t in range(4)], dim=2)

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def cache(self, name="full", gt_value=float("nan")):
        cache = self.root / name
        payloads = [{"base": self.base.clone(), "gt": torch.full_like(self.base, gt_value),
                     "lq": torch.full_like(self.base, float("nan")), "flow": "not consulted",
                     "valid_mask": "not consulted", "restored_corrections": "not consulted"} for _ in self.names]
        metadata(cache, self.names, payloads=payloads)
        return cache

    def test_adjacent_direction_batching_mask_and_four_frame_boundary(self):
        raft = FakeRAFT(displacement=1.)
        flow, mask = tool.adjacent_flows(self.base, raft)
        self.assertEqual(tuple(flow.shape), (1, 2, 2, 4, 8, 8))
        self.assertEqual(tuple(mask.shape), (1, 2, 1, 4, 8, 8))
        self.assertFalse(flow.requires_grad)
        self.assertFalse(bool(mask[:, 0, :, 0].any()))
        self.assertFalse(bool(mask[:, 1, :, -1].any()))
        self.assertFalse(bool(mask[..., -1].any()))
        self.assertTrue(bool(mask[:, 0, :, 1:, :, :-1].all()))
        self.assertTrue(bool(mask[:, 1, :, :-1, :, :-1].all()))
        self.assertEqual(len(raft.calls), 4)  # selected3 frames -> batches2+1 per direction
        self.assertTrue(all(call[2] == 12 and call[0].shape[0] <= 2 for call in raft.calls))
        # First two calls are past, next two future, normalized base only.
        targets = torch.cat([c[0] for c in raft.calls[:2]])
        sources = torch.cat([c[1] for c in raft.calls[:2]])
        frames = self.base[0].permute(1, 0, 2, 3)
        torch.testing.assert_close(targets, frames[1:] * 2 - 1, atol=0, rtol=0)
        torch.testing.assert_close(sources, frames[:-1] * 2 - 1, atol=0, rtol=0)
        torch.testing.assert_close(torch.cat([c[0] for c in raft.calls[2:]]), frames[:-1] * 2 - 1, atol=0, rtol=0)
        torch.testing.assert_close(torch.cat([c[1] for c in raft.calls[2:]]), frames[1:] * 2 - 1, atol=0, rtol=0)
        self.assertIs(tool._donor_correction(), tool._donor_correction())

    def test_export_is_base_only_self_contained_and_gt_agnostic(self):
        full = self.cache()
        out = self.root / "slim"
        summary = tool.export_slim(full, out)
        manifest = tool.base_manifest(out)
        original = tool.base_manifest(full)
        self.assertEqual(manifest["source_cache"], original["source_cache"])
        self.assertEqual(summary["n"], 4)
        for i in range(4):
            record = torch.load(out / f"base_{i:03d}.pt", weights_only=True)
            self.assertEqual(set(record["tensors"]), {"base"})
            self.assertNotIn("gt", record)
            self.assertTrue(torch.equal(record["tensors"]["base"], self.base))
            self.assertEqual(record["source_cache_sha256"], original["rows"][i]["cache_sha256"])
            self.assertEqual(tool.load_base(manifest, i)["base_tensor_sha256"], tool.tensor_digest(self.base))
        # Remove access to the full cache: slim has all required provenance.
        moved = self.root / "removed_full"
        full.rename(moved)
        self.assertEqual(tool.load_base(tool.base_manifest(out), 3)["name"], self.names[3])
        with self.assertRaisesRegex(ValueError, "already exists"):
            tool.export_slim(moved, out)

    def test_original_loader_accesses_no_other_tensor_key(self):
        full = self.cache()
        manifest = tool.base_manifest(full)
        record = torch.load(full / "clip_000.pt", weights_only=True)

        class BaseOnlyGuard(dict):
            def __getitem__(self, key):
                if key != "base": raise AssertionError("Non-base tensor field accessed")
                return super().__getitem__(key)

        record["tensors"] = BaseOnlyGuard(record["tensors"])
        with patch.object(torch, "load", return_value=record):
            self.assertTrue(torch.equal(tool.load_base(manifest, 0)["base"], self.base))

    def test_slim_detects_file_and_tensor_lineage_tampering(self):
        full, slim = self.cache(), self.root / "slim"
        tool.export_slim(full, slim)
        path = slim / "base_000.pt"
        record = torch.load(path, weights_only=True)
        record["tensors"]["base"][0, 0, 0, 0, 0] = .9
        torch.save(record, path)
        with self.assertRaisesRegex(ValueError, "file SHA"):
            tool.base_manifest(slim)
        # Even rewriting the file-level SHA cannot hide a base tensor change.
        s = json.loads((slim / "cache_summary.json").read_text())
        s["per_sequence"][0]["cache_sha256"] = tool.digest(path)
        (slim / "cache_summary.json").write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError, "base tensor SHA"):
            tool.load_base(tool.base_manifest(slim), 0)

    def test_flow_cache_full_and_slim_are_equal_and_preserve_lineage(self):
        full, slim = self.cache(), self.root / "slim"
        tool.export_slim(full, slim)
        outputs = []
        for source, name in ((full, "flow_full"), (slim, "flow_slim")):
            raft = FakeRAFT()
            result = tool.cache_flows(source, self.root / name, raft, self.weight, torch.device("cpu"))
            self.assertTrue(result["raft_state_unchanged"])
            self.assertEqual(result["raft_state_sha256_before"], result["raft_state_sha256_after"])
            self.assertFalse(result["gt_input"])
            for i in range(4):
                row = result["per_sequence"][i]
                self.assertEqual(row["cache_sha256"], tool.digest(self.root / name / row["cache_path"]))
                record = torch.load(self.root / name / row["cache_path"], weights_only=True)
                self.assertEqual(set(record["tensors"]), {"flow", "valid_mask"})
                self.assertEqual(record["base_tensor_sha256"], tool.tensor_digest(self.base))
                self.assertEqual(record["source_cache_sha256"], tool.base_manifest(full)["rows"][i]["cache_sha256"])
                if i == 0: outputs.append(record["tensors"])
        for key in ("flow", "valid_mask"):
            self.assertTrue(torch.equal(outputs[0][key], outputs[1][key]))

    def test_wrong_weights_or_mutated_raft_state_fail_without_complete_summary(self):
        full = self.cache()
        wrong = self.root / "wrong_weights.pt"; wrong.write_bytes(b"wrong")
        out = self.root / "wrong_out"
        with self.assertRaisesRegex(ValueError, "weights SHA"):
            tool.cache_flows(full, out, FakeRAFT(), wrong, torch.device("cpu"))
        self.assertFalse(out.exists())
        out = self.root / "mutated_out"
        with self.assertRaisesRegex(ValueError, "mutated RAFT"):
            tool.cache_flows(full, out, FakeRAFT(mutate=True), self.weight, torch.device("cpu"))
        self.assertFalse((out / "cache_summary.json").exists())

    def test_base_shape_range_dtype_and_train_mode_are_rejected(self):
        for base in (self.base.double(), self.base[:, :, :1], self.base[..., :-1],
                     self.base.repeat(2, 1, 1, 1, 1), self.base * float("nan"), self.base + 2):
            with self.assertRaises(ValueError): tool.adjacent_flows(base, FakeRAFT())
        with self.assertRaisesRegex(ValueError, "eval mode"):
            tool.adjacent_flows(self.base, FakeRAFT().train())


if __name__ == "__main__":
    unittest.main()
