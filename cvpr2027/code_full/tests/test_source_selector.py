"""Focused source/target boundary and protocol checks; no data or accelerators."""
import copy
import builtins
import importlib.util
import json
from pathlib import Path
import sys
import symtable
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import cache_source_corrections as cache
import train_source_selector as train
import eval_source_selector as evaluate

SPLITS = Path(__file__).resolve().parents[2] / "results/splits_cvpr.json"


def fixture_split():
    return {key: {"root": "unused", "train": ["a", "a_2"], "val": ["b"], "test": ["c"]}
            for key in ("SDSD_indoor", "SDSD_outdoor")}


class SourceProtocolTests(unittest.TestCase):
    def test_entrypoints_do_not_reference_undefined_globals(self):
        # Catch main-only errors (e.g. a parity accumulator initialized in the
        # wrong function) without importing models or opening target frames.
        for stem in ("cache_source_corrections", "train_source_selector", "eval_source_selector"):
            path = TOOLS / f"{stem}.py"
            table = symtable.symtable(path.read_text(), str(path), "exec")
            defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
            allowed = defined | set(dir(builtins)) | {"__file__", "__name__"}
            pending = list(table.get_children())
            while pending:
                scope = pending.pop()
                pending.extend(scope.get_children())
                unknown = {s.get_name() for s in scope.get_symbols()
                           if s.is_global() and s.is_referenced() and s.get_name() not in allowed}
                self.assertFalse(unknown, f"{stem}:{scope.get_name()} has undefined globals: {unknown}")

    def test_scene_grouping_preserves_domain_and_pass_pairs(self):
        self.assertEqual(cache.source_scene("SDSD_indoor/pair10_2"), "SDSD_indoor/pair10")
        self.assertNotEqual(cache.source_scene("SDSD_indoor/a"), cache.source_scene("SDSD_outdoor/a"))
        self.assertEqual(len(cache.source_names(fixture_split(), expected=4)), 4)
        for name in ("BVI/a__10fps", "SDSD_indoor/../a", "SDSD_indoor/.."):
            with self.assertRaises(ValueError): cache.source_scene(name)

    def test_pass_leakage_and_duplicate_names_fail(self):
        split = fixture_split()
        split["SDSD_indoor"]["val"] = ["a_2"]
        with self.assertRaises(ValueError): cache.source_names(split, expected=4)
        split = fixture_split()
        split["SDSD_indoor"]["train"] = ["a", "a"]
        with self.assertRaises(ValueError): cache.source_names(split, expected=4)

    def test_scene_split_does_not_depend_on_order_or_pass_duplication(self):
        names = [f"SDSD_indoor/a{i}" for i in range(8)]
        split = train.make_scene_split(names)
        self.assertEqual(split, train.make_scene_split(list(reversed(names)) + [n + "_2" for n in names]))
        self.assertEqual(len(split["fit"]), 6)
        self.assertEqual(len(split["internal_holdout"]), 2)
        self.assertFalse(set(split["fit"]) & set(split["internal_holdout"]))

    def test_expected_base_identity_and_no_target_trained_base(self):
        args = {"tag": "e17_small_pretrained_s0", "seed": 0, "iters": 20000}
        cache.check_base_args(args, "pretrained", 16)
        for changed in ({**args, "tag": "e12_bvi_pretrained_s0"}, {**args, "heldout": "sdsd"}, {**args, "seed": 1}):
            with self.assertRaises(ValueError): cache.check_base_args(changed, "pretrained", 16)

    def test_fixed_budget_source_partition_and_fit_only_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "split.json"
            split = {key: {"root": "unused", "train": [f"a{i}" for i in range(68)],
                           "val": ["val"], "test": ["test"]}
                     for key in ("SDSD_indoor", "SDSD_outdoor")}
            path.write_text(json.dumps(split))
            names = cache.source_names(split)
            groups = train.make_scene_split(names)
            clips = [{"name": n, "scene": cache.source_scene(n)} for n in names]
            fit_frames = sum(r["scene"] in groups["fit"] for r in clips) * 16
            p = {"args": {"candidate_mode": "restored", "modes": list(train.MODES), "seed": 0,
                          "steps": 1000, "batch_frames": 8, "lr": 1e-3},
                 "cache_provenance": {"domain": "sdsd", "source_names": names, "split_sha256": cache.digest(path)},
                 "split": groups, "cache_clips": clips,
                 "candidate_backend": "CPU float32 for all families, source and target",
                 "normalization": {"fit_frames": fit_frames, "fit_patches": fit_frames * 1024,
                                   "source": "Fit scenes only", "direct_mse_normalizer": .01}}
            evaluate.validate_training_metadata(p, path)
            for key, value in (("steps", 999), ("lr", .01), ("candidate_mode", "target_selected")):
                bad = copy.deepcopy(p); bad["args"][key] = value
                with self.assertRaises(ValueError): evaluate.validate_training_metadata(bad, path)
            bad = copy.deepcopy(p); bad["normalization"]["fit_frames"] += 16
            with self.assertRaises(ValueError): evaluate.validate_training_metadata(bad, path)
            bad = copy.deepcopy(p); bad["cache_clips"][0]["scene"] = "BVI/target"
            with self.assertRaises(ValueError): evaluate.validate_training_metadata(bad, path)

    def test_cache_integrity_and_completeness(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            splits = {key: {"train": [f"a{i}" for i in range(68)], "val": ["val"], "test": ["test"]}
                      for key in ("SDSD_indoor", "SDSD_outdoor")}
            path = folder / "splits.json"; path.write_text(json.dumps(splits))
            names = cache.source_names(splits)
            provenance = {"domain": "sdsd", "source_names": names, "split_sha256": cache.digest(path),
                          "source_sha256": {}, "plan": [{"sequence_index": i} for i in range(136)]}
            for key in ("base_checkpoint_sha256", "base_state_sha256", "raft_weights_sha256", "raft_state_sha256"):
                provenance[key] = "a" * 64
            rows = []
            for i, name in enumerate(names):
                clip = folder / f"clip_{i:03d}.pt"; clip.write_bytes(b"fixture")
                rows.append({"name": name, "scene": cache.source_scene(name), "cache_path": clip.name,
                             "cache_sha256": cache.digest(clip)})
            cache.write_json(folder / "provenance.json", provenance)
            cache.write_json(folder / "cache_summary.json", {"n": 136, "per_sequence": rows,
                "oracle_computed": False, "base_state_unchanged": True, "raft_state_unchanged": True})
            self.assertEqual(len(train.cache_manifest(folder, path)[0]), 136)
            (folder / "clip_001.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Cache file"): train.cache_manifest(folder, path)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch required for tensor contract")
class SourceTensorTests(unittest.TestCase):
    def test_none_arm_supplies_four_zero_token_fields(self):
        import torch
        low = torch.ones(1, 3, 16, 8, 8)
        tokens = cache.backbone_tokens(None, low)
        self.assertEqual(len(tokens), 4)
        self.assertTrue(all(t.shape == (1, 2048, 1024) and not t.any() for t in tokens))


if __name__ == "__main__":
    unittest.main()
