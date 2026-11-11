"""CPU-only reconstruction/provenance tests; no video, metrics or training runs."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "tools"))
spec = importlib.util.spec_from_file_location("_eval_observation_v2_test", CODE / "tools/eval_observation_v2.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)
from train_source_selector import make_scene_split

try:
    import torch
except ImportError:
    torch = None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@unittest.skipIf(torch is None, "Torch required; run these guards on the remote CPU")
class EvaluationProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.graph = evaluator.modules()["trajectory"]
        split = {domain: {"train": [f"scene{i}{suffix}" for i in range(count) for suffix in ("", "_2")],
                          "val": ["heldout_val"], "test": ["heldout_test"]}
                 for domain, count in (("SDSD_indoor", 23), ("SDSD_outdoor", 45))}
        cls.splits = cls.root / "splits.json"
        write_json(cls.splits, split)
        names = [f"{d}/{n}" for d in split for n in split[d]["train"]]
        origin = {"domain": "sdsd", "source_names": names,
                  "split_sha256": evaluator.digest(cls.splits),
                  "base_checkpoint_sha256": "a" * 64,
                  "source_sha256": evaluator.source_hashes()}
        control = evaluator.modules()["control"]
        norm = {"feature_mean": [0.] * 19, "feature_std": [1.] * 19,
                "direct_mse_normalizer": .01}
        cls.meta = {"schema_version": 1, "domain": "sdsd", "n": 136,
                    "oracle_computed": False, "target_frames_read": False,
                    "source_cache_provenance": origin,
                    "per_sequence": [{"name": n} for n in names],
                    "split": make_scene_split(names),
                    "variants": {k: list(v) for k, v in evaluator.VARIANTS.items()},
                    "normalization": {c: copy.deepcopy(norm) for c in evaluator.CANDIDATES},
                    "candidate_recipes": {c: control.observation_control_config(c, 16) for c in evaluator.CANDIDATES},
                    "source_sha256": evaluator.source_hashes(*evaluator.EXTRA_SOURCES)}
        cls.manifest = cls.root / "prepared_manifest.json"
        write_json(cls.manifest, cls.meta)
        cls.manifest_sha = evaluator.digest(cls.manifest)
        cls.directories = []
        for variant, (candidate, mode) in evaluator.VARIANTS.items():
            for seed in (0, 1, 2):
                directory = cls.root / f"{variant}_s{seed}"
                directory.mkdir()
                plan = evaluator.make_plan(names, cls.meta["split"], seed)
                write_json(directory / "sample_plan.json", plan)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model = cls.graph.TrajectoryReconstruction(mode=mode)
                initial = copy.deepcopy(model.state_dict())
                torch.save({"model": initial}, directory / "initial_state.pt")
                p = {"variant": variant, "seed": seed, "candidate": candidate,
                     "args": {"steps": 1000, "source_crop": 128, "loss_frames": 8,
                              "lr": .001, "device": "cpu", "clips_per_update": 1, "clip_len": 16},
                     "prepared_manifest_sha256": cls.manifest_sha,
                     "normalization": copy.deepcopy(norm),
                     "candidate_recipe": cls.meta["candidate_recipes"][candidate],
                     "source_names": names, "source_scene_split": cls.meta["split"],
                     "source_split_sha256": origin["split_sha256"],
                     "base_checkpoint_sha256": origin["base_checkpoint_sha256"],
                     "source_sha256": cls.meta["source_sha256"],
                     "initial_state_sha256": evaluator.state_digest(initial),
                     "sample_plan_sha256": evaluator.digest(directory / "sample_plan.json"),
                     "model_config": model.get_config()}
                record = {"steps": 1000, "model": initial,
                          "model_state_sha256": evaluator.state_digest(initial), "provenance": p}
                torch.save(record, directory / "final.pt")
                write_json(directory / "final.sha256.json",
                           {"checkpoint_sha256": evaluator.digest(directory / "final.pt")})
                cls.directories.append(directory)
        cls.files = {p: p.read_bytes() for d in cls.directories for p in d.iterdir()}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tearDown(self):
        for path, contents in self.files.items():
            if path.read_bytes() != contents:
                path.write_bytes(contents)

    def load(self, directories=None):
        return evaluator.load_heads(self.directories if directories is None else directories,
                                    self.meta, self.manifest_sha, self.graph)

    def mutate_record(self, change, directory=None, update_file_sha=True):
        directory = directory or self.directories[0]
        path = directory / "final.pt"
        record = torch.load(path, weights_only=True, map_location="cpu")
        change(record)
        torch.save(record, path)
        if update_file_sha:
            write_json(directory / "final.sha256.json", {"checkpoint_sha256": evaluator.digest(path)})

    def test_source_manifest_complete_and_51_17_scene_split(self):
        origin, sha = evaluator.validate_preparation(self.meta, self.manifest, self.splits)
        self.assertEqual(origin, self.meta["source_cache_provenance"])
        self.assertEqual(sha, self.manifest_sha)
        self.assertEqual(len(self.meta["split"]["fit"]), 51)
        self.assertEqual(len(self.meta["split"]["internal_holdout"]), 17)

    def test_source_hash_manifest_missing_changed_or_escaping_fails_closed(self):
        for owner in ("source_sha256", "source_cache_provenance"):
            for change in ("missing", "changed", "escape"):
                meta = copy.deepcopy(self.meta)
                mapping = meta[owner] if owner == "source_sha256" else meta[owner]["source_sha256"]
                key = next(iter(mapping))
                if change == "missing":
                    del mapping[key]
                elif change == "changed":
                    mapping[key] = "0" * 64
                else:
                    mapping["../../outside.py"] = "0" * 64
                with self.subTest(owner=owner, change=change), self.assertRaises(ValueError):
                    evaluator.validate_preparation(meta, self.manifest, self.splits)

    def test_target_or_oracle_markers_and_source_names_split_are_rejected(self):
        mutations = [lambda m: m.update(target_frames_read=True),
                     lambda m: m.update(oracle_computed=True),
                     lambda m: m.update(n=135),
                     lambda m: m["per_sequence"].reverse(),
                     lambda m: m["split"]["fit"].append(m["split"]["internal_holdout"][0]),
                     lambda m: m["source_cache_provenance"].update(split_sha256="0" * 64)]
        for change in mutations:
            meta = copy.deepcopy(self.meta)
            change(meta)
            with self.assertRaises(ValueError):
                evaluator.validate_preparation(meta, self.manifest, self.splits)

    def test_all_21_heads_load_with_matching_seed_plan_and_cpu_frozen_states(self):
        models, records, hashes, fingerprints = self.load()
        self.assertEqual(len(models), 21)
        self.assertEqual(set(models), {f"{v}_s{s}" for v in evaluator.VARIANTS for s in range(3)})
        for key, model in models.items():
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad and p.device.type == "cpu" for p in model.parameters()))
            self.assertEqual(evaluator.state_digest(model.state_dict()), fingerprints[key])
            self.assertEqual(len(hashes[key]), 64)
            self.assertEqual(model.get_config(), records[key]["model_config"])

    def test_missing_duplicate_and_unexpected_seed_rejected(self):
        for directories in (self.directories[:-1], self.directories + [self.directories[0]]):
            with self.assertRaises(ValueError):
                self.load(directories)
        self.mutate_record(lambda r: r["provenance"].update(seed=3))
        with self.assertRaises(ValueError):
            self.load()

    def test_candidate_mode_scope_and_fixed_budget_are_checked(self):
        changes = [lambda p: p.update(candidate="same_frame_detail"),
                   lambda p: p["candidate_recipe"].update(gt_input=True),
                   lambda p: p["model_config"].update(mode="conditioned"),
                   lambda p: p["normalization"]["feature_std"].__setitem__(0, 2.),
                   lambda p: p.update(prepared_manifest_sha256="b" * 64),
                   lambda p: p.update(base_checkpoint_sha256="b" * 64),
                   lambda p: p["source_names"].reverse()]
        changes += [(lambda field, value: lambda p: p["args"].update({field: value}))(field, value)
                    for field, value in (("steps", 999), ("source_crop", 256), ("loss_frames", 16),
                                         ("lr", .01), ("device", "npu"), ("clips_per_update", 8), ("clip_len", 8))]
        for change in changes:
            self.mutate_record(lambda r: change(r["provenance"]))
            with self.assertRaises(ValueError):
                self.load()
            self.tearDown()

    def test_checkpoint_file_and_tensor_fingerprints_are_both_checked(self):
        self.mutate_record(lambda r: r.update(irrelevant_extra=True), update_file_sha=False)
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.load()
        self.tearDown()
        self.mutate_record(lambda r: next(iter(r["model"].values())).add_(1))
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.load()

    def test_common_forged_sample_plan_cannot_pass_cross_variant_matching(self):
        # Every variant claims the same corrupted plan for seed0. Cross-variant
        # SHA agreement alone would pass; reconstruction of the declared plan must fail.
        for directory in self.directories:
            if not directory.name.endswith("_s0"):
                continue
            path = directory / "sample_plan.json"
            plan = json.loads(path.read_text())
            plan["batches"][0]["frames"] = [0] * 8
            write_json(path, plan)
            self.mutate_record(lambda r: r["provenance"].update(sample_plan_sha256=evaluator.digest(path)), directory)
        with self.assertRaisesRegex(ValueError, "sample plan"):
            self.load()

    def test_common_forged_initialization_cannot_pass_cross_variant_matching(self):
        for directory in self.directories:
            if not directory.name.endswith("_s0"):
                continue
            path = directory / "initial_state.pt"
            record = torch.load(path, weights_only=True, map_location="cpu")
            next(iter(record["model"].values())).add_(.25)
            torch.save(record, path)
            sha = evaluator.state_digest(record["model"])
            self.mutate_record(lambda r: r["provenance"].update(initial_state_sha256=sha), directory)
        with self.assertRaisesRegex(ValueError, "initialization"):
            self.load()

    def test_head_source_manifest_must_equal_preparation(self):
        self.mutate_record(lambda r: r["provenance"]["source_sha256"].pop("models/trajectory_reconstruction.py"))
        with self.assertRaisesRegex(ValueError, "source manifest"):
            self.load()

    def test_hash_formats_are_distinct_and_base_raft_call_sites_use_torch_format(self):
        source = (CODE / "tools/train_method_pilot.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "state_digest")
        namespace = {"hashlib": hashlib, "torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "state_digest_only", "exec"), namespace)
        state = {"scalar": torch.tensor(.5), "weight": torch.arange(6.).reshape(2, 3)}
        actual = namespace["state_digest"](state)
        expected = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            expected.update(name.encode())
            expected.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            expected.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.assertEqual(actual, expected.hexdigest())
        self.assertNotEqual(actual, evaluator.state_digest(state))
        tree = ast.parse((CODE / "tools/eval_observation_v2.py").read_text())
        calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for variable in ("base_model", "raft"):
            self.assertGreaterEqual(calls.count(f"base_state_digest({variable}.state_dict())"), 2)
            self.assertNotIn(f"state_digest({variable}.state_dict())", calls)


if __name__ == "__main__":
    unittest.main()
