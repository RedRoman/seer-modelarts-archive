"""Meaningful CPU tests for whole-pipeline color, storage firewall and gates."""
from pathlib import Path
import json
import tempfile
import unittest
import zipfile

import torch
import replay_native as replay
from restricted_source_reader import read_allowed_tensor


class NativeReplayTests(unittest.TestCase):
    def test_whole_pipeline_sees_native_and_exact_output(self):
        native = torch.tensor([.1, .2, .8]).view(1, 3, 1, 1, 1).expand(1, 3, 2, 4, 4)
        seen = []
        def pipeline(low):
            seen.append(low.clone())
            encoder = low[:, :1] * .5
            return (low * .6 + encoder).clamp(0, 1)
        rgb, direct = replay.capture_wrapper(pipeline, native)
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], native))
        self.assertTrue(torch.equal(rgb, direct[:, [2, 1, 0]]))
        self.assertFalse(torch.equal(rgb, pipeline(native[:, [2, 1, 0]])))

    def test_wrapper_does_not_mutate_low(self):
        native = torch.rand(1, 3, 2, 4, 4)
        before = native.clone()
        replay.capture_wrapper(lambda low: low.square(), native)
        self.assertTrue(torch.equal(native, before))

    def test_old_base_gate_passes_small_compatible_difference(self):
        low = torch.zeros(1, 3, 2, 4, 4)
        self.assertTrue(replay.check_old_base(low + 1e-6, low)["passed"])

    def test_old_base_gate_enforces_maximum(self):
        low = torch.zeros(1000)
        different = low.clone(); different[0] = 1.1e-4
        result = replay.check_old_base(different, low)
        self.assertLess(result["mean_abs"], replay.MEAN_ABS)
        self.assertFalse(result["passed"])

    def test_old_base_gate_enforces_mean(self):
        low = torch.zeros(1000)
        result = replay.check_old_base(low + 2e-5, low)
        self.assertLess(result["max_abs"], replay.MAX_ABS)
        self.assertFalse(result["passed"])

    def test_only_fixed_fit_names_allowed(self):
        fit = {"fit": [replay.SCENE], "internal_holdout": []}
        for index in replay.INDICES:
            row = {"scene": replay.SCENE, "name": replay.SCENE + ("_2" if index == 109 else "")}
            replay.validate_source_row(row, index, fit)
        with self.assertRaises(ValueError):
            replay.validate_source_row(row, 110, fit)
        with self.assertRaises(ValueError):
            replay.validate_source_row(row, 109, {"fit": [], "internal_holdout": [replay.SCENE]})

    def test_storage_reader_never_opens_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.pt"
            shape = (1, 3, 2, 4, 4)
            low, base = torch.rand(shape), torch.rand(shape)
            torch.save({"name": replay.SCENE, "scene": replay.SCENE,
                "tensors": {"lq": low, "base": base, "gt": torch.full(shape, -123.)}}, path)
            opened = []
            class Spy(zipfile.ZipFile):
                def read(self, name, *args, **kwargs):
                    opened.append(name)
                    return super().read(name, *args, **kwargs)
            for key, expected in (("lq", low), ("base", base)):
                opened.clear()
                _, _, actual, lineage = read_allowed_tensor(path, key, shape, archive_class=Spy)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(opened, lineage["archive_entries_read"])
                self.assertEqual(sum("/data/" in name for name in opened), 1)
                self.assertEqual(opened[-1], lineage["selected_storage_entry"])
            with self.assertRaises(ValueError):
                read_allowed_tensor(path, "gt", shape)

    def test_storage_rejects_shared_label_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.pt"; low = torch.rand(1, 3, 2, 4, 4)
            torch.save({"name": "x", "scene": "x", "tensors": {"lq": low, "gt": low}}, path)
            with self.assertRaises(ValueError):
                read_allowed_tensor(path, "lq", low.shape)

    def test_target_gate_binds_both_source_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sources = {}
            for arm in replay.SHAS:
                path = root / (arm + ".json")
                replay.write_new(path, {"passed": True, "arm": arm, "target_pixel_reads": 0,
                    "gt_storage_reads": 0, "protocol_sha256": "protocol"})
                sources[arm] = path
            gate = root / "authorization.json"
            data = {"authorized_stage": "target_bvi_val16", "protocol_sha256": "protocol",
                "source_parity_sha256": {arm: replay.digest(path) for arm, path in sources.items()}}
            replay.write_new(gate, data)
            replay.validate_authorization(gate, "protocol", sources)
            with self.assertRaises(ValueError):
                replay.validate_authorization(gate, "changed_protocol", sources)
            with self.assertRaises(ValueError):
                replay.validate_authorization(gate, "protocol", {"pretrained": sources["pretrained"]})
            sources["none"].write_text(sources["none"].read_text() + " ")
            with self.assertRaises(ValueError):
                replay.validate_authorization(gate, "protocol", sources)

    def test_write_new_preserves_existing_failed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.json"
            replay.write_new(path, {"passed": False})
            with self.assertRaises(FileExistsError):
                replay.write_new(path, {"passed": True})
            self.assertFalse(json.loads(path.read_text())["passed"])


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
