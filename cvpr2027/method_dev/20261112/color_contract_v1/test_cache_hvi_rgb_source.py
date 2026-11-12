"""Selective storage firewall and explicit native-BGR/canonical-RGB boundary tests."""
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import torch

from cache_hvi_rgb_source import (BOUNDARY, LEGACY_SHA, SOURCE_SHA, canonical_rgb,
                                  digest, load_legacy, read_rgb_lq, validate_evidence, write_new)


LEGACY = Path(__file__).parent / "dependencies/cache_hvi_source.py"
if not LEGACY.exists():
    LEGACY = Path(__file__).parent.parent / "hvi_source_anchor_v1/cache_hvi_source.py"


class ChannelSensitiveMock(torch.nn.Module):
    def forward(self, value):
        scales = torch.tensor([.5, .7, .9]).reshape(1, 3, 1, 1)
        return value * scales


class RGBExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = load_legacy(LEGACY)

    def fixture(self, directory, *, with_gt=True, alias=False):
        native = torch.arange(3 * 2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8, 3).div(1000).permute(3, 0, 1, 2).unsqueeze(0)
        tensors = {"lq": native}
        if with_gt:
            tensors["gt"] = native if alias else torch.full_like(native, float("nan"))
        path = Path(directory) / "fixture.pt"
        torch.save({"name": "SDSD/example", "scene": "SDSD/example", "tensors": tensors}, path)
        return path, native

    def test_exact_strided_native_to_rgb_and_no_other_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path, native = self.fixture(directory)
            entries = []
            class ForbidGT(zipfile.ZipFile):
                def read(self, name, *args, **kwargs):
                    entries.append(name)
                    if "/data/" in name and not name.endswith("/data/0"):
                        raise AssertionError("Non-LQ storage read")
                    return super().read(name, *args, **kwargs)
            _, _, rgb, lineage = read_rgb_lq(path, self.legacy, expected_shape=native.shape, archive_class=ForbidGT)
            self.assertTrue(torch.equal(rgb, native[:, [2, 1, 0]]))
            self.assertEqual(lineage["native_lq_tensor_sha256"], self.legacy.tensor_digest(native))
            self.assertEqual(lineage["lq_tensor_sha256"], lineage["native_lq_tensor_sha256"])
            self.assertEqual(lineage["rgb_lq_tensor_sha256"], self.legacy.tensor_digest(rgb))
            self.assertNotEqual(lineage["native_lq_tensor_sha256"], lineage["rgb_lq_tensor_sha256"])
            self.assertEqual(lineage["channel_boundary"], BOUNDARY)
            self.assertEqual(lineage["archive_entries_read"], entries)
            self.assertEqual(sum("/data/" in name for name in entries), 1)

    def test_no_gt_fixture_mock_hvi_stays_rgb_float_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, native = self.fixture(directory, with_gt=False)
            original = native.clone()
            name, scene, rgb, lineage = read_rgb_lq(path, self.legacy, expected_shape=native.shape)
            model = ChannelSensitiveMock()
            base = self.legacy.predict_float_base(model, rgb, "cpu")
            expected = rgb * torch.tensor([.5, .7, .9]).reshape(1, 3, 1, 1, 1)
            self.assertTrue(torch.equal(base, expected))
            self.assertFalse(torch.equal(base, expected[:, [2, 1, 0]]))
            self.assertTrue(torch.equal(native, original))
            self.assertFalse(torch.equal(base, (base * 255).round() / 255))
            payload = self.legacy.output_payload(name, scene, base, lineage)
            self.assertEqual(set(payload), {"name", "scene", "base", "lineage"})
            self.assertEqual(payload["lineage"]["output_color"], "RGB")

    def test_shared_gt_storage_rejected_before_any_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path, native = self.fixture(directory, alias=True)
            class ForbidAllStorage(zipfile.ZipFile):
                def read(self, name, *args, **kwargs):
                    if "/data/" in name:
                        raise AssertionError("Storage opened before alias check")
                    return super().read(name, *args, **kwargs)
            with self.assertRaisesRegex(ValueError, "shared"):
                read_rgb_lq(path, self.legacy, expected_shape=native.shape, archive_class=ForbidAllStorage)

    def test_pickle_allowlist_remains_enforced(self):
        with self.assertRaisesRegex(ValueError, "Disallowed"):
            self.legacy.MetadataOnlyUnpickler(io.BytesIO(pickle.dumps(Path("/unused"), protocol=2))).load()

    def test_legacy_code_hash_enforced(self):
        self.assertEqual(digest(LEGACY), LEGACY_SHA)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.py"
            path.write_text(LEGACY.read_text() + "\n# changed\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_legacy(path)

    def test_invalid_tensors_rejected(self):
        for value in (torch.ones(1, 3, 2, 8, 8, dtype=torch.uint8),
                      torch.full((1, 3, 2, 8, 8), float("nan")),
                      torch.full((1, 3, 2, 8, 8), 1.01), torch.ones(1, 4, 2, 8, 8)):
            with self.assertRaises(ValueError):
                canonical_rgb(value)

    def test_evidence_gate_and_proof_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = Path(directory) / "primary.json"
            proof.write_text('{"native_source_color":"BGR"}\n')
            receipt = Path(directory) / "closure.json"
            value = {"status": "pending", "native_source_color": "BGR", "source_provenance_sha256": SOURCE_SHA,
                     "proof_files": [{"path": str(proof), "sha256": digest(proof)}]}
            receipt.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "not closed"):
                validate_evidence(receipt)
            value["status"] = "closed"
            receipt.write_text(json.dumps(value))
            self.assertEqual(validate_evidence(receipt)["bgr_evidence_status"], "closed")
            proof.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_evidence(receipt)

    def test_write_new_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.json"
            write_new(path, {"new": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"overwrite": True})


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
