"""Meaningful source-storage firewall and float-preservation checks."""
import io
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

import torch

from cache_hvi_source import (MetadataOnlyUnpickler, output_payload,
                              predict_float_base, read_lq_only, tensor_digest)


class FloatHalfModel(torch.nn.Module):
    def forward(self, x):
        return x * .5


class SourceFirewallTests(unittest.TestCase):
    def fixture(self, root, *, with_other=True, alias=False):
        # A noncontiguous THWC->BCTHW view exercises exact storage strides.
        low = torch.arange(3 * 2 * 8 * 8, dtype=torch.float32).reshape(2, 8, 8, 3).div_(1000).permute(3, 0, 1, 2).unsqueeze(0)
        tensors = {"lq": low}
        if with_other:
            tensors["gt"] = low if alias else torch.full_like(low, float("nan"))
        path = root / "fixture.pt"
        torch.save({"name": "source/example", "scene": "source/example", "tensors": tensors}, path)
        return path, low

    def test_exact_strided_float_lq_and_unopened_other_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.fixture(Path(directory))
            opened = []
            class RejectOtherStorage(zipfile.ZipFile):
                def read(self, name, *args, **kwargs):
                    opened.append(name)
                    if "/data/" in name and not name.endswith("/data/0"):
                        raise AssertionError("Forbidden non-LQ storage opened")
                    return super().read(name, *args, **kwargs)
            name, scene, low, lineage = read_lq_only(path, expected.shape, archive_class=RejectOtherStorage)
            self.assertTrue(torch.equal(low, expected))
            self.assertTrue(torch.isfinite(low).all())
            self.assertEqual(lineage["lq_tensor_sha256"], tensor_digest(expected))
            self.assertEqual(sum("/data/" in p for p in opened), 1)
            self.assertEqual(lineage["archive_entries_read"], opened)

    def test_fixture_without_any_gt_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.fixture(Path(directory), with_other=False)
            _, _, low, _ = read_lq_only(path, expected.shape)
            self.assertTrue(torch.equal(low, expected))

    def test_shared_storage_rejected_before_pixel_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self.fixture(Path(directory), alias=True)
            class NoStorageReads(zipfile.ZipFile):
                def read(self, name, *args, **kwargs):
                    if "/data/" in name:
                        raise AssertionError("Storage opened before sharing guard")
                    return super().read(name, *args, **kwargs)
            with self.assertRaisesRegex(ValueError, "shared"):
                read_lq_only(path, expected.shape, archive_class=NoStorageReads)

    def test_nonallowlisted_pickle_global_refused(self):
        data = pickle.dumps(Path("/unused"), protocol=2)
        with self.assertRaisesRegex(ValueError, "Disallowed"):
            MetadataOnlyUnpickler(io.BytesIO(data)).load()

    def test_float_forward_and_output_schema_no_reencoding(self):
        low = torch.full((1, 3, 2, 8, 8), .1234567, dtype=torch.float32)
        original = low.clone()
        base = predict_float_base(FloatHalfModel(), low, torch.device("cpu"))
        self.assertTrue(torch.equal(base, low * .5))
        self.assertTrue(torch.equal(low, original))
        self.assertFalse(torch.equal(base, (base * 255).round() / 255))
        payload = output_payload("source/example", "source/example", base, {"lq_tensor_sha256": tensor_digest(low)})
        self.assertEqual(set(payload), {"name", "scene", "base", "lineage"})
        self.assertEqual(payload["lineage"]["base_tensor_sha256"], tensor_digest(base))

    def test_input_validation_rejects_uint8_and_nonfinite(self):
        with self.assertRaises(ValueError):
            predict_float_base(FloatHalfModel(), torch.ones(1, 3, 2, 8, 8, dtype=torch.uint8), "cpu")
        with self.assertRaises(ValueError):
            predict_float_base(FloatHalfModel(), torch.full((1, 3, 2, 8, 8), float("nan")), "cpu")


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
