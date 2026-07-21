from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import compare_manifests


class CompareManifestsTest(unittest.TestCase):
    def test_independent_header_comparison(self) -> None:
        header = {
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]},
        }
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.safetensors"
            model.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(8))
            reference = compare_manifests.build_reference_manifest(root)
            tensors = []
            for name, metadata in sorted(reference.items()):
                tensors.append({"name": name, **metadata})
            engine = {"tensors": tensors, "total_tensor_bytes": 8}
            report = compare_manifests.compare_manifests(reference, engine, "a" * 40)
            self.assertEqual(report.status, "ok")
            self.assertEqual(report.tensor_count, 2)
            self.assertEqual(report.mismatch_count, 0)

            engine["tensors"][0]["byte_offset"] += 1
            mismatch = compare_manifests.compare_manifests(reference, engine)
            self.assertEqual(mismatch.status, "mismatch")
            self.assertEqual(mismatch.mismatch_count, 1)

    def test_duplicate_json_keys_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(compare_manifests.ManifestError):
                compare_manifests.load_json(path)


if __name__ == "__main__":
    unittest.main()

