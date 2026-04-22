from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robustsep_pkg.core.artifact_io import read_json, read_jsonl
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data import SurrogateShardWriterConfig, write_surrogate_training_shards
from robustsep_pkg.targets import TargetGenerationPipelineConfig, generate_target_records
from tests.test_target_generation_pipeline import _write_pipeline_fixture


class SurrogateShardWriterTests(unittest.TestCase):
    def test_writes_surrogate_training_shards_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = _write_pipeline_fixture(root, n=3)
            ppp = PPP.from_base()
            records = generate_target_records(
                manifest_path,
                ppp,
                root=root,
                config=TargetGenerationPipelineConfig(drift_samples_per_patch=1),
            )
            out_dir = root / "surrogate"
            summary = write_surrogate_training_shards(
                records,
                out_dir,
                ppp,
                config=SurrogateShardWriterConfig(shard_size=2, run_id="unit"),
            )
            manifest = read_json(summary.manifest_path)
            shard0 = manifest["shards"][0]
            rows = read_jsonl(shard0["jsonl"])
            with np.load(shard0["npz"]) as arrays:
                context_shape = arrays["cmykogv_context"].shape
                lab_shape = arrays["lab_center"].shape
                intent_shape = arrays["intent_raster"].shape
                drift_shape = arrays["drift_multipliers"].shape

        self.assertEqual(summary.total_examples, 3)
        self.assertEqual(summary.num_shards, 2)
        self.assertEqual(manifest["manifest_version"], "surrogate-shards-v1")
        self.assertEqual(manifest["ppp_hash"], ppp.hash)
        self.assertEqual(context_shape, (2, 32, 32, 7))
        self.assertEqual(lab_shape, (2, 16, 16, 3))
        self.assertEqual(intent_shape, (2, 4, 4, 3))
        self.assertEqual(drift_shape, (2, 7))
        self.assertIn("target_hash", rows[0])
        self.assertIn("drift_hash", rows[0])
        self.assertEqual(rows[0]["structure_token"], "flat")

    def test_rejects_invalid_shard_size(self) -> None:
        with self.assertRaises(ValueError):
            write_surrogate_training_shards([], Path("/tmp/unused"), PPP.from_base(), config=SurrogateShardWriterConfig(shard_size=0))


if __name__ == "__main__":
    unittest.main()
