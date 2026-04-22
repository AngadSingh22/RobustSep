from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robustsep_pkg.cli import main
from tests.test_target_generation_pipeline import _write_pipeline_fixture


class RobustSepCliTests(unittest.TestCase):
    def test_generate_targets_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_manifest = _write_pipeline_fixture(root, n=2)
            out = root / "targets.jsonl"
            summary = root / "summary.json"
            rc = main(
                [
                    "generate-targets",
                    "--split-manifest",
                    str(split_manifest),
                    "--root",
                    str(root),
                    "--out",
                    str(out),
                    "--summary-out",
                    str(summary),
                    "--max-records",
                    "1",
                    "--drift-samples-per-patch",
                    "1",
                    "--stage1-steps",
                    "1",
                    "--stage2-steps",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            self.assertTrue(summary.exists())
            self.assertEqual(len(out.read_text().splitlines()), 1)

    def test_write_surrogate_shards_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_manifest = _write_pipeline_fixture(root, n=2)
            out_dir = root / "surrogate"
            rc = main(
                [
                    "write-surrogate-shards",
                    "--split-manifest",
                    str(split_manifest),
                    "--root",
                    str(root),
                    "--out-dir",
                    str(out_dir),
                    "--max-records",
                    "1",
                    "--drift-samples-per-patch",
                    "1",
                    "--shard-size",
                    "1",
                    "--stage1-steps",
                    "1",
                    "--stage2-steps",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "surrogate_training_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
