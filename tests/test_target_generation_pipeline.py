from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robustsep_pkg.core.artifact_io import read_jsonl, sha256_file
from robustsep_pkg.models.conditioning.ppp import PPP, feasibility_violations
from robustsep_pkg.targets import (
    TargetGenerationPipelineConfig,
    generate_target_records,
    load_split_manifest,
    write_target_records_jsonl,
)


def _write_pipeline_fixture(root: Path, *, n: int = 2, alpha_value: float = 0.5) -> Path:
    out_dir = root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.full((n, 16, 16, 3), 0.45, dtype=np.float32)
    lab = np.zeros((n, 16, 16, 3), dtype=np.float32)
    lab[..., 0] = 50.0
    cmyk = np.full((n, 16, 16, 4), 0.12, dtype=np.float32)
    alpha = np.full((n, 16, 16), alpha_value, dtype=np.float32)
    npz = out_dir / "patches-00000.npz"
    np.savez_compressed(npz, rgb=rgb, lab=lab, cmyk=cmyk, cmyk_projected=cmyk, alpha=alpha)

    jsonl = out_dir / "patches-00000.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "shard_index": i,
                        "source_path": f"image-{i}.png",
                        "x": i * 16,
                        "y": 0,
                        "structure": "flat",
                        "color": "neutral",
                        "stats": {},
                        "crop_meta": {},
                    }
                )
                + "\n"
            )

    manifest = {
        "split_manifest_version": "1.1",
        "root_seed": 123,
        "batch_size": 8,
        "drop_last": False,
        "total_patches": n,
        "total_shards": 1,
        "alpha_policy": "passthrough",
        "source_weight_policy": {
            "algorithm": "A-Res (Vitter 1985)",
            "default_weight": 0.0,
            "raw_weights": {"robustsep": 1.0},
            "normalized_fractions": {"robustsep": 1.0},
        },
        "families": [
            {
                "name": "robustsep",
                "split": "train",
                "num_shards": 1,
                "total_patches": n,
                "shards": [
                    {
                        "npz": str(npz.relative_to(root)),
                        "jsonl": str(jsonl.relative_to(root)),
                        "count": n,
                        "npz_sha256": sha256_file(npz),
                        "jsonl_sha256": sha256_file(jsonl),
                    }
                ],
            }
        ],
    }
    manifest_path = root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class TargetGenerationPipelineTests(unittest.TestCase):
    def test_generates_targets_and_surrogate_examples_from_v11_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = _write_pipeline_fixture(root, n=2, alpha_value=0.5)
            ppp = PPP.from_base()
            config = TargetGenerationPipelineConfig(
                drift_samples_per_patch=2,
                recompute_structure=True,
                lambda_value=0.9,
            )
            records = generate_target_records(manifest_path, ppp, root=root, config=config)

        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first.alpha_policy, "passthrough")
        self.assertEqual(first.target_result.target_cmykogv.shape, (16, 16, 7))
        self.assertFalse(any(feasibility_violations(first.target_result.target_cmykogv, ppp, lab_ref=first.enriched_sample.lab).values()))
        self.assertEqual(first.enriched_sample.intent_raster.shape, (4, 4, 3))
        self.assertAlmostEqual(float(first.enriched_sample.alpha_effective.mean()), 0.5, places=6)
        self.assertEqual(len(first.surrogate_examples), 2)
        self.assertEqual(first.surrogate_examples[0].cmykogv_context.shape, (32, 32, 7))
        self.assertEqual(first.surrogate_examples[0].lambda_value, 0.9)
        manifest_record = first.to_manifest_dict()
        self.assertEqual(manifest_record["family"], "robustsep")
        self.assertEqual(manifest_record["target"]["output_shape"], (16, 16, 7))
        self.assertEqual(manifest_record["surrogate_examples"][0]["intent_raster_shape"], (4, 4, 3))

    def test_write_target_records_jsonl_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = _write_pipeline_fixture(root, n=1)
            ppp = PPP.from_base()
            config = TargetGenerationPipelineConfig(drift_samples_per_patch=1)
            out_a = root / "targets-a.jsonl"
            out_b = root / "targets-b.jsonl"

            summary_a = write_target_records_jsonl(manifest_path, ppp, out_a, root=root, config=config)
            summary_b = write_target_records_jsonl(manifest_path, ppp, out_b, root=root, config=config)
            rows_a = read_jsonl(out_a)
            rows_b = read_jsonl(out_b)

        self.assertEqual(summary_a.records_written, 1)
        self.assertEqual(summary_b.records_written, 1)
        self.assertEqual(rows_a[0]["target"]["target_hash"], rows_b[0]["target"]["target_hash"])
        self.assertEqual(rows_a[0]["surrogate_examples"][0]["drift_hash"], rows_b[0]["surrogate_examples"][0]["drift_hash"])
        self.assertEqual(summary_a.alpha_policy, "passthrough")
        self.assertEqual(summary_a.split_manifest_version, "1.1")

    def test_rejects_non_v11_manifest(self) -> None:
        with self.assertRaises(ValueError):
            load_split_manifest(
                {
                    "split_manifest_version": "1.0",
                    "families": [],
                    "alpha_policy": "ones",
                    "source_weight_policy": {},
                }
            )


if __name__ == "__main__":
    unittest.main()
