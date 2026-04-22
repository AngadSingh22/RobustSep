from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robustsep_pkg.core.artifact_io import sha256_file
from robustsep_pkg.data import RobustSepDataset, ShardArrays, deterministic_split


def _write_shard(root: Path, shard_id: int, n: int = 2, include_alpha: bool = True) -> dict:
    out_dir = root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.full((n, 16, 16, 3), shard_id / 10.0, dtype=np.float16)
    lab = np.full((n, 16, 16, 3), shard_id, dtype=np.float16)
    cmyk = np.zeros((n, 16, 16, 4), dtype=np.float16)
    cmyk[..., 0] = 0.1 + shard_id
    cmyk_projected = cmyk.copy()
    arrays = {"rgb": rgb, "lab": lab, "cmyk": cmyk, "cmyk_projected": cmyk_projected}
    if include_alpha:
        alpha = np.full((n, 16, 16), 0.5, dtype=np.float16)
        arrays["alpha"] = alpha
    npz = out_dir / f"patches-{shard_id:05d}.npz"
    np.savez_compressed(npz, **arrays)
    jsonl = out_dir / f"patches-{shard_id:05d}.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "shard_index": i,
                "source_path": f"source-{shard_id}.png",
                "x": i,
                "y": shard_id,
                "structure": "flat",
                "color": "neutral",
                "stats": {"visible_alpha": 0.5},
                "crop_meta": {"cropped": False},
            }, sort_keys=True) + "\n")
    return {
        "npz": str(npz.relative_to(root)),
        "jsonl": str(jsonl.relative_to(root)),
        "count": n,
        "npz_sha256": sha256_file(npz),
        "jsonl_sha256": sha256_file(jsonl),
    }


class DatasetLoaderTests(unittest.TestCase):
    def test_shard_arrays_load_alpha_and_cmykogv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry = _write_shard(root, 0, n=2, include_alpha=True)
            arrays = ShardArrays(root / entry["npz"])
            self.assertEqual(arrays.rgb.shape, (2, 16, 16, 3))
            self.assertEqual(arrays.alpha.shape, (2, 16, 16))
            self.assertEqual(arrays.icc_cmyk.shape, (2, 16, 16, 4))
            self.assertEqual(arrays.cmykogv_baseline.shape, (2, 16, 16, 7))
            np.testing.assert_allclose(arrays.alpha, 0.5)
            np.testing.assert_allclose(arrays.cmykogv_baseline[..., :4], arrays.cmyk_baseline)
            np.testing.assert_allclose(arrays.cmykogv_baseline[..., 4:], 0.0)

    def test_dataset_reads_manifest_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shards = [_write_shard(root, i, n=2, include_alpha=False) for i in range(4)]
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            manifest = {
                "out_dir": "processed",
                "patches_written": 8,
                "shards": shards,
            }
            manifest_path = manifest_dir / "toy_run_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            split_map = deterministic_split(4, val_frac=0.25, test_frac=0.25, root_seed=11, scope="processed")
            ds = RobustSepDataset([manifest_path], root=root, split="train", val_frac=0.25, test_frac=0.25, root_seed=11)
            self.assertEqual(ds.num_shards, len(split_map["train"]))
            self.assertEqual(len(ds), 2 * len(split_map["train"]))
            first = ds[0]
            self.assertEqual(first.rgb.shape, (16, 16, 3))
            self.assertEqual(first.alpha.shape, (16, 16))
            self.assertEqual(first.cmykogv_baseline.shape, (16, 16, 7))
            self.assertEqual(first.record.structure, "flat")
            self.assertIn(str(manifest_path), ds.summary()["manifests"])

    def test_from_manifest_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shards = [_write_shard(root, 0, n=1)]
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            (manifest_dir / "toy_run_manifest.json").write_text(json.dumps({"out_dir": "processed", "shards": shards}), encoding="utf-8")
            ds = RobustSepDataset.from_manifest_dir(manifest_dir, root=root, val_frac=0.0, test_frac=0.0)
            self.assertEqual(len(ds), 1)


if __name__ == "__main__":
    unittest.main()
