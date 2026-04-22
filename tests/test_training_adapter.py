"""tests/test_training_adapter.py — Training adapter, batching, and source weighting."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robustsep_pkg.core.artifact_io import sha256_file
from robustsep_pkg.data import (
    RobustSepDataset,
    FamilyDataset,
    TrainingAdapter,
    ShardBatch,
    SourceWeightPolicy,
    iter_batches,
    weighted_shard_schedule,
)
from robustsep_pkg.data.shard_reader import ShardArrays, ShardReader
from robustsep_pkg.data.shard_record import ShardEntry


# ---------------------------------------------------------------------------
# Shared shard fixture
# ---------------------------------------------------------------------------

def _write_shard(root: Path, shard_id: int, n: int = 4) -> dict:
    out_dir = root / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.full((n, 16, 16, 3), shard_id / 10.0, dtype=np.float32)
    lab = np.zeros((n, 16, 16, 3), dtype=np.float32)
    lab[..., 0] = 50.0
    cmyk = np.zeros((n, 16, 16, 4), dtype=np.float32)
    cmyk[..., 0] = 0.1 * (shard_id + 1)
    npz = out_dir / f"patches-{shard_id:05d}.npz"
    np.savez_compressed(npz, rgb=rgb, lab=lab, cmyk=cmyk, cmyk_projected=cmyk)
    jsonl = out_dir / f"patches-{shard_id:05d}.jsonl"
    with jsonl.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "shard_index": i, "source_path": f"img-{shard_id}.png",
                "x": i * 16, "y": 0, "structure": "flat", "color": "neutral",
                "stats": {}, "crop_meta": {},
            }) + "\n")
    return {
        "npz": str(npz.relative_to(root)),
        "jsonl": str(jsonl.relative_to(root)),
        "count": n,
        "npz_sha256": sha256_file(npz),
        "jsonl_sha256": sha256_file(jsonl),
    }


def _write_manifest(root: Path, shards: list[dict], family: str = "robustsep") -> Path:
    m = {
        "run_id": f"test-{family}", "created_unix": 0.0,
        "out_dir": f"processed/{family}",
        "shards": shards, "manifest_hash": "dummy",
    }
    p = root / f"{family}_run_manifest.json"
    p.write_text(json.dumps(m))
    return p


def _make_dataset(root: Path, n_shards: int = 6, n_patches: int = 4,
                  family: str = "robustsep", split: str = "train") -> RobustSepDataset:
    shards = [_write_shard(root, i, n=n_patches) for i in range(n_shards)]
    mp = _write_manifest(root, shards, family=family)
    return RobustSepDataset([mp], root=root, split=split,
                             val_frac=0.17, test_frac=0.17)


# ---------------------------------------------------------------------------
# Batching tests
# ---------------------------------------------------------------------------

class BatchingTests(unittest.TestCase):

    def _make_arrays_records(self, n: int = 8):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry_d = _write_shard(root, 0, n=n)
            npz_path = root / entry_d["npz"]
            arrays = ShardArrays(npz_path)
        jsonl_data = [
            type("R", (), {"shard_index": i, "source_path": "x", "x": 0, "y": 0,
                           "structure": "flat", "color": "neutral",
                           "stats": {}, "crop_meta": {}, "extra": {}})()
            for i in range(n)
        ]
        from robustsep_pkg.data.shard_record import ShardRecord
        records = [ShardRecord(shard_index=i, source_path="x", x=0, y=0,
                               structure="flat", color="neutral") for i in range(n)]
        return arrays, records

    def test_iter_batches_full_batches(self):
        """8 patches, batch_size=4 → exactly 2 full batches."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            n = 8
            entry_d = _write_shard(root, 0, n=n)
            from robustsep_pkg.data.shard_record import ShardRecord
            arrays = ShardArrays(root / entry_d["npz"])
            records = [ShardRecord(shard_index=i, source_path="x", x=0, y=0,
                                   structure="flat", color="neutral") for i in range(n)]
            batches = list(iter_batches(arrays, records, batch_size=4))
        self.assertEqual(len(batches), 2)
        for b in batches:
            self.assertEqual(len(b), 4)
            self.assertEqual(b.rgb.shape, (4, 16, 16, 3))
            self.assertEqual(b.cmykogv_baseline.shape, (4, 16, 16, 7))

    def test_iter_batches_last_partial(self):
        """9 patches, batch_size=4 → 2 full batches + 1 partial."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            from robustsep_pkg.data.shard_record import ShardRecord
            entry_d = _write_shard(root, 0, n=9)
            arrays = ShardArrays(root / entry_d["npz"])
            records = [ShardRecord(shard_index=i, source_path="x", x=0, y=0,
                                   structure="flat", color="neutral") for i in range(9)]
            batches = list(iter_batches(arrays, records, batch_size=4))
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(batches[-1]), 1)

    def test_iter_batches_drop_last(self):
        """drop_last=True discards the partial batch."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            from robustsep_pkg.data.shard_record import ShardRecord
            entry_d = _write_shard(root, 0, n=9)
            arrays = ShardArrays(root / entry_d["npz"])
            records = [ShardRecord(shard_index=i, source_path="x", x=0, y=0,
                                   structure="flat", color="neutral") for i in range(9)]
            batches = list(iter_batches(arrays, records, batch_size=4, drop_last=True))
        self.assertEqual(len(batches), 2)
        for b in batches:
            self.assertEqual(len(b), 4)

    def test_iter_batches_batch_index_sequential(self):
        """batch_index is 0-based and sequential."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            from robustsep_pkg.data.shard_record import ShardRecord
            entry_d = _write_shard(root, 0, n=8)
            arrays = ShardArrays(root / entry_d["npz"])
            records = [ShardRecord(shard_index=i, source_path="x", x=0, y=0,
                                   structure="flat", color="neutral") for i in range(8)]
            idxs = [b.batch_index for b in iter_batches(arrays, records, batch_size=4)]
        self.assertEqual(idxs, [0, 1])

    def test_shard_batch_tensors_keys(self):
        """ShardBatch.tensors() returns all 6 expected keys."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            from robustsep_pkg.data.shard_record import ShardRecord
            entry_d = _write_shard(root, 0, n=4)
            arrays = ShardArrays(root / entry_d["npz"])
            records = [ShardRecord(shard_index=i, source_path="x", x=0, y=0,
                                   structure="flat", color="neutral") for i in range(4)]
            batch = next(iter_batches(arrays, records, batch_size=4))
        keys = set(batch.tensors().keys())
        self.assertEqual(keys, {"rgb", "alpha", "lab", "icc_cmyk", "cmyk_baseline", "cmykogv_baseline"})


# ---------------------------------------------------------------------------
# Source weighting tests
# ---------------------------------------------------------------------------

class SourceWeightingTests(unittest.TestCase):

    def _make_readers(self, n: int, root: Path, family_prefix: str = "f") -> list[ShardReader]:
        readers = []
        out_dir = root / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            entry_d = _write_shard(root, i, n=2)
            entry = ShardEntry(
                npz=entry_d["npz"], jsonl=entry_d["jsonl"], count=2,
                npz_sha256=entry_d["npz_sha256"], jsonl_sha256=entry_d["jsonl_sha256"],
            )
            readers.append(ShardReader(entry, root=root))
        return readers

    def test_equal_weights_all_shards_present(self):
        """Equal weights → all shards appear in the schedule."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readers_a = self._make_readers(3, root)
            readers_b = self._make_readers(3, root, "b")
            policy = SourceWeightPolicy(weights={"A": 1.0, "B": 1.0})
            schedule = weighted_shard_schedule({"A": readers_a, "B": readers_b}, policy, epoch=0)
        self.assertEqual(len(schedule), 6)

    def test_deterministic_same_epoch(self):
        """Same epoch → identical shard order."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readers = self._make_readers(5, root)
            policy = SourceWeightPolicy()
            s1 = weighted_shard_schedule({"robustsep": readers}, policy, epoch=3)
            s2 = weighted_shard_schedule({"robustsep": readers}, policy, epoch=3)
        self.assertEqual([id(r) for r in s1], [id(r) for r in s2])

    def test_different_epochs_different_order(self):
        """Different epochs → different shard order (with overwhelming probability)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readers = self._make_readers(10, root)
            # Use explicit policy with the family name — default_weight=0.0 (opt-in)
            policy = SourceWeightPolicy(weights={"robustsep": 1.0})
            s0 = [id(r) for r in weighted_shard_schedule({"robustsep": readers}, policy, epoch=0)]
            s1 = [id(r) for r in weighted_shard_schedule({"robustsep": readers}, policy, epoch=1)]
        self.assertNotEqual(s0, s1, "Epoch 0 and epoch 1 should produce different shard orders")

    def test_zero_weight_family_excluded(self):
        """Weight=0 excludes a family entirely."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            readers_a = self._make_readers(3, root)
            readers_b = self._make_readers(3, root)
            policy = SourceWeightPolicy(weights={"A": 1.0, "B": 0.0})
            schedule = weighted_shard_schedule({"A": readers_a, "B": readers_b}, policy, epoch=0)
        self.assertEqual(len(schedule), 3, "Zero-weight family must be excluded")
        self.assertTrue(all(r in readers_a for r in schedule))

    def test_source_weight_policy_default_weight(self):
        """Unknown family uses default_weight (now 0.0 = opt-in by default)."""
        # Default: default_weight=0.0, so unknown families are excluded
        p = SourceWeightPolicy(weights={"A": 0.5})
        self.assertAlmostEqual(p.get("A"), 0.5)
        self.assertAlmostEqual(p.get("UNKNOWN"), 0.0, msg="Unknown family must get default_weight=0.0")
        # Custom default_weight makes unlisted families included
        p2 = SourceWeightPolicy(weights={"A": 0.5}, default_weight=1.0)
        self.assertAlmostEqual(p2.get("B"), 1.0)

    def test_source_weight_policy_negative_weight_raises(self):
        p = SourceWeightPolicy(weights={"A": -1.0})
        with self.assertRaises(ValueError):
            p.get("A")

    def test_normalize_weights_sums_to_one(self):
        """normalize_weights returns fractions summing to 1.0."""
        p = SourceWeightPolicy(weights={"A": 1.0, "B": 0.5, "C": 0.5})
        fracs = p.normalize_weights(["A", "B", "C"])
        self.assertAlmostEqual(sum(fracs.values()), 1.0, places=6)
        self.assertAlmostEqual(fracs["A"], 0.5, places=6)  # A=1.0 / total=2.0
        self.assertAlmostEqual(fracs["B"], 0.25, places=6) # B=0.5 / total=2.0

    def test_normalize_weights_zero_weight_included_as_zero(self):
        """Zero-weight families appear as 0.0 in normalized output."""
        p = SourceWeightPolicy(weights={"A": 1.0, "B": 0.0})
        fracs = p.normalize_weights(["A", "B"])
        self.assertAlmostEqual(fracs["B"], 0.0)
        self.assertAlmostEqual(fracs["A"], 1.0)

    def test_normalize_weights_all_zero_returns_zeros(self):
        """All-zero weights → all fractions are zero (no divide-by-zero)."""
        p = SourceWeightPolicy(weights={"A": 0.0, "B": 0.0})
        fracs = p.normalize_weights(["A", "B"])
        self.assertEqual(fracs, {"A": 0.0, "B": 0.0})

    def test_policy_summary_keys(self):
        """policy_summary exposes priority-ordering semantics and weights."""
        p = SourceWeightPolicy(weights={"A": 1.0, "B": 0.5})
        summary = p.policy_summary(["A", "B"])
        self.assertIn("algorithm", summary)
        self.assertEqual(summary["semantics"], "weighted_priority_without_replacement")
        self.assertIn("raw_weights", summary)
        self.assertIn("normalized_priority_fractions", summary)
        self.assertIn("normalized_fractions", summary)
        self.assertIn("default_weight", summary)
        self.assertAlmostEqual(summary["default_weight"], 0.0)  # new default
        self.assertIn("A", summary["raw_weights"])



# ---------------------------------------------------------------------------
# TrainingAdapter tests
# ---------------------------------------------------------------------------

class TrainingAdapterTests(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.ds = _make_dataset(self.root, n_shards=6, n_patches=4)

    def tearDown(self):
        self._td.cleanup()

    def test_total_patches(self):
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)], batch_size=4)
        self.assertEqual(adapter.total_patches, len(self.ds))

    def test_iter_epoch_yields_batches(self):
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)], batch_size=4)
        batches = list(adapter.iter_epoch(0))
        self.assertGreater(len(batches), 0)
        for b in batches:
            self.assertIsInstance(b, ShardBatch)
            self.assertLessEqual(len(b), 4)

    def test_iter_epoch_covers_all_patches(self):
        """iter_epoch must visit every patch exactly once if no drop_last."""
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)],
                                  batch_size=2, drop_last=False)
        total = sum(len(b) for b in adapter.iter_epoch(0))
        self.assertEqual(total, len(self.ds))

    def test_iter_epoch_deterministic(self):
        """Two calls with same epoch give identical patch order."""
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)], batch_size=4)
        rgb_e0_run1 = [b.rgb[0].tolist() for b in adapter.iter_epoch(0)]
        rgb_e0_run2 = [b.rgb[0].tolist() for b in adapter.iter_epoch(0)]
        self.assertEqual(rgb_e0_run1, rgb_e0_run2)

    def test_different_epochs_different_order(self):
        """Different epochs give different shard orderings."""
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)], batch_size=2)
        rgb_e0 = [b.rgb[0, 0, 0, 0] for b in adapter.iter_epoch(0)]
        rgb_e1 = [b.rgb[0, 0, 0, 0] for b in adapter.iter_epoch(1)]
        # With 6 shards, the probability of identical order is negligible
        self.assertNotEqual(rgb_e0, rgb_e1,
                            "Different epochs must produce different shard orders")

    def test_export_split_manifest(self):
        """export_split_manifest writes a valid v1.1 JSON file with policy and alpha fields."""
        import json as _json
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)], batch_size=4)
        out = self.root / "split_manifest.json"
        adapter.export_split_manifest(out, alpha_policy="ones")
        self.assertTrue(out.exists())
        with out.open() as f:
            payload = _json.load(f)
        self.assertIn("families", payload)
        self.assertIn("total_patches", payload)
        self.assertEqual(payload["total_patches"], len(self.ds))
        # v1.1 fields
        self.assertEqual(payload["split_manifest_version"], "1.1")
        self.assertIn("source_weight_policy", payload)
        self.assertIn("alpha_policy", payload)
        self.assertEqual(payload["alpha_policy"], "ones")
        self.assertEqual(payload["source_weight_policy"]["semantics"], "weighted_priority_without_replacement")
        self.assertIn("normalized_priority_fractions", payload["source_weight_policy"])
        self.assertIn("normalized_fractions", payload["source_weight_policy"])

    def test_export_split_manifest_invalid_alpha_raises(self):
        """export_split_manifest raises ValueError for unknown alpha_policy."""
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds)], batch_size=4)
        out = self.root / "split_manifest.json"
        with self.assertRaises(ValueError):
            adapter.export_split_manifest(out, alpha_policy="INVALID")

    def test_family_names(self):
        adapter = TrainingAdapter([FamilyDataset("robustsep", self.ds),
                                   FamilyDataset("doclaynet", self.ds)], batch_size=4)
        self.assertEqual(sorted(adapter.family_names()), ["doclaynet", "robustsep"])

    def test_multi_family_total_patches(self):
        """Two families → total = sum of individual lens."""
        adapter = TrainingAdapter([FamilyDataset("A", self.ds),
                                   FamilyDataset("B", self.ds)], batch_size=4)
        self.assertEqual(adapter.total_patches, 2 * len(self.ds))


if __name__ == "__main__":
    unittest.main()
