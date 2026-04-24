from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.proposer.data import ProposerTrainingDataset, iter_proposer_shard_batches
from robustsep_pkg.models.proposer.training import (
    ProposerOptimizationConfig,
    ProposerTrainingConfig,
    train_proposer,
)
from robustsep_pkg.proposer_data import ProposerShardWriterConfig, write_proposer_training_shards
from robustsep_pkg.targets import TargetGenerationPipelineConfig, generate_target_records
from tests.test_target_generation_pipeline import _write_pipeline_fixture


class ProposerTrainingTests(unittest.TestCase):
    def _write_proposer_manifest(self, root: Path, n: int = 4) -> Path:
        split_manifest = _write_pipeline_fixture(root, n=n)
        ppp = PPP.from_base()
        records = generate_target_records(
            split_manifest,
            ppp,
            root=root,
            config=TargetGenerationPipelineConfig(drift_samples_per_patch=1),
        )
        summary = write_proposer_training_shards(
            records,
            root / "proposer",
            ppp,
            config=ProposerShardWriterConfig(shard_size=2, run_id="proposer-unit"),
        )
        return Path(summary.manifest_path)

    def test_proposer_training_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path = self._write_proposer_manifest(Path(td), n=2)
            ds = ProposerTrainingDataset(manifest_path)
            sample = ds[0]

        self.assertEqual(len(ds), 2)
        self.assertEqual(tuple(sample["rgb_patch"].shape), (16, 16, 3))
        self.assertEqual(tuple(sample["alpha"].shape), (16, 16))
        self.assertEqual(tuple(sample["lab_ref_center"].shape), (16, 16, 3))
        self.assertEqual(tuple(sample["target_cmykogv"].shape), (16, 16, 7))
        self.assertEqual(tuple(sample["intent_raster"].shape), (4, 4, 3))
        self.assertEqual(tuple(sample["intent_weights"].shape), (3,))

    def test_proposer_shard_stream_batches_cover_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path = self._write_proposer_manifest(Path(td), n=5)
            batches = list(
                iter_proposer_shard_batches(
                    manifest_path,
                    batch_size=2,
                    seed=7,
                    epoch=0,
                    shuffle_shards=True,
                    shuffle_within_shard=True,
                )
            )

        self.assertEqual(sum(int(batch["rgb_patch"].shape[0]) for batch in batches), 5)
        self.assertTrue(all(tuple(batch["target_cmykogv"].shape[1:]) == (16, 16, 7) for batch in batches))

    def test_train_proposer_writes_checkpoint_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path = self._write_proposer_manifest(root, n=4)
            result = train_proposer(
                manifest_path,
                root / "train",
                training_config=ProposerTrainingConfig(batch_size=2, epochs=1, device="cpu"),
                optimization_config=ProposerOptimizationConfig(appearance_mode="teacher_proxy"),
            )
            self.assertGreater(result.train_loss, 0.0)
            self.assertTrue(Path(result.checkpoint_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.config_path).exists())
            self.assertTrue(Path(result.progress_path).exists())
            self.assertEqual(result.dataset_examples, 4)


if __name__ == "__main__":
    unittest.main()
