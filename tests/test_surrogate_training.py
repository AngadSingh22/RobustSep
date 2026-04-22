from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset, iter_surrogate_shard_batches, ppp_condition_arrays
from robustsep_pkg.models.surrogate.model import SurrogateModelConfig
from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig
from robustsep_pkg.models.surrogate.training import (
    SurrogateLossConfig,
    SurrogateQualityGateThresholds,
    SurrogateQualityMetrics,
    SurrogateTrainingConfig,
    diagnose_surrogate_quality,
    evaluate_surrogate_quality,
    train_surrogate,
)
from robustsep_pkg.surrogate_data import SurrogateShardWriterConfig, write_surrogate_training_shards
from robustsep_pkg.targets import TargetGenerationPipelineConfig, generate_target_records
from tests.test_target_generation_pipeline import _write_pipeline_fixture


class SurrogateTrainingTests(unittest.TestCase):
    def _write_surrogate_manifest(self, root: Path, n: int = 4) -> tuple[Path, PPP]:
        split_manifest = _write_pipeline_fixture(root, n=n)
        ppp = PPP.from_base()
        records = generate_target_records(
            split_manifest,
            ppp,
            root=root,
            config=TargetGenerationPipelineConfig(drift_samples_per_patch=1),
        )
        summary = write_surrogate_training_shards(
            records,
            root / "surrogate",
            ppp,
            config=SurrogateShardWriterConfig(shard_size=2, run_id="train-unit"),
        )
        return Path(summary.manifest_path), ppp

    def test_surrogate_training_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path, ppp = self._write_surrogate_manifest(Path(td), n=2)
            ds = SurrogateTrainingDataset(manifest_path)
            sample = ds[0]
            numeric, mask, base_index = ppp_condition_arrays(ppp)

        self.assertEqual(len(ds), 2)
        self.assertEqual(tuple(sample["cmykogv_context"].shape), (32, 32, 7))
        self.assertEqual(tuple(sample["lab_center"].shape), (16, 16, 3))
        self.assertEqual(tuple(sample["drift_multipliers"].shape), (7,))
        self.assertEqual(tuple(sample["drift_trc_x"].shape), (9,))
        self.assertEqual(tuple(sample["drift_trc_y"].shape), (7, 9))
        self.assertEqual(tuple(sample["drift_vector"].shape), (56,))
        self.assertEqual(tuple(numeric.shape), (SurrogateModelConfig().ppp_numeric_dim,))
        self.assertEqual(tuple(mask.shape), (SurrogateModelConfig().ppp_override_mask_dim,))
        self.assertEqual(base_index, 0)

    def test_shard_stream_batches_cover_epoch_without_random_shard_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path, _ = self._write_surrogate_manifest(Path(td), n=5)
            batches = list(
                iter_surrogate_shard_batches(
                    manifest_path,
                    batch_size=2,
                    seed=7,
                    epoch=0,
                    shuffle_shards=True,
                    shuffle_within_shard=True,
                )
            )

        self.assertEqual(sum(int(batch["lab_center"].shape[0]) for batch in batches), 5)
        self.assertTrue(all(tuple(batch["cmykogv_context"].shape[1:]) == (32, 32, 7) for batch in batches))
        self.assertTrue(all(tuple(batch["drift_vector"].shape[1:]) == (56,) for batch in batches))
        self.assertTrue(all(tuple(batch["ppp_numeric"].shape[1:]) == (SurrogateModelConfig().ppp_numeric_dim,) for batch in batches))

    def test_train_surrogate_writes_checkpoint_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_path, _ = self._write_surrogate_manifest(root, n=4)
            result = train_surrogate(
                manifest_path,
                root / "train",
                training_config=SurrogateTrainingConfig(batch_size=2, epochs=1, device="cpu"),
                loss_config=SurrogateLossConfig(target_mode="teacher_proxy", hard_pixel_weight=0.5),
                gate_thresholds=SurrogateQualityGateThresholds(
                    threshold_mean=1e9,
                    threshold_q90=1e9,
                    threshold_spearman=-1.0,
                    threshold_top1=0.0,
                ),
                candidate_probe_config=CandidateProbeConfig(drift_sample_count=2, max_patches=1, batch_size=4),
            )
            self.assertGreater(result.train_loss, 0.0)
            self.assertTrue(Path(result.checkpoint_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.config_path).exists())
            self.assertTrue(Path(result.progress_path).exists())
            self.assertEqual(result.dataset_examples, 4)
            self.assertTrue(result.quality.ranking_evaluated)
            self.assertEqual(result.quality.probe_patches_evaluated, 1)
            self.assertEqual(result.quality.probe_candidates_per_patch, 5)
            self.assertEqual(result.quality.probe_drifts_per_candidate, 2)
            self.assertTrue(result.quality.passed)

    def test_gate_diagnosis_maps_failures_to_actions(self) -> None:
        metrics = SurrogateQualityMetrics(
            mean_delta_e00=6.0,
            q90_delta_e00=11.0,
            spearman=0.5,
            top1_agreement=0.4,
            ranking_evaluated=True,
            probe_patches_evaluated=8,
            probe_candidates_per_patch=5,
            probe_drifts_per_candidate=2,
            passed=False,
        )
        diagnosis = diagnose_surrogate_quality(metrics)

        self.assertTrue(diagnosis["failures"]["mean_delta_e00"])
        self.assertTrue(diagnosis["failures"]["q90_delta_e00"])
        self.assertIn("increase_hard_pixel_tail_weight", diagnosis["recommended_actions"])
        self.assertIn("keep_teacher_proxy_targets_and_expand_candidate_probe", diagnosis["recommended_actions"])

    def test_quality_gate_uses_probe_metrics_and_respects_thresholds(self) -> None:
        import torch

        from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset
        from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN

        with tempfile.TemporaryDirectory() as td:
            manifest_path, _ = self._write_surrogate_manifest(Path(td), n=2)
            ds = SurrogateTrainingDataset(manifest_path)
            model = ForwardSurrogateCNN()
            metrics = evaluate_surrogate_quality(
                model,
                ds,
                device=torch.device("cpu"),
                thresholds=SurrogateQualityGateThresholds(
                    threshold_mean=1e9,
                    threshold_q90=1e9,
                    threshold_spearman=-1.0,
                    threshold_top1=1.1,
                ),
                batch_size=2,
                candidate_probe_config=CandidateProbeConfig(drift_sample_count=2, max_patches=1, batch_size=4),
            )

        self.assertTrue(metrics.ranking_evaluated)
        self.assertEqual(metrics.probe_patches_evaluated, 1)
        self.assertEqual(metrics.probe_candidates_per_patch, 5)
        self.assertEqual(metrics.probe_drifts_per_candidate, 2)
        self.assertFalse(metrics.passed)


if __name__ == "__main__":
    unittest.main()
