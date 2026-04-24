from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset
from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN
from robustsep_pkg.models.surrogate.probe import (
    CandidateProbeConfig,
    _probe_dataset_indices,
    _ranking_summary,
    evaluate_candidate_probe,
    generate_lambda_probe_contexts,
)
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data import SurrogateShardWriterConfig, write_surrogate_training_shards
from robustsep_pkg.targets import TargetGenerationPipelineConfig, generate_target_records
from tests.test_target_generation_pipeline import _write_pipeline_fixture


class SurrogateCandidateProbeTests(unittest.TestCase):
    def _write_manifest(self, root: Path) -> Path:
        split_manifest = _write_pipeline_fixture(root, n=2)
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
            config=SurrogateShardWriterConfig(shard_size=2, run_id="probe-unit"),
        )
        return Path(summary.manifest_path)

    def test_generate_lambda_probe_contexts_is_deterministic_and_projected(self) -> None:
        ppp = PPP.from_base()
        context = np.zeros((32, 32, 7), dtype=np.float32)
        context[8:24, 8:24, 0] = 0.2
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 1] = 40.0
        config = CandidateProbeConfig(drift_sample_count=2, max_patches=1)

        a = generate_lambda_probe_contexts(context, lab, ppp, source_id="patch-a", config=config)
        b = generate_lambda_probe_contexts(context, lab, ppp, source_id="patch-a", config=config)

        self.assertEqual(len(a), 5)
        for left, right in zip(a, b):
            np.testing.assert_allclose(left, right)
            self.assertEqual(left.shape, (32, 32, 7))
            self.assertGreaterEqual(float(left.min()), 0.0)
            self.assertLessEqual(float(left.max()), 1.0)

    def test_evaluate_candidate_probe_reports_all_gate_metrics(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as td:
            manifest_path = self._write_manifest(Path(td))
            dataset = SurrogateTrainingDataset(manifest_path)
            model = ForwardSurrogateCNN()
            metrics = evaluate_candidate_probe(
                model,
                dataset,
                device=torch.device("cpu"),
                config=CandidateProbeConfig(drift_sample_count=2, max_patches=1, batch_size=4),
            )

        self.assertTrue(metrics.ranking_evaluated)
        self.assertEqual(metrics.patches_evaluated, 1)
        self.assertEqual(metrics.candidates_per_patch, 5)
        self.assertEqual(metrics.drifts_per_candidate, 2)
        self.assertGreaterEqual(metrics.mean_delta_e00, 0.0)
        self.assertGreaterEqual(metrics.q90_delta_e00, 0.0)
        self.assertGreaterEqual(metrics.top1_agreement, 0.0)
        self.assertLessEqual(metrics.top1_agreement, 1.0)
        self.assertGreaterEqual(metrics.spearman, -1.0)
        self.assertLessEqual(metrics.spearman, 1.0)
        self.assertGreaterEqual(metrics.strict_top1_agreement, 0.0)
        self.assertLessEqual(metrics.strict_top1_agreement, 1.0)
        self.assertGreaterEqual(metrics.tie_patch_fraction, 0.0)
        self.assertLessEqual(metrics.tie_patch_fraction, 1.0)

    def test_probe_dataset_indices_is_deterministic_and_not_prefix_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            manifest_path = self._write_manifest(Path(td))
            dataset = SurrogateTrainingDataset(manifest_path)
            config = CandidateProbeConfig(max_patches=4)
            a = _probe_dataset_indices(dataset, config)
            b = _probe_dataset_indices(dataset, config)

        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(len(a), 4)
        self.assertNotEqual(a.tolist(), list(range(4)))

    def test_ranking_summary_relaxes_tie_cases_but_keeps_strict_metrics(self) -> None:
        teacher_risk = np.asarray([0.50, 0.54, 0.80], dtype=np.float32)
        surrogate_risk = np.asarray([0.55, 0.49, 0.81], dtype=np.float32)

        summary = _ranking_summary(teacher_risk, surrogate_risk, tie_margin_delta_e00=0.10)

        self.assertEqual(summary["strict_top1_match"], 0)
        self.assertEqual(summary["top1_match"], 1)
        self.assertEqual(summary["tie_patch"], 1)
        self.assertLess(float(summary["strict_spearman"]), 1.0)
        self.assertEqual(float(summary["spearman"]), 1.0)
        self.assertAlmostEqual(float(summary["regret"]), 0.04, places=5)


if __name__ == "__main__":
    unittest.main()
