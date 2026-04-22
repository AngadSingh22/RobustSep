from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robustsep_pkg.core.artifact_io import read_json
from robustsep_pkg.eval.suite import PaperEvalConfig, edge_error_pct, psnr, run_paper_evaluation_suite, ssim
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.targets import TargetGenerationPipelineConfig
from tests.test_target_generation_pipeline import _write_pipeline_fixture


class PaperEvaluationSuiteTests(unittest.TestCase):
    def test_basic_image_metrics(self) -> None:
        a = np.zeros((16, 16), dtype=np.float32)
        b = np.zeros((16, 16), dtype=np.float32)
        self.assertEqual(psnr(a, b), float("inf"))
        self.assertAlmostEqual(ssim(a, b), 1.0)
        self.assertAlmostEqual(edge_error_pct(a, b), 0.0)

    def test_run_paper_evaluation_suite_writes_report_and_visual_npz(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split_manifest = _write_pipeline_fixture(root, n=3)
            out = root / "paper_eval.json"
            visual = root / "visual_examples.npz"
            ppp = PPP.from_base()
            summary = run_paper_evaluation_suite(
                split_manifest,
                ppp,
                out,
                root=root,
                config=PaperEvalConfig(
                    max_records=2,
                    drift_samples=2,
                    visual_examples=1,
                    target_config=TargetGenerationPipelineConfig(drift_samples_per_patch=1),
                ),
                visual_npz=visual,
            )
            report = read_json(out)
            with np.load(visual) as arrays:
                visual_keys = set(arrays.files)

        self.assertEqual(summary.records_evaluated, 2)
        self.assertEqual(report["suite"], "paper_evaluation_v1")
        self.assertEqual(report["records_evaluated"], 2)
        self.assertIn("quantitative_results", report)
        self.assertIn("percentile_robustness", report)
        self.assertIn("distribution_analysis", report)
        self.assertIn("ablation", report)
        self.assertIn("full_model", report["ablation"])
        self.assertIn("without_ppp_constraints", report["ablation"])
        self.assertIn("lab_ref", visual_keys)
        self.assertIn("delta_e00", visual_keys)


if __name__ == "__main__":
    unittest.main()
