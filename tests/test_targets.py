from __future__ import annotations

import unittest

import numpy as np

from robustsep_pkg.models.conditioning.ppp import PPP, feasibility_violations
from robustsep_pkg.targets import TargetSolverConfig, TeacherMode, generate_target_from_icc_cmyk, initialize_cmykogv_from_icc


class TargetGenerationTests(unittest.TestCase):
    def test_initialize_cmykogv_shape_and_zero_ogv(self) -> None:
        cmyk = np.zeros((16, 16, 4), dtype=np.float32)
        cmyk[..., 0] = 0.25
        out = initialize_cmykogv_from_icc(cmyk)
        self.assertEqual(out.shape, (16, 16, 7))
        np.testing.assert_allclose(out[..., :4], cmyk)
        np.testing.assert_allclose(out[..., 4:], 0.0)

    def test_target_generation_projects_to_feasible(self) -> None:
        ppp = PPP.from_base("film_generic_conservative", {"tac_max": 2.0})
        cmyk = np.ones((16, 16, 4), dtype=np.float32)
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        result = generate_target_from_icc_cmyk(cmyk, lab, ppp)
        self.assertEqual(result.target_cmykogv.shape, (16, 16, 7))
        self.assertFalse(any(feasibility_violations(result.target_cmykogv, ppp, lab_ref=lab).values()))

    def test_neutral_dark_cap_respected(self) -> None:
        ppp = PPP.from_base("film_generic_conservative")
        cmyk = np.zeros((16, 16, 4), dtype=np.float32)
        cmyk[..., 0] = 0.1
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 0] = 10.0
        result = generate_target_from_icc_cmyk(cmyk, lab, ppp)
        self.assertLessEqual(float(result.target_cmykogv[..., 4:7].sum(axis=-1).max()), ppp.neutral_ogv_max + 1e-6)

    def test_generation_is_deterministic_and_manifested(self) -> None:
        ppp = PPP.from_base()
        cmyk = np.full((16, 16, 4), 0.2, dtype=np.float32)
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        config = TargetSolverConfig(teacher_mode=TeacherMode.ICC_ONLY, stage1_steps=2, stage1_step_size=0.0)
        a = generate_target_from_icc_cmyk(cmyk, lab, ppp, config=config, source_id="patch-a")
        b = generate_target_from_icc_cmyk(cmyk, lab, ppp, config=config, source_id="patch-a")
        np.testing.assert_allclose(a.target_cmykogv, b.target_cmykogv)
        self.assertEqual(a.manifest_record.target_hash, b.manifest_record.target_hash)
        self.assertEqual(a.manifest_record.teacher_mode, "icc_only")
        self.assertEqual(a.manifest_record.source_id, "patch-a")

    def test_projected_gradient_hook_runs_and_closes(self) -> None:
        ppp = PPP.from_base("film_generic_conservative", {"tac_max": 1.5})
        cmyk = np.full((16, 16, 4), 0.2, dtype=np.float32)
        lab = np.zeros((16, 16, 3), dtype=np.float32)

        def grad(y: np.ndarray, step: int, stage: str) -> np.ndarray:
            out = np.zeros_like(y)
            out[..., 4] = -10.0
            return out

        config = TargetSolverConfig(stage1_steps=1, stage1_step_size=1.0)
        result = generate_target_from_icc_cmyk(cmyk, lab, ppp, config=config, gradient_fn=grad)
        self.assertGreater(float(result.target_cmykogv[..., 4].mean()), 0.0)
        self.assertFalse(any(feasibility_violations(result.target_cmykogv, ppp, lab_ref=lab).values()))


if __name__ == "__main__":
    unittest.main()
