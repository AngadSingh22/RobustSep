from __future__ import annotations

import unittest

import numpy as np

from robustsep_pkg.core.seeding import derive_seed
from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.eval.metrics import delta_e_00, finite_quantile
from robustsep_pkg.models.conditioning.drift import apply_drift, sample_drift_bank
from robustsep_pkg.models.conditioning.ppp import PPP, feasibility_violations, project_to_feasible
from robustsep_pkg.preprocess.intent import aggregate_patch_intents, compute_intent_maps
from robustsep_pkg.preprocess.patches import deterministic_patch_grid, extract_alpha_patches
from robustsep_pkg.preprocess.structure import structure_token_for_patch


class CorePrimitiveTests(unittest.TestCase):
    def test_delta_e_00_reference_pair(self) -> None:
        lab1 = np.array([[[50.0, 2.6772, -79.7751]]], dtype=np.float32)
        lab2 = np.array([[[50.0, 0.0, -82.7485]]], dtype=np.float32)
        got = float(delta_e_00(lab1, lab2)[0, 0])
        self.assertAlmostEqual(got, 2.0425, places=3)

    def test_finite_quantile_uses_order_statistic(self) -> None:
        values = np.arange(32, dtype=np.float32)
        self.assertEqual(finite_quantile(values, 0.90), 28.0)

    def test_projection_closes_feasibility_and_preserves_feasible(self) -> None:
        ppp = PPP.from_base()
        feasible = np.zeros((2, 2, 7), dtype=np.float32)
        feasible[..., 0] = 0.2
        feasible[..., 1] = 0.1
        feasible_projected = project_to_feasible(feasible, ppp)
        np.testing.assert_allclose(feasible_projected, feasible)

        invalid = np.ones((4, 4, 7), dtype=np.float32)
        projected = project_to_feasible(invalid, ppp)
        self.assertFalse(any(feasibility_violations(projected, ppp).values()))
        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected <= ppp.cap_vector + 1e-6))

    def test_seed_derivation_is_replayable_and_scoped(self) -> None:
        seed_a = derive_seed(123, "image", "ppp", "candidate", (4, 8), 1)
        seed_b = derive_seed(123, "image", "ppp", "candidate", (4, 8), 1)
        seed_c = derive_seed(123, "image", "ppp", "candidate", (4, 8), 2)
        self.assertEqual(seed_a, seed_b)
        self.assertNotEqual(seed_a, seed_c)

    def test_drift_bank_is_replayable_and_monotone(self) -> None:
        ppp = PPP.from_base()
        config = DriftConfig()
        bank_a = sample_drift_bank(config=config, root_seed=7, input_hash="i", ppp_hash=ppp.hash, patch_coord=(0, 0), sample_count=2)
        bank_b = sample_drift_bank(config=config, root_seed=7, input_hash="i", ppp_hash=ppp.hash, patch_coord=(0, 0), sample_count=2)
        np.testing.assert_allclose(bank_a[0].multipliers, bank_b[0].multipliers)
        np.testing.assert_allclose(bank_a[0].trc_y, bank_b[0].trc_y)
        self.assertTrue(np.all(np.diff(bank_a[0].trc_y, axis=1) >= -1e-7))
        values = np.linspace(0.0, 1.0, 7, dtype=np.float32).reshape(1, 1, 7)
        drifted = apply_drift(values, bank_a[0])
        self.assertEqual(drifted.shape, values.shape)
        self.assertTrue(np.all((drifted >= 0.0) & (drifted <= 1.0)))

    def test_patch_grid_includes_edges(self) -> None:
        coords = deterministic_patch_grid(20, 20, patch_size=16, stride=8, include_edges=True)
        self.assertEqual(coords, [(0, 0), (4, 0), (0, 4), (4, 4)])

    def test_intent_and_structure_are_deterministic(self) -> None:
        image = np.zeros((20, 20, 4), dtype=np.uint8)
        image[..., :3] = 128
        image[..., 3] = 255
        image[:, 10:, 0] = 255
        image[:, 10:, 1] = 0
        image[:, 10:, 2] = 0
        patches = list(extract_alpha_patches(image, patch_size=16, stride=8))
        self.assertTrue(patches)
        rgb = image[..., :3].astype(np.float32) / 255.0
        alpha = image[..., 3].astype(np.float32) / 255.0
        intents_a, features_a = compute_intent_maps(rgb)
        intents_b, features_b = compute_intent_maps(rgb)
        for key in intents_a:
            np.testing.assert_allclose(intents_a[key], intents_b[key])
        weights = aggregate_patch_intents(intents_a, alpha, patches[0].x, patches[0].y)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)
        token = structure_token_for_patch(features_a, alpha, patches[0].x, patches[0].y)
        self.assertIn(token, {"edge", "flat", "textured"})


if __name__ == "__main__":
    unittest.main()
