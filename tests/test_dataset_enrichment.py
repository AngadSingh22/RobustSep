"""tests/test_dataset_enrichment.py — enrichment, intent adapter, alpha fallback."""
from __future__ import annotations

import unittest

import numpy as np

from robustsep_pkg.data.enrichment import (
    EnrichmentConfig,
    EnrichedSample,
    apply_alpha_fallback,
    enrich_sample,
)
from robustsep_pkg.data.intent_adapter import (
    compute_intent_weights,
    compute_low_res_intent_raster,
    compute_structure_token,
)
from robustsep_pkg.data.shard_record import ShardRecord, ShardSample
from robustsep_pkg.core.config import PreprocessConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample(
    rgb: np.ndarray | None = None,
    lab: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    structure: str = "flat",
) -> ShardSample:
    """Minimal ShardSample for enrichment tests."""
    if rgb is None:
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    if lab is None:
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 0] = 50.0
    if alpha is None:
        alpha = np.ones((16, 16), dtype=np.float32)
    record = ShardRecord(
        shard_index=0, source_path="test.png", x=0, y=0,
        structure=structure, color="neutral",
    )
    cmyk = np.zeros((16, 16, 4), dtype=np.float32)
    cmykogv = np.zeros((16, 16, 7), dtype=np.float32)
    return ShardSample(
        rgb=rgb, alpha=alpha, lab=lab,
        icc_cmyk=cmyk, cmyk_baseline=cmyk, cmykogv_baseline=cmykogv,
        record=record,
    )


# ---------------------------------------------------------------------------
# Alpha fallback tests
# ---------------------------------------------------------------------------

class AlphaFallbackTests(unittest.TestCase):

    def test_policy_ones_returns_all_ones(self):
        """'ones' policy always returns all-ones regardless of stored alpha."""
        sample = _make_sample(alpha=np.full((16, 16), 0.3, dtype=np.float32))
        alpha = apply_alpha_fallback(sample, "ones")
        np.testing.assert_array_equal(alpha, 1.0)
        self.assertEqual(alpha.shape, (16, 16))

    def test_policy_passthrough_preserves_stored(self):
        """'passthrough' returns the stored alpha (clipped to [0,1])."""
        stored = np.full((16, 16), 0.7, dtype=np.float32)
        stored[0, 0] = -0.1  # out of range → clipped to 0
        sample = _make_sample(alpha=stored)
        alpha = apply_alpha_fallback(sample, "passthrough")
        self.assertAlmostEqual(float(alpha[4, 4]), 0.7, places=5)
        self.assertAlmostEqual(float(alpha[0, 0]), 0.0, places=5, msg="Clipped to 0")

    def test_policy_visible_threshold_bright_pixels(self):
        """'visible_threshold': pixels with L* > alpha_l_min get alpha=1."""
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 0] = 80.0   # all bright: L*=80 > alpha_l_min=10
        sample = _make_sample(lab=lab)
        alpha = apply_alpha_fallback(sample, "visible_threshold", alpha_l_min=10.0)
        np.testing.assert_array_equal(alpha, 1.0, err_msg="All bright → all alpha=1")

    def test_policy_visible_threshold_dark_pixels(self):
        """'visible_threshold': pixels with L* <= alpha_l_min get alpha=0."""
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 0] = 5.0   # all dark: L*=5 <= alpha_l_min=10
        sample = _make_sample(lab=lab)
        alpha = apply_alpha_fallback(sample, "visible_threshold", alpha_l_min=10.0)
        np.testing.assert_array_equal(alpha, 0.0, err_msg="All dark → all alpha=0")

    def test_policy_visible_threshold_mixed(self):
        """'visible_threshold': per-pixel thresholding."""
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[:8, :, 0] = 50.0   # top half bright
        lab[8:, :, 0] = 5.0    # bottom half dark
        sample = _make_sample(lab=lab)
        alpha = apply_alpha_fallback(sample, "visible_threshold", alpha_l_min=10.0)
        self.assertTrue(np.all(alpha[:8, :] == 1.0))
        self.assertTrue(np.all(alpha[8:, :] == 0.0))

    def test_invalid_policy_raises(self):
        sample = _make_sample()
        with self.assertRaises(ValueError):
            apply_alpha_fallback(sample, "INVALID_POLICY")


# ---------------------------------------------------------------------------
# Intent adapter tests
# ---------------------------------------------------------------------------

class IntentAdapterTests(unittest.TestCase):

    def test_compute_intent_weights_returns_three_keys(self):
        """compute_intent_weights returns brand, gradient, flat."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        weights = compute_intent_weights(rgb, alpha)
        self.assertIn("brand", weights)
        self.assertIn("gradient", weights)
        self.assertIn("flat", weights)

    def test_compute_intent_weights_non_negative(self):
        """All intent weights are non-negative."""
        rgb = np.random.default_rng(0).uniform(0, 1, (16, 16, 3)).astype(np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        w = compute_intent_weights(rgb, alpha)
        for k, v in w.items():
            self.assertGreaterEqual(v, 0.0, msg=f"weight '{k}' is negative: {v}")

    def test_compute_intent_weights_sum_approx_one(self):
        """For a fully-opaque patch, intent weights sum ≈ 1 (after normalisation)."""
        rgb = np.full((16, 16, 3), 0.4, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        w = compute_intent_weights(rgb, alpha)
        total = sum(w.values())
        self.assertAlmostEqual(total, 1.0, places=4,
                                msg=f"Intent weights should sum to ~1, got {total}")

    def test_compute_structure_token_returns_valid_token(self):
        """compute_structure_token returns one of the three valid tokens."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        token = compute_structure_token(rgb, alpha)
        self.assertIn(token, {"edge", "flat", "textured"})

    def test_flat_image_gives_flat_token(self):
        """A perfectly uniform patch should yield the 'flat' structure token."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        token = compute_structure_token(rgb, alpha)
        self.assertEqual(token, "flat")

    def test_compute_low_res_intent_raster_shape(self):
        """Raster shape is (raster_size, raster_size, 3)."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        raster = compute_low_res_intent_raster(rgb, alpha, raster_size=4)
        self.assertEqual(raster.shape, (4, 4, 3))

    def test_compute_low_res_intent_raster_shape_2x2(self):
        """raster_size=2 → (2, 2, 3)."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        raster = compute_low_res_intent_raster(rgb, alpha, raster_size=2)
        self.assertEqual(raster.shape, (2, 2, 3))

    def test_compute_low_res_intent_raster_non_negative(self):
        """All raster values are >= 0."""
        rgb = np.random.default_rng(1).uniform(0, 1, (16, 16, 3)).astype(np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        raster = compute_low_res_intent_raster(rgb, alpha, raster_size=4)
        self.assertTrue(np.all(raster >= 0.0))

    def test_compute_low_res_intent_raster_channels(self):
        """Raster channel 0=brand, 1=gradient, 2=flat — all non-negative for flat image."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.ones((16, 16), dtype=np.float32)
        raster = compute_low_res_intent_raster(rgb, alpha, raster_size=4)
        # For a flat uniform image, brand and gradient should be ~0, flat should be ~1
        brand_mean = float(raster[..., 0].mean())
        flat_mean = float(raster[..., 2].mean())
        self.assertGreater(flat_mean, brand_mean,
                            msg="Flat image: flat intent should dominate brand intent")

    def test_zero_alpha_does_not_crash(self):
        """Zero alpha (fully transparent) must not crash — returns zeros or uses eps."""
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        alpha = np.zeros((16, 16), dtype=np.float32)
        # Should not raise
        raster = compute_low_res_intent_raster(rgb, alpha, raster_size=4)
        self.assertEqual(raster.shape, (4, 4, 3))


# ---------------------------------------------------------------------------
# EnrichmentConfig / enrich_sample tests
# ---------------------------------------------------------------------------

class EnrichmentTests(unittest.TestCase):

    def test_enrich_no_recompute_uses_stored_structure(self):
        """Without recompute_structure, the stored record structure is returned."""
        sample = _make_sample(structure="edge")
        config = EnrichmentConfig(recompute_intent=False, recompute_structure=False)
        enriched = enrich_sample(sample, config)
        self.assertEqual(enriched.structure_token, "edge")

    def test_enrich_recompute_structure(self):
        """With recompute_structure=True, structure is computed from RGB."""
        # Use a flat uniform image → should give "flat"
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        sample = _make_sample(rgb=rgb, structure="edge")  # stored says "edge"
        config = EnrichmentConfig(recompute_structure=True)
        enriched = enrich_sample(sample, config)
        # The stored token is "edge" but recomputed from a flat image → "flat"
        self.assertEqual(enriched.structure_token, "flat",
                         msg="Recomputed structure should be 'flat' for uniform image")

    def test_enrich_no_intent_returns_none(self):
        """Without recompute_intent, intent_weights and intent_raster are None."""
        sample = _make_sample()
        config = EnrichmentConfig(recompute_intent=False)
        enriched = enrich_sample(sample, config)
        self.assertIsNone(enriched.intent_weights)
        self.assertIsNone(enriched.intent_raster)

    def test_enrich_recompute_intent_populates_weights(self):
        """With recompute_intent=True, intent_weights is a dict with expected keys."""
        sample = _make_sample()
        config = EnrichmentConfig(recompute_intent=True)
        enriched = enrich_sample(sample, config)
        self.assertIsNotNone(enriched.intent_weights)
        self.assertIn("brand", enriched.intent_weights)
        self.assertIn("gradient", enriched.intent_weights)
        self.assertIn("flat", enriched.intent_weights)

    def test_enrich_recompute_intent_populates_raster(self):
        """With recompute_intent=True, intent_raster has correct shape."""
        sample = _make_sample()
        config = EnrichmentConfig(recompute_intent=True, intent_raster_size=4)
        enriched = enrich_sample(sample, config)
        self.assertIsNotNone(enriched.intent_raster)
        self.assertEqual(enriched.intent_raster.shape, (4, 4, 3))

    def test_enrich_alpha_policy_ones(self):
        """'ones' alpha policy → all-ones effective alpha."""
        stored = np.full((16, 16), 0.3, dtype=np.float32)
        sample = _make_sample(alpha=stored)
        config = EnrichmentConfig(alpha_policy="ones")
        enriched = enrich_sample(sample, config)
        np.testing.assert_array_equal(enriched.alpha_effective, 1.0)

    def test_enrich_alpha_policy_passthrough(self):
        """'passthrough' alpha policy → stored alpha returned unchanged."""
        stored = np.full((16, 16), 0.6, dtype=np.float32)
        sample = _make_sample(alpha=stored)
        config = EnrichmentConfig(alpha_policy="passthrough")
        enriched = enrich_sample(sample, config)
        np.testing.assert_allclose(enriched.alpha_effective, 0.6, atol=1e-6)

    def test_enriched_sample_delegates_rgb(self):
        """EnrichedSample.rgb delegates to sample.rgb."""
        rgb = np.full((16, 16, 3), 0.7, dtype=np.float32)
        sample = _make_sample(rgb=rgb)
        enriched = enrich_sample(sample)
        np.testing.assert_array_equal(enriched.rgb, rgb)

    def test_to_dict_includes_structure_token(self):
        """EnrichedSample.to_dict() includes structure_token."""
        sample = _make_sample(structure="textured")
        config = EnrichmentConfig(recompute_intent=True)
        enriched = enrich_sample(sample, config)
        d = enriched.to_dict()
        self.assertIn("structure_token", d)
        self.assertIn("intent_weights", d)

    def test_to_dict_includes_raster_shape_when_present(self):
        sample = _make_sample()
        config = EnrichmentConfig(recompute_intent=True, intent_raster_size=4)
        enriched = enrich_sample(sample, config)
        d = enriched.to_dict()
        self.assertIn("intent_raster_shape", d)
        self.assertEqual(d["intent_raster_shape"], [4, 4, 3])

    def test_enrichment_config_defaults(self):
        """Default EnrichmentConfig: no recompute, ones policy, raster_size=4."""
        cfg = EnrichmentConfig()
        self.assertFalse(cfg.recompute_intent)
        self.assertFalse(cfg.recompute_structure)
        self.assertEqual(cfg.alpha_policy, "ones")
        self.assertEqual(cfg.intent_raster_size, 4)


if __name__ == "__main__":
    unittest.main()
