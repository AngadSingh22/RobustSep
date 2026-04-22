from __future__ import annotations

import unittest

import numpy as np

from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.models.conditioning.drift import sample_drift_bank
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data import build_surrogate_example, extract_center_context, pad_patch_to_context


class SurrogateDataTests(unittest.TestCase):
    def test_pad_patch_to_context_shape(self) -> None:
        patch = np.zeros((16, 16, 7), dtype=np.float32)
        context = pad_patch_to_context(patch, context_size=32)
        self.assertEqual(context.shape, (32, 32, 7))

    def test_extract_center_context(self) -> None:
        cmykogv = np.zeros((40, 40, 7), dtype=np.float32)
        lab = np.zeros((40, 40, 3), dtype=np.float32)
        cmykogv[20, 20, 0] = 1.0
        out = extract_center_context(cmykogv, lab, center_x=20, center_y=20)
        self.assertEqual(out.cmykogv_context.shape, (32, 32, 7))
        self.assertEqual(out.lab_center.shape, (16, 16, 3))
        self.assertEqual(out.cmykogv_context[16, 16, 0], 1.0)

    def test_build_surrogate_example_contract(self) -> None:
        ppp = PPP.from_base()
        drift = sample_drift_bank(DriftConfig(sample_count=1), 7, "image", ppp.hash, (0, 0), sample_count=1)[0]
        patch = np.full((16, 16, 7), 0.25, dtype=np.float32)
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        example = build_surrogate_example(
            patch,
            lab,
            ppp=ppp,
            drift=drift,
            structure_token="flat",
            intent_weights={"brand": 0.0, "flat": 1.0, "gradient": 0.0},
            lambda_value=0.5,
            metadata={"source": "unit"},
        )
        self.assertEqual(example.cmykogv_context.shape, (32, 32, 7))
        self.assertEqual(example.lab_center.shape, (16, 16, 3))
        self.assertEqual(example.intent_raster.shape, (4, 4, 3))
        self.assertEqual(example.drifted_context.shape, (32, 32, 7))
        meta = example.to_metadata()
        self.assertEqual(meta["source"], "unit")
        self.assertEqual(meta["structure_token"], "flat")

    def test_default_intent_raster_channel_order_matches_data_adapter(self) -> None:
        ppp = PPP.from_base()
        drift = sample_drift_bank(DriftConfig(sample_count=1), 7, "image", ppp.hash, (0, 0), sample_count=1)[0]
        patch = np.zeros((16, 16, 7), dtype=np.float32)
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        example = build_surrogate_example(
            patch,
            lab,
            ppp=ppp,
            drift=drift,
            structure_token="edge",
            intent_weights={"brand": 0.2, "gradient": 0.3, "flat": 0.5},
        )
        np.testing.assert_allclose(example.intent_raster[0, 0], [0.2, 0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
