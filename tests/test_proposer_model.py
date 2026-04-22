from __future__ import annotations

import unittest

import torch

from robustsep_pkg.models.proposer.losses import lambda_monotonicity_hinge, proposer_vae_loss
from robustsep_pkg.models.proposer.model import ConditionalVAEProposer, ProposerModelConfig, build_proposer_input


class ProposerModelTests(unittest.TestCase):
    def test_build_proposer_input_shape(self) -> None:
        rgb = torch.zeros(2, 16, 16, 3)
        alpha = torch.ones(2, 16, 16)
        intent_weights = torch.zeros(2, 3)
        intent_raster = torch.zeros(2, 4, 4, 3)
        x = build_proposer_input(rgb, alpha, intent_weights, intent_raster)
        self.assertEqual(tuple(x.shape), (2, 10, 16, 16))

    def test_forward_output_shape(self) -> None:
        config = ProposerModelConfig()
        model = ConditionalVAEProposer(config)
        n = 2
        z = torch.zeros(n, config.latent_dim)
        out = model(
            torch.zeros(n, 16, 16, 3),
            torch.ones(n, 16, 16),
            base_family_index=torch.zeros(n, dtype=torch.long),
            ppp_numeric=torch.zeros(n, config.ppp_numeric_dim),
            ppp_override_mask=torch.zeros(n, config.ppp_override_mask_dim),
            structure_index=torch.zeros(n, dtype=torch.long),
            intent_weights=torch.zeros(n, 3),
            intent_raster=torch.zeros(n, 4, 4, 3),
            lambda_value=torch.full((n,), 0.5),
            z=z,
        )
        self.assertEqual(config.condition_dim, 289)
        self.assertEqual(tuple(out.cmykogv.shape), (n, 7, 16, 16))
        self.assertEqual(tuple(out.latent_mean.shape), (n, config.latent_dim))

    def test_losses_are_finite_and_monotonicity_penalizes_decrease(self) -> None:
        pred = torch.zeros(2, 7, 16, 16)
        target = torch.ones(2, 7, 16, 16)
        mean = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        losses = proposer_vae_loss(pred, target, mean, logvar)
        self.assertTrue(torch.isfinite(losses["total"]))
        low = torch.zeros(2, 7, 16, 16)
        high = torch.zeros(2, 7, 16, 16)
        high[:, 4:7] = 0.5
        self.assertEqual(float(lambda_monotonicity_hinge([low, high])), 0.0)
        self.assertGreater(float(lambda_monotonicity_hinge([high, low])), 0.0)


if __name__ == "__main__":
    unittest.main()
