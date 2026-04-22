from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class ForwardSurrogateModelTests(unittest.TestCase):
    def test_forward_shape_and_condition_dim(self) -> None:
        import torch

        from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN, SurrogateModelConfig

        config = SurrogateModelConfig()
        model = ForwardSurrogateCNN(config)
        n = 2
        out = model(
            torch.zeros(n, 32, 32, 7),
            base_family_index=torch.zeros(n, dtype=torch.long),
            ppp_numeric=torch.zeros(n, config.ppp_numeric_dim),
            ppp_override_mask=torch.zeros(n, config.ppp_override_mask_dim),
            structure_index=torch.zeros(n, dtype=torch.long),
            intent_weights=torch.zeros(n, 3),
            intent_raster=torch.zeros(n, 4, 4, 3),
            lambda_value=torch.full((n,), 0.5),
            drift_vector=torch.zeros(n, config.drift_dim),
        )
        self.assertEqual(config.base_condition_dim, 289)
        self.assertEqual(config.surrogate_condition_dim, 345)
        self.assertEqual(tuple(out.shape), (n, 3, 16, 16))


if __name__ == "__main__":
    unittest.main()
