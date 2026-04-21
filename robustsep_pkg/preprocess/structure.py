from __future__ import annotations

import numpy as np

from robustsep_pkg.core.config import PreprocessConfig


def structure_token_for_patch(
    feature_maps: dict[str, np.ndarray],
    alpha: np.ndarray,
    x: int,
    y: int,
    patch_size: int = 16,
    config: PreprocessConfig = PreprocessConfig(),
) -> str:
    sl = np.s_[y : y + patch_size, x : x + patch_size]
    weights = np.clip(alpha[sl].astype(np.float32), 0.0, 1.0)
    denom = max(float(weights.sum()), config.eps)

    edge = feature_maps["edge"][sl]
    var_lab = feature_maps["var_lab"][sl]
    rho = feature_maps["rho"][sl]
    log_abs = feature_maps["log_abs"][sl]

    edge_density = float(((edge > config.theta_edge).astype(np.float32) * weights).sum() / denom)
    var_mean = float((var_lab * weights).sum() / denom)
    coherence = float((rho * weights).sum() / denom)
    texture_energy = float((log_abs * weights).sum() / denom)

    if edge_density >= config.theta_edge_density and coherence >= config.theta_edge_coh:
        return "edge"
    if var_mean <= config.theta_flat_var_patch and edge_density <= config.theta_flat_density:
        return "flat"
    if texture_energy >= 0.0:
        return "textured"
    return "textured"
