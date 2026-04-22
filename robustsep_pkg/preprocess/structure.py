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
    """Assign a structure token to patch (x, y).

    Token is one of ``{"edge", "flat", "textured"}``.  Exactly matches the
    deterministic rule in the resolved spec:

    .. code-block:: text

        edge      if edge_density >= theta_edge_density and coherence >= theta_edge_coh
        flat      if var_lab <= theta_flat_var_patch and edge_density <= theta_flat_density
        textured  otherwise
    """
    sl = np.s_[y : y + patch_size, x : x + patch_size]
    # Apply alpha_gamma exponent to alpha weights so this is consistent with
    # aggregate_patch_intents and the spec: mean_A(f) = sum alpha^gamma_alpha * f / sum alpha^gamma_alpha
    alpha_patch = np.clip(alpha[sl].astype(np.float32), 0.0, 1.0)
    weights = np.power(alpha_patch, config.alpha_gamma) if config.alpha_gamma != 1.0 else alpha_patch
    denom = max(float(weights.sum()), config.eps)

    edge = feature_maps["edge"][sl]
    var_lab = feature_maps["var_lab"][sl]
    rho = feature_maps["rho"][sl]
    log_abs = feature_maps["log_abs"][sl]

    edge_density = float(((edge > config.theta_edge).astype(np.float32) * weights).sum() / denom)
    var_mean = float((var_lab * weights).sum() / denom)
    coherence = float((rho * weights).sum() / denom)
    # texture_energy is used for logging/future use; the spec does not use it as a
    # branching condition (log_abs >= 0 always, so "if texture_energy >= 0" was dead code
    # that made the flat branch unreachable).
    _texture_energy = float((log_abs * weights).sum() / denom)  # noqa: F841

    if edge_density >= config.theta_edge_density and coherence >= config.theta_edge_coh:
        return "edge"
    if var_mean <= config.theta_flat_var_patch and edge_density <= config.theta_flat_density:
        return "flat"
    return "textured"
