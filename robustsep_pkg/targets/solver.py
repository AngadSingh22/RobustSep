from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from robustsep_pkg.eval.metrics import delta_e_00
from robustsep_pkg.models.conditioning.drift import DriftSample, apply_drift
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.refiner.solver import pi_k

GradientFn = Callable[[np.ndarray, int, str], np.ndarray]

_EPS = 1e-8
_LAB_BIAS = np.asarray([100.0, 0.0, 0.0], dtype=np.float32)
_LAB_JACOBIAN = np.asarray(
    [
        [-18.0, -20.0, -16.0, -55.0, -12.0, -10.0, -14.0],
        [-35.0, 55.0, -8.0, 0.0, 35.0, -45.0, 50.0],
        [-20.0, -30.0, 70.0, 0.0, 55.0, 35.0, -55.0],
    ],
    dtype=np.float32,
)
_TRUST_CHANNEL_WEIGHTS = np.asarray([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)


@dataclass(frozen=True)
class ProjectedGradientTrace:
    stage1_steps: int
    stage2_steps: int
    projected_each_step: bool
    max_abs_update: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | bool | dict[str, float]]:
        return {
            "stage1_steps": self.stage1_steps,
            "stage2_steps": self.stage2_steps,
            "projected_each_step": self.projected_each_step,
            "max_abs_update": self.max_abs_update,
            "diagnostics": self.diagnostics,
        }


def projected_gradient_solve(
    initial: np.ndarray,
    ppp: PPP,
    *,
    lab_ref: np.ndarray | None = None,
    alpha_weights: np.ndarray | None = None,
    drift_samples: Sequence[DriftSample] | None = None,
    stage1_steps: int = 0,
    stage2_steps: int = 0,
    stage1_step_size: float = 0.0,
    stage2_step_size: float = 0.0,
    alpha_app: float = 1.0,
    alpha_anchor: float = 0.25,
    alpha_ogv: float = 0.05,
    alpha_neutral: float = 0.10,
    beta_risk: float = 1.0,
    beta_trust: float = 0.50,
    beta_ogv: float = 0.10,
    risk_q: float = 0.90,
    gradient_fn: GradientFn | None = None,
) -> tuple[np.ndarray, ProjectedGradientTrace]:
    """Two-stage fixed-step projected-gradient solve for CMYKOGV targets.

    With no custom `gradient_fn`, Stage 1 follows the documented nominal
    objective using the package's deterministic Lab proxy: appearance match,
    ICC-anchor appearance, OGV usage, and neutral/dark suppression. Stage 2
    adds q-risk over supplied drift samples, a channel-weighted trust region
    around the Stage 1 result, and an OGV penalty. Every update is closed by
    `Pi_K`.
    """
    y = pi_k(initial.astype(np.float32, copy=True), ppp, lab_ref=lab_ref)
    initial_projected = y.copy()
    lab = _coerce_lab(lab_ref, y.shape[:2])
    weights = _normalised_alpha(alpha_weights, y.shape[:2])
    max_abs_update = 0.0

    diagnostics: dict[str, float] = {}
    diagnostics["stage1_objective_initial"] = _stage1_objective(
        y, initial_projected, lab, weights, ppp, alpha_app, alpha_anchor, alpha_ogv, alpha_neutral
    )

    for step in range(stage1_steps):
        if gradient_fn is None:
            grad = _stage1_gradient(y, initial_projected, lab, weights, ppp, alpha_app, alpha_anchor, alpha_ogv, alpha_neutral)
        else:
            grad = gradient_fn(y.copy(), step, "stage1").astype(np.float32)
        if stage1_step_size != 0.0:
            update = stage1_step_size * grad
            max_abs_update = max(max_abs_update, float(np.max(np.abs(update))) if update.size else 0.0)
            y = pi_k(y - update, ppp, lab_ref=lab_ref)

    y_stage1 = y.copy()
    diagnostics["stage1_objective_final"] = _stage1_objective(
        y_stage1, initial_projected, lab, weights, ppp, alpha_app, alpha_anchor, alpha_ogv, alpha_neutral
    )
    diagnostics["stage2_objective_initial"] = _stage2_objective(
        y_stage1, y_stage1, lab, weights, drift_samples, beta_risk, beta_trust, beta_ogv, risk_q
    )

    for step in range(stage2_steps):
        if gradient_fn is None:
            grad = _stage2_gradient(y, y_stage1, lab, weights, drift_samples, beta_risk, beta_trust, beta_ogv, risk_q)
        else:
            grad = gradient_fn(y.copy(), step, "stage2").astype(np.float32)
        if stage2_step_size != 0.0:
            update = stage2_step_size * grad
            max_abs_update = max(max_abs_update, float(np.max(np.abs(update))) if update.size else 0.0)
            y = pi_k(y - update, ppp, lab_ref=lab_ref)

    diagnostics["stage2_objective_final"] = _stage2_objective(
        y, y_stage1, lab, weights, drift_samples, beta_risk, beta_trust, beta_ogv, risk_q
    )
    diagnostics["stage2_risk_selected_index"] = float(_risk_selected_index(y, lab, weights, drift_samples, risk_q))

    trace = ProjectedGradientTrace(
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        projected_each_step=True,
        max_abs_update=max_abs_update,
        diagnostics=diagnostics,
    )
    return y.astype(np.float32), trace


def render_cmykogv_lab_proxy(values: np.ndarray) -> np.ndarray:
    """Deterministic local Lab proxy used until a gated ICC/surrogate teacher exists."""
    y = np.asarray(values, dtype=np.float32)
    return (_LAB_BIAS + np.tensordot(y, _LAB_JACOBIAN.T, axes=1)).astype(np.float32)


def _calibrated_render(y: np.ndarray, anchor_y: np.ndarray, anchor_lab: np.ndarray) -> np.ndarray:
    return (anchor_lab + render_cmykogv_lab_proxy(y) - render_cmykogv_lab_proxy(anchor_y)).astype(np.float32)


def _stage1_gradient(
    y: np.ndarray,
    y0: np.ndarray,
    lab_ref: np.ndarray,
    weights: np.ndarray,
    ppp: PPP,
    alpha_app: float,
    alpha_anchor: float,
    alpha_ogv: float,
    alpha_neutral: float,
) -> np.ndarray:
    pred = _calibrated_render(y, y0, lab_ref)
    anchor = lab_ref
    grad_lab = alpha_app * _weighted_lab_residual(pred, lab_ref, weights)
    grad_lab += alpha_anchor * _weighted_lab_residual(pred, anchor, weights)
    grad = _lab_to_ink_gradient(grad_lab)
    grad[..., 4:7] += alpha_ogv * weights[..., None]

    if lab_ref is not None:
        neutral = _neutral_dark_mask(lab_ref, ppp)
        grad[..., 4:7] += alpha_neutral * weights[..., None] * neutral[..., None]
        cmy = y[..., :3]
        cmy_mean = np.mean(cmy, axis=-1, keepdims=True)
        grad[..., :3] += alpha_neutral * weights[..., None] * neutral[..., None] * (cmy - cmy_mean)
    return grad.astype(np.float32)


def _stage2_gradient(
    y: np.ndarray,
    y_stage1: np.ndarray,
    lab_ref: np.ndarray,
    weights: np.ndarray,
    drift_samples: Sequence[DriftSample] | None,
    beta_risk: float,
    beta_trust: float,
    beta_ogv: float,
    risk_q: float,
) -> np.ndarray:
    grad = np.zeros_like(y, dtype=np.float32)
    if drift_samples:
        idx = _risk_selected_index(y, lab_ref, weights, drift_samples, risk_q)
        drift = drift_samples[idx]
        drifted = apply_drift(y, drift)
        pred = _calibrated_render(drifted, y_stage1, lab_ref)
        grad_lab = _weighted_lab_residual(pred, lab_ref, weights)
        drift_scale = drift.multipliers.reshape((1, 1, -1)).astype(np.float32)
        grad += beta_risk * _lab_to_ink_gradient(grad_lab) * drift_scale
    grad += beta_trust * _TRUST_CHANNEL_WEIGHTS.reshape((1, 1, -1)) * (y - y_stage1) * weights[..., None]
    grad[..., 4:7] += beta_ogv * weights[..., None]
    return grad.astype(np.float32)


def _stage1_objective(
    y: np.ndarray,
    y0: np.ndarray,
    lab_ref: np.ndarray,
    weights: np.ndarray,
    ppp: PPP,
    alpha_app: float,
    alpha_anchor: float,
    alpha_ogv: float,
    alpha_neutral: float,
) -> float:
    pred = _calibrated_render(y, y0, lab_ref)
    anchor = lab_ref
    app = _weighted_delta_e(pred, lab_ref, weights)
    anch = _weighted_delta_e(pred, anchor, weights)
    ogv = float(np.sum(weights * y[..., 4:7].sum(axis=-1)))
    neutral = float(np.sum(weights * _neutral_dark_mask(lab_ref, ppp) * y[..., 4:7].sum(axis=-1)))
    return float(alpha_app * app + alpha_anchor * anch + alpha_ogv * ogv + alpha_neutral * neutral)


def _stage2_objective(
    y: np.ndarray,
    y_stage1: np.ndarray,
    lab_ref: np.ndarray,
    weights: np.ndarray,
    drift_samples: Sequence[DriftSample] | None,
    beta_risk: float,
    beta_trust: float,
    beta_ogv: float,
    risk_q: float,
) -> float:
    risk = 0.0
    if drift_samples:
        errors = [_weighted_delta_e(_calibrated_render(apply_drift(y, drift), y_stage1, lab_ref), lab_ref, weights) for drift in drift_samples]
        idx = max(0, min(len(errors) - 1, int(np.ceil(risk_q * len(errors))) - 1))
        risk = float(np.sort(np.asarray(errors, dtype=np.float32))[idx])
    trust = float(np.sum(weights[..., None] * _TRUST_CHANNEL_WEIGHTS.reshape((1, 1, -1)) * (y - y_stage1) ** 2))
    ogv = float(np.sum(weights * y[..., 4:7].sum(axis=-1)))
    return float(beta_risk * risk + beta_trust * trust + beta_ogv * ogv)


def _weighted_delta_e(pred: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * delta_e_00(pred, ref)))


def _risk_selected_index(
    y: np.ndarray,
    lab_ref: np.ndarray,
    weights: np.ndarray,
    drift_samples: Sequence[DriftSample] | None,
    risk_q: float,
) -> int:
    if not drift_samples:
        return -1
    errors = np.asarray(
        [_weighted_delta_e(_calibrated_render(apply_drift(y, drift), y, lab_ref), lab_ref, weights) for drift in drift_samples],
        dtype=np.float32,
    )
    order = np.argsort(errors, kind="mergesort")
    idx = max(0, min(errors.size - 1, int(np.ceil(risk_q * errors.size)) - 1))
    return int(order[idx])


def _weighted_lab_residual(pred: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (pred - ref).astype(np.float32) * weights[..., None]


def _lab_to_ink_gradient(grad_lab: np.ndarray) -> np.ndarray:
    return np.tensordot(grad_lab, _LAB_JACOBIAN, axes=1).astype(np.float32)


def _coerce_lab(lab_ref: np.ndarray | None, spatial_shape: tuple[int, int]) -> np.ndarray:
    if lab_ref is None:
        return np.zeros(spatial_shape + (3,), dtype=np.float32)
    lab = np.asarray(lab_ref, dtype=np.float32)
    if lab.shape != spatial_shape + (3,):
        raise ValueError(f"expected lab_ref shape {spatial_shape + (3,)}, got {lab.shape}")
    return lab


def _normalised_alpha(alpha_weights: np.ndarray | None, spatial_shape: tuple[int, int]) -> np.ndarray:
    if alpha_weights is None:
        alpha = np.ones(spatial_shape, dtype=np.float32)
    else:
        alpha = np.clip(np.asarray(alpha_weights, dtype=np.float32), 0.0, 1.0)
        if alpha.shape != spatial_shape:
            raise ValueError(f"expected alpha_weights shape {spatial_shape}, got {alpha.shape}")
    total = float(alpha.sum())
    if total <= _EPS:
        return np.zeros(spatial_shape, dtype=np.float32)
    return (alpha / total).astype(np.float32)


def _neutral_dark_mask(lab_ref: np.ndarray, ppp: PPP) -> np.ndarray:
    chroma = np.sqrt(lab_ref[..., 1] ** 2 + lab_ref[..., 2] ** 2)
    return ((chroma <= ppp.neutral_chroma_threshold) | (lab_ref[..., 0] <= ppp.dark_l_threshold)).astype(np.float32)
