from __future__ import annotations

import math

import numpy as np


def finite_quantile(values: np.ndarray, q: float) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if arr.size == 0:
        raise ValueError("finite_quantile requires at least one value")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    idx = max(0, min(arr.size - 1, math.ceil(q * arr.size) - 1))
    return float(arr[idx])


def weighted_mean(values: np.ndarray, weights: np.ndarray, eps: float = 1e-8) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    denom = float(w.sum())
    if denom <= eps:
        return 0.0
    return float((v * w).sum() / denom)


def weighted_order_statistic(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if v.size != w.size:
        raise ValueError("values and weights must have the same flattened size")
    mask = w > 0
    if not np.any(mask):
        return 0.0
    v = v[mask]
    w = w[mask]
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cutoff = q * float(w.sum())
    idx = int(np.searchsorted(np.cumsum(w), cutoff, side="left"))
    return float(v[min(idx, v.size - 1)])


def delta_e_00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference for Lab arrays with last axis 3."""
    l1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=np.float64), -1, 0)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) * 0.5
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where((c1p * c2p) == 0.0, 0.0, dhp)
    dhp_term = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(dhp) * 0.5)

    l_bar = (l1 + l2) * 0.5
    c_bar_p = (c1p + c2p) * 0.5
    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    h_bar = np.where(
        (c1p * c2p) == 0.0,
        h_sum,
        np.where(h_diff <= 180.0, h_sum * 0.5, np.where(h_sum < 360.0, (h_sum + 360.0) * 0.5, (h_sum - 360.0) * 0.5)),
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    delta_theta = 30.0 * np.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt((c_bar_p**7) / (c_bar_p**7 + 25.0**7))
    s_l = 1.0 + (0.015 * ((l_bar - 50.0) ** 2)) / np.sqrt(20.0 + ((l_bar - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c
    return np.sqrt(
        (dlp / s_l) ** 2
        + (dcp / s_c) ** 2
        + (dhp_term / s_h) ** 2
        + r_t * (dcp / s_c) * (dhp_term / s_h)
    ).astype(np.float32)


def patch_error(
    delta_e: np.ndarray,
    alpha: np.ndarray,
    intent_maps: dict[str, np.ndarray],
    beta_brand: float = 2.0,
    beta_gradient: float = 1.5,
    beta_flat: float = 0.5,
    alpha_gamma: float = 1.0,
    tail_q: float | None = None,
    rho_tail: float = 0.0,
) -> float:
    gain = (
        1.0
        + beta_brand * intent_maps.get("brand", 0.0)
        + beta_gradient * intent_maps.get("gradient", 0.0)
        + beta_flat * intent_maps.get("flat", 0.0)
    )
    weights = np.power(np.clip(alpha.astype(np.float32), 0.0, 1.0), alpha_gamma) * gain
    mean = weighted_mean(delta_e, weights)
    if tail_q is None or rho_tail == 0.0:
        return mean
    return mean + rho_tail * weighted_order_statistic(delta_e, weights, tail_q)
