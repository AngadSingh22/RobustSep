from __future__ import annotations

import numpy as np

from robustsep_pkg.core.config import PreprocessConfig
from robustsep_pkg.preprocess.color import rgb_to_lab_d50
from robustsep_pkg.preprocess.patches import raised_cosine_window


def _pad_reflect(x: np.ndarray, radius: int) -> np.ndarray:
    return np.pad(x, ((radius, radius), (radius, radius)), mode="reflect")


def box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x.astype(np.float32)
    padded = _pad_reflect(np.asarray(x, dtype=np.float32), radius)
    out = np.zeros_like(x, dtype=np.float32)
    side = 2 * radius + 1
    for dy in range(side):
        for dx in range(side):
            out += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return out / float(side * side)


def sobel_xy(luminance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(luminance, dtype=np.float32)
    p = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -p[:-2, :-2]
        - 2.0 * p[1:-1, :-2]
        - p[2:, :-2]
        + p[:-2, 2:]
        + 2.0 * p[1:-1, 2:]
        + p[2:, 2:]
    ) / 8.0
    gy = (
        -p[:-2, :-2]
        - 2.0 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2.0 * p[2:, 1:-1]
        + p[2:, 2:]
    ) / 8.0
    return gx.astype(np.float32), gy.astype(np.float32)


def laplacian_of_gaussian_proxy(luminance: np.ndarray) -> np.ndarray:
    x = np.asarray(luminance, dtype=np.float32)
    p = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    lap = -4.0 * p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return lap.astype(np.float32)


def compute_feature_maps(rgb: np.ndarray, config: PreprocessConfig = PreprocessConfig()) -> dict[str, np.ndarray]:
    lab = rgb_to_lab_d50(rgb)
    l = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    gx, gy = sobel_xy(l)
    edge = np.sqrt(gx * gx + gy * gy)
    local_l = box_mean(l, config.local_radius)
    local_l2 = box_mean(l * l, config.local_radius)
    local_var_l = np.maximum(local_l2 - local_l * local_l, 0.0)
    local_chroma_var = np.maximum(box_mean(chroma * chroma, config.local_radius) - box_mean(chroma, config.local_radius) ** 2, 0.0)
    var_lab = local_var_l + local_chroma_var

    jxx = box_mean(gx * gx, config.local_radius)
    jyy = box_mean(gy * gy, config.local_radius)
    jxy = box_mean(gx * gy, config.local_radius)
    trace = jxx + jyy
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    lambda1 = 0.5 * (trace + disc)
    lambda2 = 0.5 * (trace - disc)
    rho = (lambda1 - lambda2) / (lambda1 + lambda2 + config.eps)

    hue = np.arctan2(b, a)
    valid = chroma > 1.0
    cos_mean = box_mean(np.where(valid, np.cos(hue), 0.0), config.local_radius)
    sin_mean = box_mean(np.where(valid, np.sin(hue), 0.0), config.local_radius)
    valid_mean = np.maximum(box_mean(valid.astype(np.float32), config.local_radius), config.eps)
    hue_smoothness = 1.0 - np.sqrt(cos_mean * cos_mean + sin_mean * sin_mean) / valid_mean
    hue_smoothness = np.clip(hue_smoothness, 0.0, 1.0)
    return {
        "lab": lab,
        "chroma": chroma.astype(np.float32),
        "edge": edge.astype(np.float32),
        "edge_smooth": box_mean(edge, config.local_radius).astype(np.float32),
        "var_lab": var_lab.astype(np.float32),
        "rho": rho.astype(np.float32),
        "hue_smoothness": hue_smoothness.astype(np.float32),
        "log_abs": np.abs(laplacian_of_gaussian_proxy(l)).astype(np.float32),
    }


def compute_intent_maps(
    rgb: np.ndarray,
    user_brand_mask: np.ndarray | None = None,
    config: PreprocessConfig = PreprocessConfig(),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    f = compute_feature_maps(rgb, config)
    flat_raw = (f["var_lab"] <= config.theta_flat_var) & (f["edge"] <= config.theta_flat_edge)
    gradient_raw = (
        (f["edge_smooth"] >= config.theta_grad_min)
        & (f["edge_smooth"] <= config.theta_grad_max)
        & (f["rho"] >= config.theta_grad_coh)
        & (f["hue_smoothness"] <= config.theta_hue_smooth)
        & (f["var_lab"] > config.theta_flat_var)
    )
    brand_heuristic = (
        (f["chroma"] >= config.theta_brand_chroma)
        & (f["edge"] >= config.theta_brand_edge)
        & (f["rho"] >= config.theta_brand_coh)
    )
    if user_brand_mask is not None:
        brand_raw = brand_heuristic | (np.asarray(user_brand_mask) > 0)
    else:
        brand_raw = brand_heuristic
    brand = brand_raw.astype(np.float32)
    gradient = (gradient_raw.astype(np.float32) * (1.0 - brand)).astype(np.float32)
    flat_candidate = flat_raw.astype(np.float32) * (1.0 - brand) * (1.0 - gradient)
    flat = np.maximum(flat_candidate, 1.0 - brand - gradient).astype(np.float32)
    return {"brand": brand, "gradient": gradient, "flat": flat}, f


def aggregate_patch_intents(
    intent_maps: dict[str, np.ndarray],
    alpha: np.ndarray,
    x: int,
    y: int,
    patch_size: int = 16,
    alpha_gamma: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, float]:
    window = raised_cosine_window(patch_size)
    a = np.power(np.clip(alpha[y : y + patch_size, x : x + patch_size].astype(np.float32), 0.0, 1.0), alpha_gamma)
    base = window * a
    denom = max(float(base.sum()), eps)
    raw = {
        name: float((base * m[y : y + patch_size, x : x + patch_size]).sum() / denom)
        for name, m in intent_maps.items()
    }
    total = max(sum(raw.values()), eps)
    return {name: value / total for name, value in raw.items()}
