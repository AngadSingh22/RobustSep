from __future__ import annotations

import numpy as np

D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
D50 = np.array([0.96422, 1.0, 0.82521], dtype=np.float32)

SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float32,
)
BRADFORD_INV = np.linalg.inv(BRADFORD).astype(np.float32)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_rgb_to_xyz_d65(linear_rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(np.asarray(linear_rgb, dtype=np.float32), SRGB_TO_XYZ_D65.T, axes=1).astype(np.float32)


def adapt_xyz_d65_to_d50(xyz_d65: np.ndarray) -> np.ndarray:
    src = BRADFORD @ D65
    dst = BRADFORD @ D50
    adapt = BRADFORD_INV @ np.diag(dst / src) @ BRADFORD
    return np.tensordot(np.asarray(xyz_d65, dtype=np.float32), adapt.T, axes=1).astype(np.float32)


def xyz_d50_to_lab(xyz_d50: np.ndarray) -> np.ndarray:
    scaled = np.asarray(xyz_d50, dtype=np.float32) / D50
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(scaled > eps, np.cbrt(np.maximum(scaled, 0.0)), (kappa * scaled + 16.0) / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1).astype(np.float32)


def rgb_to_lab_d50(rgb: np.ndarray) -> np.ndarray:
    return xyz_d50_to_lab(adapt_xyz_d65_to_d50(linear_rgb_to_xyz_d65(srgb_to_linear(rgb))))


def rgb_to_cmyk_baseline(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    cmy = 1.0 - rgb
    k = np.min(cmy, axis=-1, keepdims=True)
    denom = np.maximum(1.0 - k, 1e-6)
    cmy_out = np.where(k >= 0.999, 0.0, (cmy - k) / denom)
    return np.concatenate([cmy_out, k], axis=-1).astype(np.float32)


def cmyk_to_cmykogv(cmyk: np.ndarray) -> np.ndarray:
    cmyk = np.asarray(cmyk, dtype=np.float32)
    zeros = np.zeros(cmyk.shape[:-1] + (3,), dtype=np.float32)
    return np.concatenate([cmyk, zeros], axis=-1)
