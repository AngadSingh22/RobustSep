from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustsep_pkg.core.channels import CHANNELS_CMYKOGV, ensure_cmykogv_last_axis
from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.core.seeding import derive_seed


@dataclass(frozen=True)
class DriftSample:
    multipliers: np.ndarray
    trc_x: np.ndarray
    trc_y: np.ndarray


def _sample_truncated_normal(rng: np.random.Generator, sigma: float, bound: float, shape: tuple[int, ...]) -> np.ndarray:
    out = rng.normal(0.0, sigma, size=shape)
    bad = np.abs(out) > bound
    attempts = 0
    while np.any(bad) and attempts < 100:
        out[bad] = rng.normal(0.0, sigma, size=int(np.count_nonzero(bad)))
        bad = np.abs(out) > bound
        attempts += 1
    return np.clip(out, -bound, bound)


def isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Deterministic PAVA for a 1D sequence."""
    levels: list[float] = []
    weights: list[int] = []
    for value in values.astype(float):
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            w = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / w
            levels[-2:] = [merged]
            weights[-2:] = [w]
    return np.asarray([level for level, weight in zip(levels, weights) for _ in range(weight)], dtype=np.float32)


def sample_drift_bank(
    config: DriftConfig,
    root_seed: int,
    input_hash: str,
    ppp_hash: str,
    patch_coord: tuple[int, int] | str,
    candidate_index: int | None = None,
    sample_count: int | None = None,
) -> list[DriftSample]:
    n = sample_count or config.sample_count
    samples: list[DriftSample] = []
    j_count = config.trc_interior_knots
    trc_x = np.linspace(0.0, 1.0, j_count + 2, dtype=np.float32)
    for drift_i in range(n):
        seed = derive_seed(root_seed, input_hash, ppp_hash, "drift", patch_coord, candidate_index, drift_i)
        rng = np.random.default_rng(seed)
        multipliers = np.empty((len(CHANNELS_CMYKOGV),), dtype=np.float32)
        trc_y = np.empty((len(CHANNELS_CMYKOGV), j_count + 2), dtype=np.float32)
        for channel_idx, channel in enumerate(CHANNELS_CMYKOGV):
            log_m = _sample_truncated_normal(
                rng,
                config.multiplier_sigma[channel],
                config.multiplier_clip[channel],
                (),
            )
            multipliers[channel_idx] = float(np.exp(log_m))
            deltas = _sample_truncated_normal(
                rng,
                config.trc_sigma[channel],
                config.trc_clip[channel],
                (j_count,),
            )
            raw = trc_x.copy()
            raw[1:-1] = raw[1:-1] + deltas
            raw[0] = 0.0
            raw[-1] = 1.0
            y = isotonic_non_decreasing(raw)
            y[0] = 0.0
            y[-1] = 1.0
            trc_y[channel_idx] = np.clip(y, 0.0, 1.0)
        samples.append(DriftSample(multipliers=multipliers, trc_x=trc_x.copy(), trc_y=trc_y))
    return samples


def apply_drift(values: np.ndarray, drift: DriftSample) -> np.ndarray:
    ensure_cmykogv_last_axis(values.shape)
    z = values.astype(np.float32, copy=False)
    out = np.empty_like(z, dtype=np.float32)
    for channel_idx in range(len(CHANNELS_CMYKOGV)):
        curved = np.interp(z[..., channel_idx].reshape(-1), drift.trc_x, drift.trc_y[channel_idx]).reshape(z.shape[:-1])
        out[..., channel_idx] = drift.multipliers[channel_idx] * curved
    return np.clip(out, 0.0, 1.0)
