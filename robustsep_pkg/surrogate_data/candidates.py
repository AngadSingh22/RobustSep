from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustsep_pkg.core.seeding import derive_seed
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.refiner.solver import pi_k


@dataclass(frozen=True)
class LambdaCandidateConfig:
    lambda_values: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.0)
    root_seed: int = 20260422
    ogv_probe_scale: float = 0.10
    ink_noise_scale: float = 0.012


def generate_lambda_candidate_contexts(
    cmykogv_context: np.ndarray,
    lab_ref: np.ndarray,
    ppp: PPP,
    *,
    source_id: str,
    config: LambdaCandidateConfig = LambdaCandidateConfig(),
) -> tuple[np.ndarray, ...]:
    """Generate deterministic lambda candidate contexts from one base context.

    The same utility is used by shard generation and the quality probe so the
    model is trained on the candidate family it is later asked to rank.
    """
    context = np.asarray(cmykogv_context, dtype=np.float32)
    if context.shape != (32, 32, 7):
        raise ValueError(f"expected cmykogv_context shape (32,32,7), got {context.shape}")
    lab = np.asarray(lab_ref, dtype=np.float32)
    if lab.shape != (16, 16, 3):
        raise ValueError(f"expected lab_ref shape (16,16,3), got {lab.shape}")

    center = center_patch(context)
    chroma_signal = lab_to_ogv_signal(lab)
    out: list[np.ndarray] = []
    for candidate_index, lambda_value in enumerate(config.lambda_values):
        lam = float(lambda_value)
        seed = derive_seed(config.root_seed, source_id, ppp.hash, "surrogate_candidate_probe", "center", candidate_index)
        rng = np.random.default_rng(seed)
        y = center.copy()
        if lam > 0.0:
            noise = rng.random((16, 16, 3), dtype=np.float32)
            ogv_delta = lam * config.ogv_probe_scale * (0.75 * chroma_signal + 0.25 * noise)
            y[..., 4:7] += ogv_delta.astype(np.float32)
            y[..., :3] *= np.float32(max(0.0, 1.0 - 0.035 * lam))
            y += rng.normal(0.0, config.ink_noise_scale * lam, size=y.shape).astype(np.float32)
        y = pi_k(y, ppp, lab_ref=lab)
        candidate_context = context.copy()
        start = (context.shape[0] - 16) // 2
        candidate_context[start : start + 16, start : start + 16, :] = y
        out.append(candidate_context.astype(np.float32))
    return tuple(out)


def center_patch(context: np.ndarray) -> np.ndarray:
    start = (context.shape[0] - 16) // 2
    return context[start : start + 16, start : start + 16, :]


def lab_to_ogv_signal(lab: np.ndarray) -> np.ndarray:
    a = np.clip(lab[..., 1] / 128.0, -1.0, 1.0)
    b = np.clip(lab[..., 2] / 128.0, -1.0, 1.0)
    signal = np.empty(lab.shape[:2] + (3,), dtype=np.float32)
    signal[..., 0] = np.clip(0.5 * a + 0.5 * b, 0.0, 1.0)
    signal[..., 1] = np.clip(-a, 0.0, 1.0)
    signal[..., 2] = np.clip(0.5 * a - 0.5 * b, 0.0, 1.0)
    return signal
