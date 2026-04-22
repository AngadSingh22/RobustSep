from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Surrogate candidate probe requires PyTorch. Install torch to use this module.") from exc

from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.core.seeding import derive_seed
from robustsep_pkg.eval.metrics import delta_e_00, finite_quantile
from robustsep_pkg.models.conditioning.drift import DriftSample, apply_drift, sample_drift_bank
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.refiner.solver import pi_k
from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset
from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN
from robustsep_pkg.targets.teacher import calibrated_cmykogv_lab


@dataclass(frozen=True)
class CandidateProbeConfig:
    """Deterministic heldout probe used by the surrogate quality gate."""

    lambda_values: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.0)
    drift_sample_count: int = 32
    root_seed: int = 20260422
    max_patches: int | None = None
    risk_q: float = 0.90
    batch_size: int = 64
    ogv_probe_scale: float = 0.10
    ink_noise_scale: float = 0.012
    drift_config: DriftConfig = field(default_factory=DriftConfig)


@dataclass(frozen=True)
class CandidateProbeMetrics:
    mean_delta_e00: float
    q90_delta_e00: float
    spearman: float
    top1_agreement: float
    ranking_evaluated: bool
    patches_evaluated: int
    candidates_per_patch: int
    drifts_per_candidate: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@torch.no_grad()
def evaluate_candidate_probe(
    model: ForwardSurrogateCNN,
    dataset: SurrogateTrainingDataset,
    *,
    device: torch.device | str,
    config: CandidateProbeConfig = CandidateProbeConfig(),
) -> CandidateProbeMetrics:
    """Run the documented fixed candidate probe against the ICC/Lab teacher proxy.

    For each heldout patch, five deterministic lambda candidates are generated,
    each candidate is evaluated over a PPP-family drift bank, and ranking
    metrics compare surrogate candidate ordering against the teacher ordering.
    """
    if config.drift_sample_count < 1:
        raise ValueError("candidate probe requires at least one drift sample")
    if len(config.lambda_values) < 2:
        raise ValueError("candidate probe requires at least two lambda candidates")
    if config.batch_size < 1:
        raise ValueError("candidate probe batch_size must be >= 1")

    model.eval()
    torch_device = torch.device(device)
    ppp = dataset.ppp
    drift_bank = _fixed_family_drift_bank(ppp, config)
    patch_count = _patch_count(len(dataset), config.max_patches)

    pixel_errors: list[np.ndarray] = []
    spearman_values: list[float] = []
    top1_matches = 0

    for dataset_index in range(patch_count):
        sample = dataset[dataset_index]
        source_id = _sample_source_id(sample, dataset_index)
        context = sample["cmykogv_context"].numpy().astype(np.float32)
        lab_ref = sample["lab_center"].numpy().astype(np.float32)
        candidate_contexts = generate_lambda_probe_contexts(
            context,
            lab_ref,
            ppp,
            source_id=source_id,
            config=config,
        )
        probe = _evaluate_one_patch(
            model,
            sample,
            candidate_contexts,
            lab_ref,
            drift_bank,
            device=torch_device,
            config=config,
        )
        pixel_errors.append(probe["pixel_errors"])
        spearman_values.append(float(probe["spearman"]))
        top1_matches += int(probe["top1_match"])

    if patch_count == 0:
        return CandidateProbeMetrics(
            mean_delta_e00=0.0,
            q90_delta_e00=0.0,
            spearman=0.0,
            top1_agreement=0.0,
            ranking_evaluated=False,
            patches_evaluated=0,
            candidates_per_patch=len(config.lambda_values),
            drifts_per_candidate=config.drift_sample_count,
        )

    flat_errors = np.concatenate(pixel_errors, axis=0)
    return CandidateProbeMetrics(
        mean_delta_e00=float(np.mean(flat_errors)) if flat_errors.size else 0.0,
        q90_delta_e00=finite_quantile(flat_errors, 0.90) if flat_errors.size else 0.0,
        spearman=float(np.mean(spearman_values)) if spearman_values else 0.0,
        top1_agreement=float(top1_matches / patch_count),
        ranking_evaluated=True,
        patches_evaluated=patch_count,
        candidates_per_patch=len(config.lambda_values),
        drifts_per_candidate=config.drift_sample_count,
    )


def generate_lambda_probe_contexts(
    cmykogv_context: np.ndarray,
    lab_ref: np.ndarray,
    ppp: PPP,
    *,
    source_id: str,
    config: CandidateProbeConfig = CandidateProbeConfig(),
) -> tuple[np.ndarray, ...]:
    """Generate the five deterministic lambda candidates for one patch context."""
    context = np.asarray(cmykogv_context, dtype=np.float32)
    if context.shape != (32, 32, 7):
        raise ValueError(f"expected cmykogv_context shape (32,32,7), got {context.shape}")
    lab = np.asarray(lab_ref, dtype=np.float32)
    if lab.shape != (16, 16, 3):
        raise ValueError(f"expected lab_ref shape (16,16,3), got {lab.shape}")

    center = _center_patch(context)
    chroma_signal = _lab_to_ogv_signal(lab)
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


def _evaluate_one_patch(
    model: ForwardSurrogateCNN,
    sample: dict[str, Any],
    candidate_contexts: Sequence[np.ndarray],
    lab_ref: np.ndarray,
    drift_bank: Sequence[DriftSample],
    *,
    device: torch.device,
    config: CandidateProbeConfig,
) -> dict[str, Any]:
    contexts: list[np.ndarray] = []
    drift_vectors: list[np.ndarray] = []
    lambda_values: list[float] = []
    teacher_labs: list[np.ndarray] = []
    teacher_risk_by_candidate: list[list[float]] = [[] for _ in candidate_contexts]

    for candidate_index, context in enumerate(candidate_contexts):
        center = _center_patch(context)
        for drift in drift_bank:
            teacher_lab = calibrated_cmykogv_lab(apply_drift(center, drift), anchor_cmykogv=center, anchor_lab=lab_ref)
            teacher_labs.append(teacher_lab)
            teacher_risk_by_candidate[candidate_index].append(float(np.mean(delta_e_00(teacher_lab, lab_ref))))
            contexts.append(context)
            drift_vectors.append(_drift_vector(drift))
            lambda_values.append(float(config.lambda_values[candidate_index]))

    predictions = _predict_probe_labs(
        model,
        sample,
        contexts,
        drift_vectors,
        lambda_values,
        device=device,
        batch_size=config.batch_size,
    )
    teacher = np.stack(teacher_labs, axis=0)
    pixel_errors = delta_e_00(predictions, teacher).reshape(-1)

    surrogate_risk_by_candidate: list[list[float]] = [[] for _ in candidate_contexts]
    offset = 0
    for candidate_index in range(len(candidate_contexts)):
        for _ in drift_bank:
            surrogate_risk_by_candidate[candidate_index].append(float(np.mean(delta_e_00(predictions[offset], lab_ref))))
            offset += 1

    teacher_risk = np.asarray(
        [finite_quantile(np.asarray(values, dtype=np.float32), config.risk_q) for values in teacher_risk_by_candidate],
        dtype=np.float32,
    )
    surrogate_risk = np.asarray(
        [finite_quantile(np.asarray(values, dtype=np.float32), config.risk_q) for values in surrogate_risk_by_candidate],
        dtype=np.float32,
    )
    return {
        "pixel_errors": pixel_errors.astype(np.float32),
        "spearman": _spearman(teacher_risk, surrogate_risk),
        "top1_match": int(np.argmin(teacher_risk) == np.argmin(surrogate_risk)),
    }


def _predict_probe_labs(
    model: ForwardSurrogateCNN,
    sample: dict[str, Any],
    contexts: Sequence[np.ndarray],
    drift_vectors: Sequence[np.ndarray],
    lambda_values: Sequence[float],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    n = len(contexts)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        size = end - start
        pred = model(
            torch.from_numpy(np.stack(contexts[start:end], axis=0)).to(device),
            base_family_index=sample["base_family_index"].reshape(1).repeat(size).to(device),
            ppp_numeric=sample["ppp_numeric"].reshape(1, -1).repeat(size, 1).to(device),
            ppp_override_mask=sample["ppp_override_mask"].reshape(1, -1).repeat(size, 1).to(device),
            structure_index=sample["structure_index"].reshape(1).repeat(size).to(device),
            intent_weights=sample["intent_weights"].reshape(1, -1).repeat(size, 1).to(device),
            intent_raster=sample["intent_raster"].reshape(1, *sample["intent_raster"].shape).repeat(size, 1, 1, 1).to(device),
            lambda_value=torch.tensor(lambda_values[start:end], dtype=torch.float32, device=device),
            drift_vector=torch.from_numpy(np.stack(drift_vectors[start:end], axis=0)).to(device),
        )
        outputs.append(pred.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def _fixed_family_drift_bank(ppp: PPP, config: CandidateProbeConfig) -> list[DriftSample]:
    return sample_drift_bank(
        config.drift_config,
        config.root_seed,
        ppp.base_family,
        ppp.hash,
        "surrogate_quality_gate_family_bank",
        sample_count=config.drift_sample_count,
    )


def _patch_count(dataset_len: int, max_patches: int | None) -> int:
    if max_patches is None:
        return dataset_len
    if max_patches < 0:
        raise ValueError("max_patches must be non-negative or None")
    return min(dataset_len, max_patches)


def _sample_source_id(sample: dict[str, Any], dataset_index: int) -> str:
    for key in ("source_id", "target_hash", "drift_hash"):
        value = sample.get(key)
        if value:
            return str(value)
    return f"dataset-index-{dataset_index}"


def _center_patch(context: np.ndarray) -> np.ndarray:
    start = (context.shape[0] - 16) // 2
    return context[start : start + 16, start : start + 16, :]


def _lab_to_ogv_signal(lab: np.ndarray) -> np.ndarray:
    a = np.clip(lab[..., 1] / 128.0, -1.0, 1.0)
    b = np.clip(lab[..., 2] / 128.0, -1.0, 1.0)
    signal = np.empty(lab.shape[:2] + (3,), dtype=np.float32)
    signal[..., 0] = np.clip(0.5 * a + 0.5 * b, 0.0, 1.0)  # orange/warm
    signal[..., 1] = np.clip(-a, 0.0, 1.0)  # green
    signal[..., 2] = np.clip(0.5 * a - 0.5 * b, 0.0, 1.0)  # violet/magenta-blue
    return signal


def _drift_vector(drift: DriftSample) -> np.ndarray:
    return np.concatenate([drift.multipliers.reshape(-1), drift.trc_y[:, 1:-1].reshape(-1)], axis=0).astype(np.float32)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _average_ranks(np.asarray(a, dtype=np.float64))
    rb = _average_ranks(np.asarray(b, dtype=np.float64))
    ra -= np.mean(ra)
    rb -= np.mean(rb)
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.sum(ra * rb) / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks
