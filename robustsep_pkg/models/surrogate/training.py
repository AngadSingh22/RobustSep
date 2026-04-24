from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("Surrogate training requires PyTorch. Install torch to use this module.") from exc

from robustsep_pkg.core.artifact_io import read_json, sha256_file, write_json
from robustsep_pkg.eval.metrics import delta_e_00, finite_quantile
from robustsep_pkg.models.conditioning.drift import DriftSample, apply_drift
from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset, iter_surrogate_shard_batches
from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN, SurrogateModelConfig
from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig, evaluate_candidate_probe
from robustsep_pkg.targets.teacher import calibrated_cmykogv_lab


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 20260422
    device: str = "auto"
    progress_interval_batches: int = 100
    initial_checkpoint: str | None = None


@dataclass(frozen=True)
class SurrogateLossConfig:
    target_mode: str = "teacher_delta"
    hard_pixel_weight: float = 0.0
    hard_pixel_quantile: float = 0.90


@dataclass(frozen=True)
class SurrogateQualityGateThresholds:
    threshold_mean: float = 3.0
    threshold_q90: float = 5.0
    threshold_spearman: float = 0.80
    threshold_top1: float = 0.80
    threshold_mean_regret: float = 0.25
    threshold_q90_regret: float = 1.0


@dataclass(frozen=True)
class SurrogateQualityMetrics:
    mean_delta_e00: float
    q90_delta_e00: float
    spearman: float
    top1_agreement: float
    ranking_evaluated: bool
    probe_patches_evaluated: int
    probe_candidates_per_patch: int
    probe_drifts_per_candidate: int
    passed: bool
    mean_regret_delta_e00: float = 0.0
    q90_regret_delta_e00: float = 0.0
    mean_teacher_margin_delta_e00: float = 0.0
    q90_teacher_margin_delta_e00: float = 0.0
    strict_spearman: float = 0.0
    strict_top1_agreement: float = 0.0
    tie_patch_fraction: float = 0.0

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SurrogateTrainingResult:
    checkpoint_path: str
    report_path: str
    config_path: str
    progress_path: str
    manifest_path: str
    manifest_sha256: str
    dataset_examples: int
    train_loss: float
    quality: SurrogateQualityMetrics
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "report_path": self.report_path,
            "config_path": self.config_path,
            "progress_path": self.progress_path,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "dataset_examples": self.dataset_examples,
            "train_loss": self.train_loss,
            "quality": self.quality.to_dict(),
            "device": self.device,
        }


def train_surrogate(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    training_config: SurrogateTrainingConfig = SurrogateTrainingConfig(),
    model_config: SurrogateModelConfig = SurrogateModelConfig(),
    gate_thresholds: SurrogateQualityGateThresholds = SurrogateQualityGateThresholds(),
    candidate_probe_config: CandidateProbeConfig = CandidateProbeConfig(),
    loss_config: SurrogateLossConfig = SurrogateLossConfig(),
) -> SurrogateTrainingResult:
    torch.manual_seed(training_config.seed)
    device = _resolve_device(training_config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest_path = Path(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    dataset = SurrogateTrainingDataset(manifest_path, model_config=model_config)
    model = ForwardSurrogateCNN(model_config).to(device)
    if training_config.initial_checkpoint:
        checkpoint = torch.load(training_config.initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "surrogate_checkpoint.pth"
    report_path = out_path / "surrogate_training_report.json"
    config_path = out_path / "surrogate_training_config.json"
    progress_path = out_path / "surrogate_training_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    config_payload = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_version": manifest.get("manifest_version"),
        "dataset_examples": len(dataset),
        "dataset_shards": len(manifest.get("shards", [])),
        "training_config": asdict(training_config),
        "model_config": asdict(model_config),
        "gate_thresholds": asdict(gate_thresholds),
        "candidate_probe_config": asdict(candidate_probe_config),
        "loss_config": asdict(loss_config),
        "training_loader": {
            "mode": "shard_stream",
            "shuffle_shards": True,
            "shuffle_within_shard": True,
            "num_workers": 0,
            "ignored_requested_num_workers": training_config.num_workers,
        },
    }
    write_json(config_path, config_payload)

    model.train()
    losses: list[float] = []
    examples_seen = 0
    batches_seen = 0
    for epoch in range(training_config.epochs):
        for batch in iter_surrogate_shard_batches(
            manifest_path,
            batch_size=training_config.batch_size,
            model_config=model_config,
            seed=training_config.seed,
            epoch=epoch,
            shuffle_shards=True,
            shuffle_within_shard=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(model, batch, device, loss_config)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            batch_size = int(batch["lab_center"].shape[0])
            examples_seen += batch_size
            batches_seen += 1
            if _should_write_progress(training_config.progress_interval_batches, batches_seen):
                _append_progress(
                    progress_path,
                    {
                        "phase": "train",
                        "epoch": epoch,
                        "batches_seen": batches_seen,
                        "examples_seen": examples_seen,
                        "latest_loss": loss_value,
                        "mean_loss": float(np.mean(losses)),
                    },
                )

    _append_progress(
        progress_path,
        {
            "phase": "quality_gate",
            "batches_seen": batches_seen,
            "examples_seen": examples_seen,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "probe_max_patches": candidate_probe_config.max_patches,
            "probe_drift_samples": candidate_probe_config.drift_sample_count,
        },
    )
    quality = evaluate_surrogate_quality(
        model,
        dataset,
        device=device,
        thresholds=gate_thresholds,
        batch_size=training_config.batch_size,
        candidate_probe_config=candidate_probe_config,
    )
    _append_progress(
        progress_path,
        {
            "phase": "done",
            "epochs": training_config.epochs,
            "batches_seen": batches_seen,
            "examples_seen": examples_seen,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "quality": quality.to_dict(),
        },
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "gate_thresholds": asdict(gate_thresholds),
            "candidate_probe_config": asdict(candidate_probe_config),
            "loss_config": asdict(loss_config),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "dataset_examples": len(dataset),
        },
        checkpoint_path,
    )
    result = SurrogateTrainingResult(
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
        config_path=str(config_path),
        progress_path=str(progress_path),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        dataset_examples=len(dataset),
        train_loss=float(np.mean(losses)) if losses else 0.0,
        quality=quality,
        device=str(device),
    )
    report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def diagnose_surrogate_quality(
    quality: SurrogateQualityMetrics,
    thresholds: SurrogateQualityGateThresholds = SurrogateQualityGateThresholds(),
) -> dict[str, Any]:
    """Classify quality-gate failures into actionable training adjustments."""
    failures = {
        "mean_delta_e00": quality.mean_delta_e00 > thresholds.threshold_mean,
        "q90_delta_e00": quality.q90_delta_e00 > thresholds.threshold_q90,
        "spearman": quality.spearman < thresholds.threshold_spearman,
        "top1_agreement": quality.top1_agreement < thresholds.threshold_top1,
        "mean_regret_delta_e00": quality.mean_regret_delta_e00 > thresholds.threshold_mean_regret,
        "q90_regret_delta_e00": quality.q90_regret_delta_e00 > thresholds.threshold_q90_regret,
    }
    actions: list[str] = []
    if failures["mean_delta_e00"] or failures["q90_delta_e00"]:
        actions.append("fix_teacher_schema_normalization_or_data")
    if failures["spearman"] or failures["top1_agreement"]:
        if not failures["mean_regret_delta_e00"] and not failures["q90_regret_delta_e00"]:
            actions.append("relax_margin_tied_ranking_gate")
        else:
            actions.append("add_candidate_distribution_training_and_rank_loss")
    return {
        "passed": quality.passed,
        "failures": failures,
        "recommended_actions": actions,
    }


@torch.no_grad()
def evaluate_surrogate_quality(
    model: ForwardSurrogateCNN,
    dataset: SurrogateTrainingDataset,
    *,
    device: torch.device | str,
    thresholds: SurrogateQualityGateThresholds = SurrogateQualityGateThresholds(),
    batch_size: int = 32,
    candidate_probe_config: CandidateProbeConfig = CandidateProbeConfig(),
) -> SurrogateQualityMetrics:
    model.eval()
    probe = evaluate_candidate_probe(model, dataset, device=device, config=candidate_probe_config)
    if probe.ranking_evaluated:
        mean = probe.mean_delta_e00
        q90 = probe.q90_delta_e00
    else:
        mean, q90 = _evaluate_delta_e_fallback(model, dataset, device=device, batch_size=batch_size)
    metrics = SurrogateQualityMetrics(
        mean_delta_e00=mean,
        q90_delta_e00=q90,
        spearman=probe.spearman,
        top1_agreement=probe.top1_agreement,
        ranking_evaluated=probe.ranking_evaluated,
        probe_patches_evaluated=probe.patches_evaluated,
        probe_candidates_per_patch=probe.candidates_per_patch,
        probe_drifts_per_candidate=probe.drifts_per_candidate,
        mean_regret_delta_e00=probe.mean_regret_delta_e00,
        q90_regret_delta_e00=probe.q90_regret_delta_e00,
        mean_teacher_margin_delta_e00=probe.mean_teacher_margin_delta_e00,
        q90_teacher_margin_delta_e00=probe.q90_teacher_margin_delta_e00,
        strict_spearman=probe.strict_spearman,
        strict_top1_agreement=probe.strict_top1_agreement,
        tie_patch_fraction=probe.tie_patch_fraction,
        passed=(
            mean <= thresholds.threshold_mean
            and q90 <= thresholds.threshold_q90
            and probe.ranking_evaluated
            and probe.spearman >= thresholds.threshold_spearman
            and probe.top1_agreement >= thresholds.threshold_top1
            and probe.mean_regret_delta_e00 <= thresholds.threshold_mean_regret
            and probe.q90_regret_delta_e00 <= thresholds.threshold_q90_regret
        ),
    )
    return metrics


@torch.no_grad()
def _evaluate_delta_e_fallback(
    model: ForwardSurrogateCNN,
    dataset: SurrogateTrainingDataset,
    *,
    device: torch.device | str,
    batch_size: int,
) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors: list[np.ndarray] = []
    for batch in loader:
        pred_raw = _forward_batch(model, batch, torch.device(device)).detach().cpu().permute(0, 2, 3, 1).numpy()
        if _batch_schema_version(batch) >= 2:
            pred = pred_raw + batch["lab_ref_center"].numpy()
            target = batch["teacher_lab_drifted"].numpy()
        else:
            pred = pred_raw
            target = batch["lab_center"].numpy()
        errors.append(delta_e_00(pred, target).reshape(-1))
    flat = np.concatenate(errors, axis=0) if errors else np.zeros((0,), dtype=np.float32)
    return (float(np.mean(flat)) if flat.size else 0.0, finite_quantile(flat, 0.90) if flat.size else 0.0)


def _target_batch(batch: dict[str, Any], device: torch.device | str, loss_config: SurrogateLossConfig) -> torch.Tensor:
    if loss_config.target_mode == "teacher_delta":
        return _lab_delta_target_batch(batch, "teacher_lab_drifted", device)
    if loss_config.target_mode == "lab_anchor":
        return batch["lab_center"].to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    if loss_config.target_mode != "teacher_proxy":
        raise ValueError(f"unknown surrogate target_mode: {loss_config.target_mode}")
    return _teacher_proxy_target_batch(batch, device)


def _loss_for_batch(
    model: ForwardSurrogateCNN,
    batch: dict[str, Any],
    device: torch.device | str,
    loss_config: SurrogateLossConfig,
) -> torch.Tensor:
    if loss_config.target_mode != "teacher_delta":
        pred = _forward_batch(model, batch, device)
        target = _target_batch(batch, device, loss_config)
        return _surrogate_loss(pred, target, loss_config)

    pred_drifted = _forward_batch(model, batch, device)
    target_drifted = _lab_delta_target_batch(batch, "teacher_lab_drifted", device)
    drifted_loss = _surrogate_loss(pred_drifted, target_drifted, loss_config)

    if "teacher_lab_nominal" not in batch:
        return drifted_loss
    nominal_batch = dict(batch)
    nominal_batch["drift_vector"] = _identity_drift_vector_batch(batch)
    pred_nominal = _forward_batch(model, nominal_batch, device)
    target_nominal = _lab_delta_target_batch(batch, "teacher_lab_nominal", device)
    nominal_loss = _surrogate_loss(pred_nominal, target_nominal, loss_config)
    return 0.5 * (drifted_loss + nominal_loss)


def _lab_delta_target_batch(batch: dict[str, Any], target_key: str, device: torch.device | str) -> torch.Tensor:
    ref = batch["lab_ref_center"].to(device, non_blocking=True).float()
    target = batch[target_key].to(device, non_blocking=True).float()
    return (target - ref).permute(0, 3, 1, 2).float()


def _teacher_proxy_target_batch(batch: dict[str, Any], device: torch.device | str) -> torch.Tensor:
    context = batch["cmykogv_context"].detach().cpu().numpy().astype(np.float32)
    lab_anchor = batch["lab_center"].detach().cpu().numpy().astype(np.float32)
    multipliers = batch["drift_multipliers"].detach().cpu().numpy().astype(np.float32)
    trc_x = batch["drift_trc_x"].detach().cpu().numpy().astype(np.float32)
    trc_y = batch["drift_trc_y"].detach().cpu().numpy().astype(np.float32)
    start = (context.shape[1] - 16) // 2
    center = context[:, start : start + 16, start : start + 16, :]
    targets: list[np.ndarray] = []
    for idx in range(center.shape[0]):
        drift = DriftSample(multipliers=multipliers[idx], trc_x=trc_x[idx], trc_y=trc_y[idx])
        drifted = apply_drift(center[idx], drift)
        targets.append(calibrated_cmykogv_lab(drifted, anchor_cmykogv=center[idx], anchor_lab=lab_anchor[idx]))
    target = np.stack(targets, axis=0).astype(np.float32)
    return torch.from_numpy(target).to(device, non_blocking=True).permute(0, 3, 1, 2).float()


def _surrogate_loss(pred: torch.Tensor, target: torch.Tensor, loss_config: SurrogateLossConfig) -> torch.Tensor:
    per_channel = nn.functional.smooth_l1_loss(pred, target, reduction="none")
    per_pixel = per_channel.mean(dim=1)
    if loss_config.hard_pixel_weight <= 0.0:
        return per_pixel.mean()
    q = min(max(float(loss_config.hard_pixel_quantile), 0.0), 1.0)
    thresholds = torch.quantile(per_pixel.detach().flatten(1), q, dim=1).reshape(-1, 1, 1)
    weights = 1.0 + float(loss_config.hard_pixel_weight) * (per_pixel.detach() >= thresholds).float()
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


def _forward_batch(model: ForwardSurrogateCNN, batch: dict[str, Any], device: torch.device | str) -> torch.Tensor:
    device = torch.device(device)
    return model(
        batch["cmykogv_context"].to(device),
        base_family_index=batch["base_family_index"].to(device),
        ppp_numeric=batch["ppp_numeric"].to(device),
        ppp_override_mask=batch["ppp_override_mask"].to(device),
        structure_index=batch["structure_index"].to(device),
        intent_weights=batch["intent_weights"].to(device),
        intent_raster=batch["intent_raster"].to(device),
        lambda_value=batch["lambda_value"].to(device),
        drift_vector=batch["drift_vector"].to(device),
    )


def _identity_drift_vector_batch(batch: dict[str, Any]) -> torch.Tensor:
    drift_vector = batch["drift_vector"]
    size = int(drift_vector.shape[0])
    multipliers = torch.ones((size, 7), dtype=drift_vector.dtype, device=drift_vector.device)
    trc_x = batch["drift_trc_x"].to(device=drift_vector.device, dtype=drift_vector.dtype)
    interiors = trc_x[:, 1:-1].reshape(size, 1, -1).repeat(1, 7, 1).reshape(size, -1)
    return torch.cat([multipliers, interiors], dim=1)


def _batch_schema_version(batch: dict[str, Any]) -> int:
    version = batch.get("schema_version", 1)
    if isinstance(version, torch.Tensor):
        return int(version.flatten()[0].item())
    return int(version)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _should_write_progress(interval: int, batches_seen: int) -> bool:
    return interval > 0 and batches_seen % interval == 0


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
