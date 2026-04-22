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
from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset, iter_surrogate_shard_batches
from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN, SurrogateModelConfig
from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig, evaluate_candidate_probe


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


@dataclass(frozen=True)
class SurrogateQualityGateThresholds:
    threshold_mean: float = 3.0
    threshold_q90: float = 5.0
    threshold_spearman: float = 0.80
    threshold_top1: float = 0.80


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    loss_fn = nn.SmoothL1Loss()

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
            pred = _forward_batch(model, batch, device)
            target = batch["lab_center"].to(device, non_blocking=True).permute(0, 3, 1, 2).float()
            loss = loss_fn(pred, target)
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors: list[np.ndarray] = []
    for batch in loader:
        pred = _forward_batch(model, batch, torch.device(device)).detach().cpu().permute(0, 2, 3, 1).numpy()
        target = batch["lab_center"].numpy()
        errors.append(delta_e_00(pred, target).reshape(-1))
    flat = np.concatenate(errors, axis=0) if errors else np.zeros((0,), dtype=np.float32)
    fallback_mean = float(np.mean(flat)) if flat.size else 0.0
    fallback_q90 = finite_quantile(flat, 0.90) if flat.size else 0.0
    probe = evaluate_candidate_probe(model, dataset, device=device, config=candidate_probe_config)
    mean = probe.mean_delta_e00 if probe.ranking_evaluated else fallback_mean
    q90 = probe.q90_delta_e00 if probe.ranking_evaluated else fallback_q90
    metrics = SurrogateQualityMetrics(
        mean_delta_e00=mean,
        q90_delta_e00=q90,
        spearman=probe.spearman,
        top1_agreement=probe.top1_agreement,
        ranking_evaluated=probe.ranking_evaluated,
        probe_patches_evaluated=probe.patches_evaluated,
        probe_candidates_per_patch=probe.candidates_per_patch,
        probe_drifts_per_candidate=probe.drifts_per_candidate,
        passed=(
            mean <= thresholds.threshold_mean
            and q90 <= thresholds.threshold_q90
            and probe.ranking_evaluated
            and probe.spearman >= thresholds.threshold_spearman
            and probe.top1_agreement >= thresholds.threshold_top1
        ),
    )
    return metrics


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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _should_write_progress(interval: int, batches_seen: int) -> bool:
    return interval > 0 and batches_seen % interval == 0


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
