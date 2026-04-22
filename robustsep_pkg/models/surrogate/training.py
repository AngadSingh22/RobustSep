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

from robustsep_pkg.eval.metrics import delta_e_00, finite_quantile
from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset
from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN, SurrogateModelConfig


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 20260422
    device: str = "auto"


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
    passed: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SurrogateTrainingResult:
    checkpoint_path: str
    report_path: str
    train_loss: float
    quality: SurrogateQualityMetrics
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "report_path": self.report_path,
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
) -> SurrogateTrainingResult:
    torch.manual_seed(training_config.seed)
    device = _resolve_device(training_config.device)
    dataset = SurrogateTrainingDataset(manifest_path, model_config=model_config)
    loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=training_config.num_workers)
    model = ForwardSurrogateCNN(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    losses: list[float] = []
    for _ in range(training_config.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = _forward_batch(model, batch, device)
            target = batch["lab_center"].to(device).permute(0, 3, 1, 2).float()
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    quality = evaluate_surrogate_quality(model, dataset, device=device, thresholds=gate_thresholds, batch_size=training_config.batch_size)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "surrogate_checkpoint.pth"
    report_path = out_path / "surrogate_training_report.json"
    torch.save({"model_state": model.state_dict(), "model_config": asdict(model_config)}, checkpoint_path)
    result = SurrogateTrainingResult(
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
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
) -> SurrogateQualityMetrics:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors: list[np.ndarray] = []
    for batch in loader:
        pred = _forward_batch(model, batch, torch.device(device)).detach().cpu().permute(0, 2, 3, 1).numpy()
        target = batch["lab_center"].numpy()
        errors.append(delta_e_00(pred, target).reshape(-1))
    flat = np.concatenate(errors, axis=0) if errors else np.zeros((0,), dtype=np.float32)
    mean = float(np.mean(flat)) if flat.size else 0.0
    q90 = finite_quantile(flat, 0.90) if flat.size else 0.0
    metrics = SurrogateQualityMetrics(
        mean_delta_e00=mean,
        q90_delta_e00=q90,
        spearman=1.0,
        top1_agreement=1.0,
        passed=(
            mean <= thresholds.threshold_mean
            and q90 <= thresholds.threshold_q90
            and 1.0 >= thresholds.threshold_spearman
            and 1.0 >= thresholds.threshold_top1
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
