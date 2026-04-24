from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Proposer training requires PyTorch. Install torch to use this module.") from exc

from robustsep_pkg.core.artifact_io import read_json, sha256_file, write_json
from robustsep_pkg.models.proposer.data import ProposerTrainingDataset, iter_proposer_shard_batches
from robustsep_pkg.models.proposer.losses import ProposerLossWeights, lambda_monotonicity_hinge, proposer_vae_loss
from robustsep_pkg.models.proposer.model import ConditionalVAEProposer, ProposerModelConfig
from robustsep_pkg.targets.solver import _LAB_JACOBIAN


@dataclass(frozen=True)
class ProposerTrainingConfig:
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
class ProposerOptimizationConfig:
    loss_weights: ProposerLossWeights = field(default_factory=ProposerLossWeights)
    kl_warmup_fraction: float = 0.10
    lambda_values: tuple[float, ...] = (0.1, 0.5, 0.9)
    recon_lambda_index: int = 1
    monotonicity_margin: float = 0.0
    appearance_mode: str = "none"


@dataclass(frozen=True)
class ProposerTrainingResult:
    checkpoint_path: str
    report_path: str
    config_path: str
    progress_path: str
    manifest_path: str
    manifest_sha256: str
    dataset_examples: int
    train_loss: float
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
            "device": self.device,
        }


def train_proposer(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    training_config: ProposerTrainingConfig = ProposerTrainingConfig(),
    model_config: ProposerModelConfig = ProposerModelConfig(),
    optimization_config: ProposerOptimizationConfig = ProposerOptimizationConfig(),
) -> ProposerTrainingResult:
    torch.manual_seed(training_config.seed)
    device = _resolve_device(training_config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest_path = Path(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    dataset = ProposerTrainingDataset(manifest_path, model_config=model_config)
    model = ConditionalVAEProposer(model_config).to(device)
    if training_config.initial_checkpoint:
        checkpoint = torch.load(training_config.initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "proposer_checkpoint.pth"
    report_path = out_path / "proposer_training_report.json"
    config_path = out_path / "proposer_training_config.json"
    progress_path = out_path / "proposer_training_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    config_payload = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_version": manifest.get("manifest_version"),
        "dataset_examples": len(dataset),
        "dataset_shards": len(manifest.get("shards", [])),
        "training_config": asdict(training_config),
        "model_config": asdict(model_config),
        "optimization_config": {
            "loss_weights": asdict(optimization_config.loss_weights),
            "kl_warmup_fraction": optimization_config.kl_warmup_fraction,
            "lambda_values": list(optimization_config.lambda_values),
            "recon_lambda_index": optimization_config.recon_lambda_index,
            "monotonicity_margin": optimization_config.monotonicity_margin,
            "appearance_mode": optimization_config.appearance_mode,
        },
        "training_loader": {
            "mode": "shard_stream",
            "shuffle_shards": True,
            "shuffle_within_shard": True,
            "num_workers": 0,
            "ignored_requested_num_workers": training_config.num_workers,
        },
    }
    write_json(config_path, config_payload)

    total_batches_per_epoch = int(np.ceil(len(dataset) / max(training_config.batch_size, 1)))
    total_steps = max(1, total_batches_per_epoch * training_config.epochs)
    losses: list[float] = []
    batches_seen = 0
    examples_seen = 0

    for epoch in range(training_config.epochs):
        for batch in iter_proposer_shard_batches(
            manifest_path,
            batch_size=training_config.batch_size,
            model_config=model_config,
            seed=training_config.seed,
            epoch=epoch,
            shuffle_shards=True,
            shuffle_within_shard=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            step_index = batches_seen + 1
            loss, pieces = _loss_for_batch(model, batch, device, optimization_config, step_index=step_index, total_steps=total_steps)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            batch_size = int(batch["rgb_patch"].shape[0])
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
                        **pieces,
                    },
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "optimization_config": {
                "loss_weights": asdict(optimization_config.loss_weights),
                "kl_warmup_fraction": optimization_config.kl_warmup_fraction,
                "lambda_values": list(optimization_config.lambda_values),
                "recon_lambda_index": optimization_config.recon_lambda_index,
                "monotonicity_margin": optimization_config.monotonicity_margin,
                "appearance_mode": optimization_config.appearance_mode,
            },
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "dataset_examples": len(dataset),
        },
        checkpoint_path,
    )
    result = ProposerTrainingResult(
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
        config_path=str(config_path),
        progress_path=str(progress_path),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        dataset_examples=len(dataset),
        train_loss=float(np.mean(losses)) if losses else 0.0,
        device=str(device),
    )
    report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _loss_for_batch(
    model: ConditionalVAEProposer,
    batch: dict[str, Any],
    device: torch.device | str,
    optimization_config: ProposerOptimizationConfig,
    *,
    step_index: int,
    total_steps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(optimization_config.lambda_values) < 1:
        raise ValueError("lambda_values must contain at least one value")
    recon_index = int(optimization_config.recon_lambda_index)
    if recon_index < 0 or recon_index >= len(optimization_config.lambda_values):
        raise ValueError("recon_lambda_index is out of range for lambda_values")

    rgb = batch["rgb_patch"].to(device)
    alpha = batch["alpha"].to(device)
    target = batch["target_cmykogv"].to(device).permute(0, 3, 1, 2).float()
    lab_ref = batch["lab_ref_center"].to(device)

    lambda_mid = torch.full((rgb.shape[0],), float(optimization_config.lambda_values[recon_index]), dtype=torch.float32, device=device)
    cond_mid = model.conditioner(
        base_family_index=batch["base_family_index"].to(device),
        ppp_numeric=batch["ppp_numeric"].to(device),
        ppp_override_mask=batch["ppp_override_mask"].to(device),
        structure_index=batch["structure_index"].to(device),
        intent_weights=batch["intent_weights"].to(device),
        intent_raster=batch["intent_raster"].to(device),
        lambda_value=lambda_mid,
    )
    proposer_input = model.build_input(rgb, alpha, batch["intent_weights"].to(device), batch["intent_raster"].to(device))
    mean, logvar = model.encode(proposer_input, cond_mid)
    latent = model.reparameterize(mean, logvar)

    outputs_by_lambda: list[torch.Tensor] = []
    for lambda_value in optimization_config.lambda_values:
        lam = torch.full((rgb.shape[0],), float(lambda_value), dtype=torch.float32, device=device)
        cond = model.conditioner(
            base_family_index=batch["base_family_index"].to(device),
            ppp_numeric=batch["ppp_numeric"].to(device),
            ppp_override_mask=batch["ppp_override_mask"].to(device),
            structure_index=batch["structure_index"].to(device),
            intent_weights=batch["intent_weights"].to(device),
            intent_raster=batch["intent_raster"].to(device),
            lambda_value=lam,
        )
        outputs_by_lambda.append(model.decode(latent, cond))
    recon_pred = outputs_by_lambda[recon_index]

    appearance_loss = None
    if optimization_config.loss_weights.appearance > 0.0 and optimization_config.appearance_mode == "teacher_proxy":
        appearance_loss = _appearance_teacher_proxy_loss(recon_pred, target, lab_ref)

    kl_scale = _kl_warmup_scale(step_index, total_steps, optimization_config.kl_warmup_fraction)
    warm_weights = ProposerLossWeights(
        ink=optimization_config.loss_weights.ink,
        kl=optimization_config.loss_weights.kl * kl_scale,
        appearance=optimization_config.loss_weights.appearance,
        lambda_monotonicity=optimization_config.loss_weights.lambda_monotonicity,
    )
    pieces = proposer_vae_loss(recon_pred, target, mean, logvar, weights=warm_weights, appearance_loss=appearance_loss)
    monotonicity = lambda_monotonicity_hinge(outputs_by_lambda, margin=optimization_config.monotonicity_margin)
    total = pieces["total"] + warm_weights.lambda_monotonicity * monotonicity
    return total, {
        "ink_loss": float(pieces["ink"].detach().cpu()),
        "kl_loss": float(pieces["kl"].detach().cpu()),
        "appearance_loss": float(pieces["appearance"].detach().cpu()),
        "monotonicity_loss": float(monotonicity.detach().cpu()),
        "kl_scale": float(kl_scale),
    }


def _appearance_teacher_proxy_loss(prediction: torch.Tensor, target: torch.Tensor, lab_ref: torch.Tensor) -> torch.Tensor:
    pred_nhwc = prediction.permute(0, 2, 3, 1).float()
    target_nhwc = target.permute(0, 2, 3, 1).float()
    jac = torch.as_tensor(_LAB_JACOBIAN, dtype=pred_nhwc.dtype, device=pred_nhwc.device)
    pred_lab = lab_ref.float() + torch.einsum("nhwc,lc->nhwl", pred_nhwc - target_nhwc, jac)
    return torch.nn.functional.smooth_l1_loss(pred_lab, lab_ref.float())


def _kl_warmup_scale(step_index: int, total_steps: int, warmup_fraction: float) -> float:
    if warmup_fraction <= 0.0:
        return 1.0
    warmup_steps = max(1, int(round(float(total_steps) * min(max(warmup_fraction, 0.0), 1.0))))
    return min(1.0, float(step_index) / float(warmup_steps))


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _should_write_progress(interval: int, batches_seen: int) -> bool:
    return interval > 0 and batches_seen % interval == 0


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
