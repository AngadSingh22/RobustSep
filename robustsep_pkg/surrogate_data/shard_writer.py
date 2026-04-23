from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from robustsep_pkg.core.artifact_io import canonical_json_hash, sha256_file, write_json, write_jsonl
from robustsep_pkg.models.conditioning.drift import apply_drift
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data.candidates import LambdaCandidateConfig, center_patch, generate_lambda_candidate_contexts
from robustsep_pkg.targets.teacher import calibrated_cmykogv_lab


@dataclass(frozen=True)
class SurrogateShardWriterConfig:
    shard_size: int = 4096
    run_id: str = "surrogate_training_shards"
    manifest_version: str = "surrogate-shards-v2"
    candidate_config: LambdaCandidateConfig = field(default_factory=LambdaCandidateConfig)


@dataclass(frozen=True)
class SurrogateShardSummary:
    manifest_path: str
    output_dir: str
    total_examples: int
    num_shards: int
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "output_dir": self.output_dir,
            "total_examples": self.total_examples,
            "num_shards": self.num_shards,
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True)
class _SurrogateRow:
    cmykogv_context: np.ndarray
    lab_ref_center: np.ndarray
    teacher_lab_nominal: np.ndarray
    teacher_lab_drifted: np.ndarray
    lab_center: np.ndarray
    intent_raster: np.ndarray
    intent_weights: np.ndarray
    candidate_type_index: int
    lambda_value: float
    drift_multipliers: np.ndarray
    drift_trc_x: np.ndarray
    drift_trc_y: np.ndarray
    metadata: dict[str, Any]


def write_surrogate_training_shards(
    generated_records: Iterable[Any],
    out_dir: str | Path,
    ppp: PPP,
    *,
    config: SurrogateShardWriterConfig = SurrogateShardWriterConfig(),
) -> SurrogateShardSummary:
    """Write durable forward-surrogate training shards from generated targets."""
    if config.shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {config.shard_size}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    buffer: list[_SurrogateRow] = []
    total = 0
    shard_index = 0

    for row in _iter_surrogate_rows(generated_records, ppp, config=config):
        buffer.append(row)
        if len(buffer) >= config.shard_size:
            shards.append(_write_one_shard(out_path, shard_index, buffer))
            total += len(buffer)
            shard_index += 1
            buffer = []

    if buffer:
        shards.append(_write_one_shard(out_path, shard_index, buffer))
        total += len(buffer)

    manifest = {
        "manifest_version": config.manifest_version,
        "schema": {
            "version": 2,
            "output_mode": "local_delta_lab",
            "lab_ref_field": "lab_ref_center",
            "nominal_teacher_field": "teacher_lab_nominal",
            "drifted_teacher_field": "teacher_lab_drifted",
            "candidate_distribution": "lambda_probe",
            "candidate_lambda_values": list(config.candidate_config.lambda_values),
            "obsolete_prior_versions": ["surrogate-shards-v1"],
        },
        "run_id": config.run_id,
        "created_unix": time.time(),
        "out_dir": str(out_path),
        "total_examples": total,
        "num_shards": len(shards),
        "ppp_hash": ppp.hash,
        "ppp": ppp.to_dict(),
        "shards": shards,
    }
    manifest_hash = canonical_json_hash({k: v for k, v in manifest.items() if k != "created_unix"})
    manifest["manifest_hash"] = manifest_hash
    manifest_path = out_path / "surrogate_training_manifest.json"
    write_json(manifest_path, manifest)
    return SurrogateShardSummary(
        manifest_path=str(manifest_path),
        output_dir=str(out_path),
        total_examples=total,
        num_shards=len(shards),
        manifest_hash=manifest_hash,
    )


def _iter_surrogate_rows(
    generated_records: Iterable[Any],
    ppp: PPP,
    *,
    config: SurrogateShardWriterConfig,
) -> Iterator[_SurrogateRow]:
    for target_record in generated_records:
        for example_index, example in enumerate(target_record.surrogate_examples):
            intent_weights = np.asarray(
                [
                    float(example.intent_weights.get("brand", 0.0)),
                    float(example.intent_weights.get("gradient", 0.0)),
                    float(example.intent_weights.get("flat", 0.0)),
                ],
                dtype=np.float32,
            )
            source_id = str(example.metadata.get("source_id") or target_record.target_result.manifest_record.target_hash)
            candidate_contexts = generate_lambda_candidate_contexts(
                np.asarray(example.cmykogv_context, dtype=np.float32),
                np.asarray(example.lab_center, dtype=np.float32),
                ppp,
                source_id=source_id,
                config=config.candidate_config,
            )
            for candidate_index, candidate_context in enumerate(candidate_contexts):
                lambda_value = float(config.candidate_config.lambda_values[candidate_index])
                lab_ref = np.asarray(example.lab_center, dtype=np.float32)
                center = center_patch(candidate_context)
                teacher_nominal = calibrated_cmykogv_lab(center, anchor_cmykogv=center, anchor_lab=lab_ref)
                teacher_drifted = _teacher_lab_for_context(candidate_context, example)
                metadata = {
                    "family": target_record.family_name,
                    "split": target_record.split,
                    "source_path": target_record.sample_record.source_path,
                    "x": target_record.sample_record.x,
                    "y": target_record.sample_record.y,
                    "source_id": source_id,
                    "target_hash": target_record.target_result.manifest_record.target_hash,
                    "initial_hash": target_record.target_result.manifest_record.initial_hash,
                    "ppp_hash": ppp.hash,
                    "structure_token": example.structure_token,
                    "candidate_type": "lambda_probe",
                    "candidate_type_index": 0,
                    "candidate_index": candidate_index,
                    "lambda_value": lambda_value,
                    "example_index": example_index,
                    "drift_hash": _drift_hash(example.drift),
                }
                yield _SurrogateRow(
                    cmykogv_context=np.asarray(candidate_context, dtype=np.float32),
                    lab_ref_center=lab_ref,
                    teacher_lab_nominal=teacher_nominal,
                    teacher_lab_drifted=teacher_drifted,
                    lab_center=lab_ref,
                    intent_raster=np.asarray(example.intent_raster, dtype=np.float32),
                    intent_weights=intent_weights,
                    candidate_type_index=0,
                    lambda_value=lambda_value,
                    drift_multipliers=np.asarray(example.drift.multipliers, dtype=np.float32),
                    drift_trc_x=np.asarray(example.drift.trc_x, dtype=np.float32),
                    drift_trc_y=np.asarray(example.drift.trc_y, dtype=np.float32),
                    metadata=metadata,
                )


def _write_one_shard(out_dir: Path, shard_index: int, rows: list[_SurrogateRow]) -> dict[str, Any]:
    stem = f"surrogate-{shard_index:05d}"
    npz_path = out_dir / f"{stem}.npz"
    jsonl_path = out_dir / f"{stem}.jsonl"
    np.savez_compressed(
        npz_path,
        cmykogv_context=np.stack([r.cmykogv_context for r in rows], axis=0),
        lab_ref_center=np.stack([r.lab_ref_center for r in rows], axis=0),
        teacher_lab_nominal=np.stack([r.teacher_lab_nominal for r in rows], axis=0),
        teacher_lab_drifted=np.stack([r.teacher_lab_drifted for r in rows], axis=0),
        lab_center=np.stack([r.lab_center for r in rows], axis=0),
        intent_raster=np.stack([r.intent_raster for r in rows], axis=0),
        intent_weights=np.stack([r.intent_weights for r in rows], axis=0),
        candidate_type_index=np.asarray([r.candidate_type_index for r in rows], dtype=np.int64),
        lambda_value=np.asarray([r.lambda_value for r in rows], dtype=np.float32),
        drift_multipliers=np.stack([r.drift_multipliers for r in rows], axis=0),
        drift_trc_x=np.stack([r.drift_trc_x for r in rows], axis=0),
        drift_trc_y=np.stack([r.drift_trc_y for r in rows], axis=0),
    )
    write_jsonl(jsonl_path, [r.metadata | {"shard_index": i} for i, r in enumerate(rows)])
    return {
        "npz": str(npz_path),
        "jsonl": str(jsonl_path),
        "count": len(rows),
        "npz_sha256": sha256_file(npz_path),
        "jsonl_sha256": sha256_file(jsonl_path),
    }


def _teacher_lab_for_context(context: np.ndarray, example: Any) -> np.ndarray:
    candidate_context = np.asarray(context, dtype=np.float32)
    drifted = apply_drift(candidate_context, example.drift)
    lab = np.asarray(example.lab_center, dtype=np.float32)
    start = (candidate_context.shape[0] - 16) // 2
    anchor = candidate_context[start : start + 16, start : start + 16, :]
    values = drifted[start : start + 16, start : start + 16, :]
    return calibrated_cmykogv_lab(values, anchor_cmykogv=anchor, anchor_lab=lab)


def _drift_hash(drift: Any) -> str:
    return canonical_json_hash(
        {
            "multipliers": np.asarray(drift.multipliers).tolist(),
            "trc_x": np.asarray(drift.trc_x).tolist(),
            "trc_y": np.asarray(drift.trc_y).tolist(),
        }
    )
