from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from robustsep_pkg.core.artifact_io import canonical_json_hash, sha256_file, write_json, write_jsonl
from robustsep_pkg.models.conditioning.ppp import PPP


@dataclass(frozen=True)
class ProposerShardWriterConfig:
    shard_size: int = 4096
    run_id: str = "proposer_training_shards"
    manifest_version: str = "proposer-shards-v1"


@dataclass(frozen=True)
class ProposerShardSummary:
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
class _ProposerRow:
    rgb_patch: np.ndarray
    alpha: np.ndarray
    lab_ref_center: np.ndarray
    target_cmykogv: np.ndarray
    intent_raster: np.ndarray
    intent_weights: np.ndarray
    metadata: dict[str, Any]


def write_proposer_training_shards(
    generated_records: Iterable[Any],
    out_dir: str | Path,
    ppp: PPP,
    *,
    config: ProposerShardWriterConfig = ProposerShardWriterConfig(),
) -> ProposerShardSummary:
    if config.shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {config.shard_size}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    buffer: list[_ProposerRow] = []
    total = 0
    shard_index = 0

    for record in generated_records:
        buffer.append(_row_from_generated_record(record, ppp))
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
            "version": 1,
            "rgb_field": "rgb_patch",
            "alpha_field": "alpha",
            "lab_ref_field": "lab_ref_center",
            "target_field": "target_cmykogv",
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
    manifest_path = out_path / "proposer_training_manifest.json"
    write_json(manifest_path, manifest)
    return ProposerShardSummary(
        manifest_path=str(manifest_path),
        output_dir=str(out_path),
        total_examples=total,
        num_shards=len(shards),
        manifest_hash=manifest_hash,
    )


def _row_from_generated_record(record: Any, ppp: PPP) -> _ProposerRow:
    intent_weights = record.enriched_sample.intent_weights
    intent_raster = record.enriched_sample.intent_raster
    if intent_weights is None or intent_raster is None:
        raise ValueError("proposer shard generation requires intent weights and intent raster")
    source_id = record.target_result.manifest_record.source_id or canonical_json_hash(
        {
            "family": record.family_name,
            "source_path": record.sample_record.source_path,
            "x": record.sample_record.x,
            "y": record.sample_record.y,
            "local_index": record.local_index,
        }
    )
    metadata = {
        "family": record.family_name,
        "split": record.split,
        "source_path": record.sample_record.source_path,
        "x": record.sample_record.x,
        "y": record.sample_record.y,
        "source_id": source_id,
        "target_hash": record.target_result.manifest_record.target_hash,
        "initial_hash": record.target_result.manifest_record.initial_hash,
        "ppp_hash": ppp.hash,
        "structure_token": record.enriched_sample.structure_token,
    }
    return _ProposerRow(
        rgb_patch=np.asarray(record.enriched_sample.rgb, dtype=np.float32),
        alpha=np.asarray(record.enriched_sample.alpha_effective, dtype=np.float32),
        lab_ref_center=np.asarray(record.enriched_sample.lab, dtype=np.float32),
        target_cmykogv=np.asarray(record.target_result.target_cmykogv, dtype=np.float32),
        intent_raster=np.asarray(intent_raster, dtype=np.float32),
        intent_weights=np.asarray(
            [
                float(intent_weights.get("brand", 0.0)),
                float(intent_weights.get("gradient", 0.0)),
                float(intent_weights.get("flat", 0.0)),
            ],
            dtype=np.float32,
        ),
        metadata=metadata,
    )


def _write_one_shard(out_dir: Path, shard_index: int, rows: list[_ProposerRow]) -> dict[str, Any]:
    stem = f"proposer-{shard_index:05d}"
    npz_path = out_dir / f"{stem}.npz"
    jsonl_path = out_dir / f"{stem}.jsonl"
    np.savez_compressed(
        npz_path,
        rgb_patch=np.stack([row.rgb_patch for row in rows], axis=0),
        alpha=np.stack([row.alpha for row in rows], axis=0),
        lab_ref_center=np.stack([row.lab_ref_center for row in rows], axis=0),
        target_cmykogv=np.stack([row.target_cmykogv for row in rows], axis=0),
        intent_raster=np.stack([row.intent_raster for row in rows], axis=0),
        intent_weights=np.stack([row.intent_weights for row in rows], axis=0),
    )
    write_jsonl(jsonl_path, [row.metadata | {"shard_index": idx} for idx, row in enumerate(rows)])
    return {
        "npz": str(npz_path),
        "jsonl": str(jsonl_path),
        "count": len(rows),
        "npz_sha256": sha256_file(npz_path),
        "jsonl_sha256": sha256_file(jsonl_path),
    }
