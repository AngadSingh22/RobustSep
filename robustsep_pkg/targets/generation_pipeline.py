from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from robustsep_pkg.core.artifact_io import canonical_json, canonical_json_hash, read_json, read_jsonl
from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.data.enrichment import EnrichedSample, EnrichmentConfig, enrich_sample
from robustsep_pkg.data.shard_reader import ShardArrays
from robustsep_pkg.data.shard_record import ShardRecord, ShardSample
from robustsep_pkg.models.conditioning.drift import DriftSample, sample_drift_bank
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data import SurrogateExample, build_surrogate_example
from robustsep_pkg.targets.config import TargetSolverConfig
from robustsep_pkg.targets.generator import TargetGenerationResult, generate_target_from_icc_cmyk


@dataclass(frozen=True)
class TargetGenerationPipelineConfig:
    """Configuration for split-manifest driven target generation."""

    target_solver_config: TargetSolverConfig = field(default_factory=TargetSolverConfig)
    drift_config: DriftConfig = field(default_factory=DriftConfig)
    drift_samples_per_patch: int = 1
    lambda_value: float = 0.5
    recompute_structure: bool = False
    alpha_l_min: float = 10.0
    intent_raster_size: int = 4
    include_surrogate_examples: bool = True


@dataclass(frozen=True)
class GeneratedTargetRecord:
    """One generated target plus enrichment and optional surrogate examples."""

    family_name: str
    split: str
    shard_npz: str
    shard_jsonl: str
    local_index: int
    alpha_policy: str
    sample_record: ShardRecord
    enriched_sample: EnrichedSample
    target_result: TargetGenerationResult
    surrogate_examples: tuple[SurrogateExample, ...] = ()

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "family": self.family_name,
            "split": self.split,
            "shard": {
                "npz": self.shard_npz,
                "jsonl": self.shard_jsonl,
                "local_index": self.local_index,
            },
            "sample": self.sample_record.to_dict(),
            "alpha_policy": self.alpha_policy,
            "enrichment": {
                "structure_token": self.enriched_sample.structure_token,
                "intent_weights": dict(self.enriched_sample.intent_weights or {}),
                "intent_raster_shape": _shape_or_none(self.enriched_sample.intent_raster),
                "alpha_effective_mean": float(np.mean(self.enriched_sample.alpha_effective)),
            },
            "target": self.target_result.to_manifest_dict(),
            "surrogate_examples": [
                {
                    **example.to_metadata(),
                    "drift_index": idx,
                    "drift_hash": _drift_hash(example.drift),
                }
                for idx, example in enumerate(self.surrogate_examples)
            ],
        }


@dataclass(frozen=True)
class TargetGenerationSummary:
    output_jsonl: str
    records_written: int
    split_manifest_version: str
    alpha_policy: str
    ppp_hash: str
    source_weight_policy: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_jsonl": self.output_jsonl,
            "records_written": self.records_written,
            "split_manifest_version": self.split_manifest_version,
            "alpha_policy": self.alpha_policy,
            "ppp_hash": self.ppp_hash,
            "source_weight_policy": self.source_weight_policy,
        }


def load_split_manifest(path_or_manifest: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    """Load and validate a TrainingAdapter v1.1 split manifest."""
    if isinstance(path_or_manifest, dict):
        manifest = path_or_manifest
        manifest_path = None
    else:
        manifest_path = Path(path_or_manifest)
        manifest = read_json(manifest_path)
    _validate_split_manifest_v11(manifest)
    return manifest, manifest_path


def iter_generated_target_records(
    split_manifest: str | Path | dict[str, Any],
    ppp: PPP,
    *,
    root: str | Path | None = None,
    config: TargetGenerationPipelineConfig = TargetGenerationPipelineConfig(),
) -> Iterator[GeneratedTargetRecord]:
    """Stream generated targets from a v1.1 split manifest.

    The data-layer boundary is intentionally narrow: this reads the exported
    manifest, loads each shard with public data APIs, calls enrich_sample with
    the manifest alpha policy, then runs the target and surrogate helpers.
    """
    manifest, manifest_path = load_split_manifest(split_manifest)
    alpha_policy = str(manifest["alpha_policy"])
    root_seed = int(manifest.get("root_seed", 20260422))

    for family in manifest["families"]:
        family_name = str(family["name"])
        split = str(family["split"])
        for shard in family["shards"]:
            npz_path = _resolve_manifest_path(shard["npz"], root=root, manifest_path=manifest_path)
            jsonl_path = _resolve_manifest_path(shard["jsonl"], root=root, manifest_path=manifest_path)
            arrays = ShardArrays(npz_path)
            raw_records = read_jsonl(jsonl_path)
            if len(raw_records) != arrays.rgb.shape[0]:
                raise ValueError(
                    f"{jsonl_path}: {len(raw_records)} records do not match "
                    f"{npz_path}: {arrays.rgb.shape[0]} array rows"
                )
            for local_index, raw_record in enumerate(raw_records):
                record = ShardRecord.from_dict(raw_record)
                sample = ShardSample(record=record, **arrays.sample(local_index))
                enriched = enrich_sample(
                    sample,
                    EnrichmentConfig(
                        recompute_intent=True,
                        recompute_structure=config.recompute_structure,
                        alpha_policy=alpha_policy,
                        alpha_l_min=config.alpha_l_min,
                        intent_raster_size=config.intent_raster_size,
                    ),
                )
                source_id = _source_id(family_name, shard["npz"], record, local_index)
                drift_bank: tuple[DriftSample, ...] = ()
                if config.drift_samples_per_patch > 0:
                    drift_bank = tuple(
                        sample_drift_bank(
                            config.drift_config,
                            root_seed,
                            source_id,
                            ppp.hash,
                            (record.x, record.y),
                            sample_count=config.drift_samples_per_patch,
                        )
                    )
                target = generate_target_from_icc_cmyk(
                    enriched.sample.icc_cmyk,
                    enriched.lab,
                    ppp,
                    config=config.target_solver_config,
                    source_id=source_id,
                    alpha_weights=enriched.alpha_effective,
                    drift_samples=drift_bank,
                )
                examples: tuple[SurrogateExample, ...] = ()
                if config.include_surrogate_examples and drift_bank:
                    examples = tuple(
                        _build_surrogate_examples(
                            target,
                            enriched,
                            ppp,
                            source_id=source_id,
                            sample_record=record,
                            config=config,
                            drift_bank=drift_bank,
                        )
                    )
                yield GeneratedTargetRecord(
                    family_name=family_name,
                    split=split,
                    shard_npz=str(shard["npz"]),
                    shard_jsonl=str(shard["jsonl"]),
                    local_index=local_index,
                    alpha_policy=alpha_policy,
                    sample_record=record,
                    enriched_sample=enriched,
                    target_result=target,
                    surrogate_examples=examples,
                )


def generate_target_records(
    split_manifest: str | Path | dict[str, Any],
    ppp: PPP,
    *,
    root: str | Path | None = None,
    config: TargetGenerationPipelineConfig = TargetGenerationPipelineConfig(),
) -> list[GeneratedTargetRecord]:
    """Materialize generated target records into memory."""
    return list(iter_generated_target_records(split_manifest, ppp, root=root, config=config))


def write_target_records_jsonl(
    split_manifest: str | Path | dict[str, Any],
    ppp: PPP,
    output_jsonl: str | Path,
    *,
    root: str | Path | None = None,
    config: TargetGenerationPipelineConfig = TargetGenerationPipelineConfig(),
) -> TargetGenerationSummary:
    """Generate target records and write their JSON manifest rows."""
    manifest, _ = load_split_manifest(split_manifest)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in iter_generated_target_records(split_manifest, ppp, root=root, config=config):
            f.write(canonical_json(record.to_manifest_dict()) + "\n")
            count += 1
    return TargetGenerationSummary(
        output_jsonl=str(output_path),
        records_written=count,
        split_manifest_version=str(manifest["split_manifest_version"]),
        alpha_policy=str(manifest["alpha_policy"]),
        ppp_hash=ppp.hash,
        source_weight_policy=dict(manifest["source_weight_policy"]),
    )


def _build_surrogate_examples(
    target: TargetGenerationResult,
    enriched: EnrichedSample,
    ppp: PPP,
    *,
    source_id: str,
    sample_record: ShardRecord,
    config: TargetGenerationPipelineConfig,
    drift_bank: Sequence[DriftSample],
) -> Iterator[SurrogateExample]:
    if enriched.intent_weights is None or enriched.intent_raster is None:
        raise ValueError("target pipeline requires recomputed intent weights and intent raster")
    for drift in drift_bank:
        yield build_surrogate_example(
            target.target_cmykogv,
            enriched.lab,
            ppp=ppp,
            drift=drift,
            structure_token=enriched.structure_token,
            intent_weights=enriched.intent_weights,
            intent_raster=enriched.intent_raster,
            lambda_value=config.lambda_value,
            metadata={
                "source_id": source_id,
                "source_path": sample_record.source_path,
                "x": sample_record.x,
                "y": sample_record.y,
            },
        )


def _validate_split_manifest_v11(manifest: dict[str, Any]) -> None:
    version = str(manifest.get("split_manifest_version", ""))
    if version != "1.1":
        raise ValueError(f"expected split_manifest_version '1.1', got {version!r}")
    for key in ("alpha_policy", "source_weight_policy", "families"):
        if key not in manifest:
            raise ValueError(f"split manifest missing required key {key!r}")
    if not isinstance(manifest["families"], list):
        raise ValueError("split manifest 'families' must be a list")


def _resolve_manifest_path(path: str | Path, *, root: str | Path | None, manifest_path: Path | None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if root is not None:
        return Path(root) / p
    if manifest_path is not None:
        candidate = manifest_path.parent / p
        if candidate.exists():
            return candidate
    return p


def _source_id(family_name: str, shard_npz: str, record: ShardRecord, local_index: int) -> str:
    payload = {
        "family": family_name,
        "shard_npz": shard_npz,
        "source_path": record.source_path,
        "x": record.x,
        "y": record.y,
        "local_index": local_index,
    }
    return canonical_json_hash(payload)


def _shape_or_none(array: np.ndarray | None) -> list[int] | None:
    return None if array is None else list(array.shape)


def _drift_hash(drift: DriftSample) -> str:
    return canonical_json_hash(
        {
            "multipliers": drift.multipliers.tolist(),
            "trc_x": drift.trc_x.tolist(),
            "trc_y": drift.trc_y.tolist(),
        }
    )
