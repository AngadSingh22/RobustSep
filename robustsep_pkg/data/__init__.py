"""Dataset loader, training adapter, and enrichment for staged RobustSep patch shards."""

from robustsep_pkg.data.shard_record import ShardEntry, ShardRecord, ShardSample
from robustsep_pkg.data.shard_reader import ShardArrays, ShardReader
from robustsep_pkg.data.split import deterministic_split
from robustsep_pkg.data.dataset import RobustSepDataset
from robustsep_pkg.data.batching import ShardBatch, iter_batches
from robustsep_pkg.data.source_weighting import SourceWeightPolicy, weighted_shard_schedule
from robustsep_pkg.data.training_adapter import FamilyDataset, TrainingAdapter
from robustsep_pkg.data.enrichment import (
    EnrichmentConfig,
    EnrichedSample,
    enrich_sample,
    apply_alpha_fallback,
)
from robustsep_pkg.data.intent_adapter import (
    compute_intent_weights,
    compute_structure_token,
    compute_low_res_intent_raster,
)

__all__ = [
    # Core loader
    "ShardEntry",
    "ShardRecord",
    "ShardSample",
    "ShardArrays",
    "ShardReader",
    "deterministic_split",
    "RobustSepDataset",
    # Batching
    "ShardBatch",
    "iter_batches",
    # Source weighting
    "SourceWeightPolicy",
    "weighted_shard_schedule",
    # Training adapter
    "FamilyDataset",
    "TrainingAdapter",
    # Enrichment
    "EnrichmentConfig",
    "EnrichedSample",
    "enrich_sample",
    "apply_alpha_fallback",
    # Intent adapter
    "compute_intent_weights",
    "compute_structure_token",
    "compute_low_res_intent_raster",
]
