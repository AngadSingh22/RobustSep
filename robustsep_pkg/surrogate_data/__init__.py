"""Forward-surrogate data example helpers."""

from robustsep_pkg.surrogate_data.candidates import LambdaCandidateConfig, generate_lambda_candidate_contexts
from robustsep_pkg.surrogate_data.context import ContextWindow, extract_center_context, pad_patch_to_context
from robustsep_pkg.surrogate_data.examples import SurrogateExample, build_surrogate_example
from robustsep_pkg.surrogate_data.shard_writer import (
    SurrogateShardSummary,
    SurrogateShardWriterConfig,
    write_surrogate_training_shards,
)

__all__ = [
    "ContextWindow",
    "SurrogateExample",
    "LambdaCandidateConfig",
    "SurrogateShardWriterConfig",
    "SurrogateShardSummary",
    "extract_center_context",
    "generate_lambda_candidate_contexts",
    "pad_patch_to_context",
    "build_surrogate_example",
    "write_surrogate_training_shards",
]
