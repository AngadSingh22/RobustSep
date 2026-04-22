"""Synthetic CMYKOGV target generation."""

from robustsep_pkg.targets.config import TargetSolverConfig, TeacherMode
from robustsep_pkg.targets.generation_pipeline import (
    GeneratedTargetRecord,
    TargetGenerationPipelineConfig,
    TargetGenerationSummary,
    generate_target_records,
    iter_generated_target_records,
    load_split_manifest,
    write_target_records_jsonl,
)
from robustsep_pkg.targets.generator import TargetGenerationResult, generate_target_from_icc_cmyk, initialize_cmykogv_from_icc
from robustsep_pkg.targets.manifest import TargetManifestRecord
from robustsep_pkg.targets.solver import ProjectedGradientTrace, projected_gradient_solve

__all__ = [
    "TeacherMode",
    "TargetSolverConfig",
    "TargetGenerationPipelineConfig",
    "TargetGenerationSummary",
    "TargetGenerationResult",
    "GeneratedTargetRecord",
    "TargetManifestRecord",
    "ProjectedGradientTrace",
    "load_split_manifest",
    "iter_generated_target_records",
    "generate_target_records",
    "write_target_records_jsonl",
    "initialize_cmykogv_from_icc",
    "generate_target_from_icc_cmyk",
    "projected_gradient_solve",
]
