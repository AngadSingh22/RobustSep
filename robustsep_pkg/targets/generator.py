from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robustsep_pkg.core.channels import ensure_cmykogv_last_axis
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.preprocess.color import cmyk_to_cmykogv
from robustsep_pkg.targets.config import TargetSolverConfig
from robustsep_pkg.targets.manifest import TargetManifestRecord
from robustsep_pkg.targets.solver import GradientFn, ProjectedGradientTrace, projected_gradient_solve


@dataclass(frozen=True)
class TargetGenerationResult:
    initial_cmykogv: np.ndarray
    target_cmykogv: np.ndarray
    trace: ProjectedGradientTrace
    manifest_record: TargetManifestRecord

    def to_manifest_dict(self) -> dict[str, Any]:
        return self.manifest_record.to_dict()


def initialize_cmykogv_from_icc(icc_cmyk: np.ndarray, ppp: PPP | None = None, lab_ref: np.ndarray | None = None) -> np.ndarray:
    """Append zero OGV channels to an ICC CMYK baseline and optionally project."""
    cmykogv = cmyk_to_cmykogv(np.asarray(icc_cmyk, dtype=np.float32))
    if ppp is not None:
        from robustsep_pkg.models.refiner.solver import pi_k

        cmykogv = pi_k(cmykogv, ppp, lab_ref=lab_ref)
    ensure_cmykogv_last_axis(cmykogv.shape)
    return cmykogv.astype(np.float32)


def generate_target_from_icc_cmyk(
    icc_cmyk: np.ndarray,
    lab_ref: np.ndarray,
    ppp: PPP,
    *,
    config: TargetSolverConfig = TargetSolverConfig(),
    source_id: str | None = None,
    gradient_fn: GradientFn | None = None,
) -> TargetGenerationResult:
    """Generate a deterministic CMYKOGV target from an ICC CMYK baseline."""
    initial = initialize_cmykogv_from_icc(icc_cmyk, ppp=ppp, lab_ref=lab_ref)
    target, trace = projected_gradient_solve(
        initial,
        ppp,
        lab_ref=lab_ref,
        stage1_steps=config.stage1_steps,
        stage2_steps=config.stage2_steps,
        stage1_step_size=config.stage1_step_size,
        stage2_step_size=config.stage2_step_size,
        gradient_fn=gradient_fn,
    )
    manifest_record = TargetManifestRecord.build(
        initial=initial,
        target=target,
        ppp=ppp,
        config=config,
        source_id=source_id,
        diagnostics=trace.to_dict(),
    )
    return TargetGenerationResult(
        initial_cmykogv=initial,
        target_cmykogv=target,
        trace=trace,
        manifest_record=manifest_record,
    )
