from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Lambda schedule — §4 (Proposer And Synthetic Target Training)
# ---------------------------------------------------------------------------

#: "The default lambda schedule appends candidates without resampling earlier."
LAMBDA_SCHEDULE: dict[int, list[float]] = {
    3: [0.1, 0.5, 0.9],
    4: [0.1, 0.5, 0.9, 0.0],
    5: [0.1, 0.5, 0.9, 0.0, 1.0],
}

#: Maximum supported K_cap per spec.
K_BASE: int = 3
K_CAP_MAX: int = 5


# ---------------------------------------------------------------------------
# CandidateRecord
# ---------------------------------------------------------------------------


@dataclass
class CandidateRecord:
    """All per-candidate state needed for deterministic selection and audit.

    Fields correspond 1-to-1 with the quantities named in the spec
    §7 (Error, Risk, Selection, And Escalation).

    Parameters
    ----------
    candidate_index:
        0-based candidate number *k*.  Matches the position in the lambda
        schedule and the proposer seed index.
    lambda_val:
        The λ value used for this candidate.  Taken from :data:`LAMBDA_SCHEDULE`.
    values:
        The projected (feasible) CMYKOGV patch of shape ``(H, W, 7)``,
        ``float32``.  Must be the output of ``project_to_feasible``.
    is_feasible:
        ``True`` iff ``values`` satisfies ``K(PPP)`` after projection.
        In practice always ``True`` when ``values`` comes from ``Pi_K``.
    risk:
        ``Risk_q(y) = sort([Err(y,ε_i)]_{i=1..N})[ceil(q*N)-1]``
        (0-based index, non-interpolated finite order statistic).
    nominal_error:
        ``Err(y, ε=identity)`` — error under *no* drift, used as tie-break (step 4).
    mean_tac:
        Pixel-mean total area coverage ``mean_p(Σ_j y_p_j)``.
        Lower is preferred at tie-break step 5.
    mean_ogv:
        Pixel-mean ``mean_p(O_p + G_p + V_p)``.
        Direction of preference depends on ``lambda_val`` (step 6).
    risk_threshold_exceeded:
        ``True`` iff ``risk > ppp.risk_threshold``.
    rejection_reason:
        Human-readable reason string if this candidate was discarded,
        else ``None``.
    per_drift_errors:
        Shape ``(N,)`` float32 — the full error vector before the quantile,
        stored for diagnostic replay and re-scoring.
    extra:
        Catch-all dict for optional diagnostic fields (e.g. per-constraint
        tightness flags, TV score).
    """

    candidate_index: int
    lambda_val: float
    values: np.ndarray  # (H, W, 7) float32
    is_feasible: bool
    risk: float
    nominal_error: float
    mean_tac: float
    mean_ogv: float
    risk_threshold_exceeded: bool
    rejection_reason: str | None = None
    per_drift_errors: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def mean_cmyk_tac(self) -> float:
        """TAC over CMYK channels only (C+M+Y+K, excluding OGV)."""
        if self.values.shape[-1] < 4:
            return 0.0
        return float(self.values[..., :4].sum(axis=-1).mean())

    @property
    def is_selectable_unconditional(self) -> bool:
        """Feasible without any threshold check (used internally)."""
        return self.is_feasible

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary for manifests and decision reports."""
        return {
            "candidate_index": self.candidate_index,
            "lambda_val": self.lambda_val,
            "is_feasible": self.is_feasible,
            "risk": float(self.risk),
            "nominal_error": float(self.nominal_error),
            "mean_tac": float(self.mean_tac),
            "mean_ogv": float(self.mean_ogv),
            "risk_threshold_exceeded": self.risk_threshold_exceeded,
            "rejection_reason": self.rejection_reason,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# PatchResult
# ---------------------------------------------------------------------------


@dataclass
class PatchResult:
    """Output of :func:`~robustsep_pkg.engine.escalation.run_patch_engine`.

    Parameters
    ----------
    selected:
        The winning :class:`CandidateRecord`, or ``None`` if the patch engine
        failed for all candidates.
    all_candidates:
        Every candidate evaluated during this run (across all escalation
        rounds), in evaluation order. Used for audit / replay.
    escalation_count:
        Number of times *K* was incremented beyond ``K_base``.
    used_fallback_n:
        ``True`` if the engine re-evaluated with ``N_fallback`` drift samples
        because primary escalation was exhausted.
    failure_reason:
        Human-readable reason when ``selected is None``.
    final_k:
        The candidate count used at the point of selection (or K_cap on
        failure).
    final_n:
        The drift sample count used at the point of selection.
    """

    selected: CandidateRecord | None
    all_candidates: list[CandidateRecord]
    escalation_count: int
    used_fallback_n: bool
    failure_reason: str | None
    final_k: int
    final_n: int

    def succeeded(self) -> bool:
        return self.selected is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded(),
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "escalation_count": self.escalation_count,
            "used_fallback_n": self.used_fallback_n,
            "failure_reason": self.failure_reason,
            "final_k": self.final_k,
            "final_n": self.final_n,
        }
