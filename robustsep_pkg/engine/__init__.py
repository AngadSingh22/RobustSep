"""robustsep_pkg.engine — non-neural patch-engine components.

Public API
----------
CandidateRecord
    Per-candidate state: feasibility, risk, nominal error, TAC, OGV,
    rejection reason, per-drift errors.
PatchResult
    Full engine output including escalation diagnostics.
LAMBDA_SCHEDULE
    {3: [0.1, 0.5, 0.9], 4: [...], 5: [...]} — spec lambda schedules.
K_BASE, K_CAP_MAX
    Default and maximum candidate counts.
GatingParams
    Immutable compute budget (k_base, k_cap, n_default, n_fallback, priority).
compute_gating(intent_weights, drift_config) -> GatingParams
    Spec formula: priority_u -> K_cap_u, N_fallback_u.
is_selectable(rec, ppp) -> bool
    Steps 1-2 of spec selection ordering (feasibility + hard threshold).
select_candidate(candidates, ppp) -> CandidateRecord | None
    Spec 7-step deterministic total ordering.
run_patch_engine(propose_fn, ppp, ...) -> PatchResult
    Full escalation loop (K escalation -> N_fallback -> soft/hard failure).
aggregate_risk(per_drift_errors, q) -> float
    Risk_q = sort(errors)[ceil(q*N)-1].
"""
from __future__ import annotations

from robustsep_pkg.engine.candidate import (
    LAMBDA_SCHEDULE,
    K_BASE,
    K_CAP_MAX,
    CandidateRecord,
    PatchResult,
)
from robustsep_pkg.engine.escalation import (
    GatingParams,
    aggregate_risk,
    compute_gating,
    run_patch_engine,
)
from robustsep_pkg.engine.selection import is_selectable, select_candidate

__all__ = [
    "CandidateRecord",
    "PatchResult",
    "LAMBDA_SCHEDULE",
    "K_BASE",
    "K_CAP_MAX",
    "GatingParams",
    "compute_gating",
    "aggregate_risk",
    "is_selectable",
    "select_candidate",
    "run_patch_engine",
]
