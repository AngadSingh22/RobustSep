from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from robustsep_pkg.core.config import DriftConfig, RiskConfig
from robustsep_pkg.engine.candidate import (
    LAMBDA_SCHEDULE,
    K_BASE,
    CandidateRecord,
    PatchResult,
)
from robustsep_pkg.engine.selection import is_selectable, select_candidate
from robustsep_pkg.eval.metrics import finite_quantile
from robustsep_pkg.models.conditioning.ppp import PPP


# ---------------------------------------------------------------------------
# Compute gating
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatingParams:
    """Patch-level compute budget deriving from intent weights and PPP.

    Attributes
    ----------
    k_base : int
        Starting candidate count (always 3 per spec).
    k_cap : int
        Maximum candidate count for this patch (3, 4, or 5).
    n_default : int
        Default drift sample count (always 32 per spec).
    n_fallback : int
        Fallback drift count after primary escalation is exhausted (32 or 64).
    priority : float
        ``2*w_B + 1.5*w_G + 0.5*w_F`` (logged for diagnostics).
    """

    k_base: int
    k_cap: int
    n_default: int
    n_fallback: int
    priority: float


def compute_gating(
    intent_weights: dict[str, float],
    drift_config: DriftConfig = DriftConfig(),
) -> GatingParams:
    """Compute the per-patch K_cap and N budgets from intent weights.

    Implements the spec formula verbatim (§7 Compute gating)::

        priority_u = 2*w_B + 1.5*w_G + 0.5*w_F

        K_cap_u = 5  if w_B >= 0.25 or w_G >= 0.35
                = 4  if priority_u >= 1.0
                = 3  otherwise

        N_u          = 32  (drift_config.sample_count)
        N_fallback_u = 64  if w_B >= 0.25 or w_G >= 0.35
                     = N_u otherwise

    Parameters
    ----------
    intent_weights:
        Dict with keys ``"brand"``, ``"flat"``, ``"gradient"`` (per-patch
        aggregated scalars from :func:`~robustsep_pkg.preprocess.intent.\
aggregate_patch_intents`).  Missing keys default to 0.
    drift_config:
        Provides ``sample_count`` (N_default) and ``fallback_sample_count``
        (N_fallback defaults).
    """
    w_b = float(intent_weights.get("brand", 0.0))
    w_g = float(intent_weights.get("gradient", 0.0))
    w_f = float(intent_weights.get("flat", 0.0))

    priority = 2.0 * w_b + 1.5 * w_g + 0.5 * w_f

    high_priority = w_b >= 0.25 or w_g >= 0.35
    if high_priority:
        k_cap = 5
    elif priority >= 1.0:
        k_cap = 4
    else:
        k_cap = 3

    n_fallback = (
        drift_config.fallback_sample_count if high_priority else drift_config.sample_count
    )

    return GatingParams(
        k_base=K_BASE,
        k_cap=k_cap,
        n_default=drift_config.sample_count,
        n_fallback=n_fallback,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Risk aggregation helper
# ---------------------------------------------------------------------------


def aggregate_risk(
    per_drift_errors: np.ndarray,
    q: float,
) -> float:
    """Compute ``Risk_q`` from per-drift error array.

    Implements (§7)::

        Risk_q(y) = sort([Err(y, ε_i)]_{i=1..N})[ceil(q*N) - 1]

    This is the same non-interpolated order statistic used by
    :func:`~robustsep_pkg.eval.metrics.finite_quantile`.

    Parameters
    ----------
    per_drift_errors:
        1-D array of length N containing ``Err(y, ε_i)`` for each drift
        sample.
    q:
        Risk quantile (e.g. 0.90).
    """
    return float(finite_quantile(per_drift_errors, q))


# ---------------------------------------------------------------------------
# Propose-and-score factory type alias
# ---------------------------------------------------------------------------

#: Type of the injected factory function.
#: ``propose_and_score(k, n_drift) -> CandidateRecord``
#: This decouples the neural model from the engine.
ProposeFn = Callable[[int, int], CandidateRecord]


# ---------------------------------------------------------------------------
# Main patch engine loop
# ---------------------------------------------------------------------------


def run_patch_engine(
    propose_fn: ProposeFn,
    ppp: PPP,
    intent_weights: dict[str, float],
    drift_config: DriftConfig = DriftConfig(),
    risk_config: RiskConfig = RiskConfig(),
    lambda_schedule: list[float] | None = None,
) -> PatchResult:
    """Run the full escalation loop for one patch.

    The engine is **non-neural** — the proposer and scorer are injected via
    *propose_fn* so any implementation (neural or analytic) can be plugged in.

    Algorithm (verbatim from spec §7)::

        K ← K_base
        while K <= K_cap_u:
            evaluate candidates 0..K-1
            (reuse cached; only call propose_fn for new indices)
            if selectable candidate exists: return winner
            K ← K + 1

        if no winner and N_fallback > N_default:
            re-evaluate all K_cap candidates with N_fallback drift samples
            update per_drift_errors and risk

        if still no winner:
            if ppp.risk_threshold_hard is False:
                return feasible minimum-risk candidate (soft fallback)
            else:
                return PatchResult with failure_reason set (fatal)

    Parameters
    ----------
    propose_fn:
        ``propose_fn(k, n_drift) -> CandidateRecord`` — called for each
        new candidate index *k* with the current drift count *n_drift*.
        **Must** set ``per_drift_errors``, ``risk``, ``nominal_error``,
        ``mean_tac``, ``mean_ogv``, ``is_feasible``, and
        ``risk_threshold_exceeded``.
    ppp:
        Effective PPP for this patch.
    intent_weights:
        Per-patch intent weight scalars (keys: ``"brand"``, ``"flat"``,
        ``"gradient"``).
    drift_config:
        Provides N defaults.
    risk_config:
        Provides q (quantile), eps, etc.
    lambda_schedule:
        Override the lambda schedule; defaults to the spec's built-in
        schedule (``LAMBDA_SCHEDULE[5][:k_cap]``).

    Returns
    -------
    :class:`~robustsep_pkg.engine.candidate.PatchResult`
    """
    gating = compute_gating(intent_weights, drift_config)
    k_cap = gating.k_cap
    n_default = gating.n_default
    n_fallback = gating.n_fallback

    lam_sched = lambda_schedule if lambda_schedule is not None else LAMBDA_SCHEDULE[5]

    # Cache: candidate_index -> CandidateRecord (evaluated at n_default)
    cache: dict[int, CandidateRecord] = {}
    all_candidates: list[CandidateRecord] = []
    escalation_count = 0
    winner: CandidateRecord | None = None

    # ── Primary escalation loop ──────────────────────────────────────────
    k = K_BASE
    while k <= k_cap:
        # Evaluate any new candidates beyond what is cached
        for idx in range(len(cache), k):
            rec = propose_fn(idx, n_default)
            # Stamp the risk_threshold_exceeded flag here so the engine
            # controls the threshold check consistently.
            object.__setattr__ if False else None  # mypy hint
            rec.risk_threshold_exceeded = rec.risk > ppp.risk_threshold
            if not rec.is_feasible:
                rec.rejection_reason = "infeasible"
            elif rec.risk_threshold_exceeded and ppp.risk_threshold_hard:
                rec.rejection_reason = "risk_threshold_hard"
            cache[idx] = rec
            all_candidates.append(rec)

        current_pool = list(cache.values())
        winner = select_candidate(current_pool, ppp)
        if winner is not None:
            return PatchResult(
                selected=winner,
                all_candidates=all_candidates,
                escalation_count=escalation_count,
                used_fallback_n=False,
                failure_reason=None,
                final_k=k,
                final_n=n_default,
            )

        # No winner yet — escalate
        k += 1
        escalation_count += 1

    # ── N_fallback re-evaluation ─────────────────────────────────────────
    used_fallback_n = False
    if n_fallback > n_default:
        used_fallback_n = True
        fallback_cache: dict[int, CandidateRecord] = {}
        for idx in range(k_cap):
            rec = propose_fn(idx, n_fallback)
            rec.risk_threshold_exceeded = rec.risk > ppp.risk_threshold
            if not rec.is_feasible:
                rec.rejection_reason = "infeasible"
            elif rec.risk_threshold_exceeded and ppp.risk_threshold_hard:
                rec.rejection_reason = "risk_threshold_hard"
            fallback_cache[idx] = rec
            all_candidates.append(rec)

        winner = select_candidate(list(fallback_cache.values()), ppp)
        if winner is not None:
            return PatchResult(
                selected=winner,
                all_candidates=all_candidates,
                escalation_count=escalation_count,
                used_fallback_n=True,
                failure_reason=None,
                final_k=k_cap,
                final_n=n_fallback,
            )

    # ── Soft / hard threshold fallback ───────────────────────────────────
    # Gather all feasible candidates from primary + fallback runs
    feasible = [c for c in all_candidates if c.is_feasible]

    if not ppp.risk_threshold_hard and feasible:
        # Soft mode: return the feasible minimum-risk candidate even if its
        # risk exceeds the threshold (spec: "choose the feasible minimum-risk
        # candidate if PPP threshold is soft").
        soft_winner = min(feasible, key=lambda c: (c.risk, c.nominal_error, c.candidate_index))
        soft_winner.rejection_reason = None  # clear any prior reason
        return PatchResult(
            selected=soft_winner,
            all_candidates=all_candidates,
            escalation_count=escalation_count,
            used_fallback_n=used_fallback_n,
            failure_reason="soft_fallback: risk_threshold_exceeded but returned minimum-risk feasible",
            final_k=k_cap,
            final_n=n_fallback if used_fallback_n else n_default,
        )

    # Fatal failure
    reason_parts = []
    if not feasible:
        reason_parts.append("no_feasible_candidates")
    elif ppp.risk_threshold_hard:
        reason_parts.append("all_feasible_candidates_exceed_hard_risk_threshold")
    return PatchResult(
        selected=None,
        all_candidates=all_candidates,
        escalation_count=escalation_count,
        used_fallback_n=used_fallback_n,
        failure_reason="; ".join(reason_parts) or "unknown",
        final_k=k_cap,
        final_n=n_fallback if used_fallback_n else n_default,
    )
