from __future__ import annotations

from robustsep_pkg.engine.candidate import CandidateRecord
from robustsep_pkg.models.conditioning.ppp import PPP


# ---------------------------------------------------------------------------
# Selectability predicate
# ---------------------------------------------------------------------------


def is_selectable(rec: CandidateRecord, ppp: PPP) -> bool:
    """Return True iff candidate *rec* may be selected under *ppp*.

    Step 1 in the spec's 7-step total ordering:
    *"feasible after Pi_K"* is a hard requirement — infeasible candidates
    are always discarded.

    Step 2: *"below risk threshold if the threshold is hard, else threshold
    breach is logged"*.  When ``ppp.risk_threshold_hard`` is True this becomes
    a discard condition; when False the breach is retained for diagnostics but
    the candidate remains eligible.
    """
    if not rec.is_feasible:
        return False
    if ppp.risk_threshold_hard and rec.risk_threshold_exceeded:
        return False
    return True


# ---------------------------------------------------------------------------
# Selection sort key
# ---------------------------------------------------------------------------


def _selection_key(rec: CandidateRecord) -> tuple:
    """Primary sort key for items that are *already* confirmed selectable.

    Implements steps 3-7 of the spec deterministic total ordering::

        3. minimum Risk_q
        4. minimum nominal_error (identity-drift Err)
        5. lower mean_tac
        6. lambda-aware tie-break:
               lower mean_ogv   when lambda_val <= 0.5  (conservative)
               higher mean_ogv  when lambda_val >  0.5  (saturated)
        7. lower candidate_index

    The lambda-aware OGV rule is encoded as a scalar that is sorted
    *ascending*, so for lambda > 0.5 we negate mean_ogv so that a higher
    OGV value sorts earlier (is preferred).
    """
    ogv_tiebreak = (
        rec.mean_ogv if rec.lambda_val <= 0.5 else -rec.mean_ogv
    )
    return (
        rec.risk,            # step 3 — ascending (lower risk preferred)
        rec.nominal_error,   # step 4 — ascending
        rec.mean_tac,        # step 5 — ascending
        ogv_tiebreak,        # step 6 — ascending (see above)
        rec.candidate_index, # step 7 — ascending (lower index preferred)
    )


# ---------------------------------------------------------------------------
# Public selection function
# ---------------------------------------------------------------------------


def select_candidate(
    candidates: list[CandidateRecord],
    ppp: PPP,
) -> CandidateRecord | None:
    """Apply the spec's 7-step deterministic total ordering and return the winner.

    Parameters
    ----------
    candidates:
        All candidate records to choose from.  Typically the union of all
        candidates evaluated so far in a single escalation round.
    ppp:
        The effective PPP for this patch, used for threshold enforcement.

    Returns
    -------
    The best :class:`~robustsep_pkg.engine.candidate.CandidateRecord` or
    ``None`` if no candidate passes feasibility + hard threshold checks.
    """
    selectable = [c for c in candidates if is_selectable(c, ppp)]
    if not selectable:
        return None
    return min(selectable, key=_selection_key)
