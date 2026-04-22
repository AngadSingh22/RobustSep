"""test_patch_engine.py — Non-neural patch-engine: selection, escalation, compute gating.

Tests every step of the spec §7 deterministic ordering, all escalation branches,
compute gating formulas, lambda schedule, and risk index arithmetic.
"""
from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from robustsep_pkg.core.config import DriftConfig, RiskConfig
from robustsep_pkg.engine import (
    CandidateRecord,
    K_BASE,
    LAMBDA_SCHEDULE,
    GatingParams,
    PatchResult,
    aggregate_risk,
    compute_gating,
    is_selectable,
    run_patch_engine,
    select_candidate,
)
from robustsep_pkg.models.conditioning.ppp import PPP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALUES = np.zeros((16, 16, 7), dtype=np.float32)


def _ppp(*, risk_threshold: float = 3.0, hard: bool = False, pair_caps: dict | None = None) -> PPP:
    return PPP(
        base_family="film_generic_conservative",
        caps={"C": 1.0, "M": 1.0, "Y": 1.0, "K": 1.0, "O": 0.6, "G": 0.6, "V": 0.6},
        tac_max=3.0, ogv_max=0.85,
        pair_caps=pair_caps if pair_caps is not None else {"CO": 1.1, "MG": 1.1, "YV": 1.1},
        neutral_ogv_max=0.05, neutral_chroma_threshold=5.0, dark_l_threshold=18.0,
        risk_threshold=risk_threshold, risk_threshold_hard=hard,
        override_mask={},
    )


def _rec(
    *,
    k: int = 0,
    lam: float = 0.1,
    risk: float = 1.0,
    nominal: float = 1.0,
    tac: float = 1.0,
    ogv: float = 0.1,
    feasible: bool = True,
    threshold_exceeded: bool = False,
    per_drift: np.ndarray | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_index=k,
        lambda_val=lam,
        values=_VALUES.copy(),
        is_feasible=feasible,
        risk=risk,
        nominal_error=nominal,
        mean_tac=tac,
        mean_ogv=ogv,
        risk_threshold_exceeded=threshold_exceeded,
        per_drift_errors=per_drift if per_drift is not None else np.array([risk], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Test class: Selection sort-key total ordering
# ---------------------------------------------------------------------------

class SelectionOrderingTests(unittest.TestCase):
    """Spec §7 steps 1-7 exercised exhaustively."""

    def test_step1_infeasible_candidate_is_rejected(self):
        """Infeasible candidates are never returned."""
        ppp = _ppp()
        infeas = _rec(k=0, feasible=False)
        feas = _rec(k=1, risk=5.0)  # worse risk but feasible
        winner = select_candidate([infeas, feas], ppp)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_index, 1)

    def test_step1_all_infeasible_returns_none(self):
        ppp = _ppp()
        candidates = [_rec(k=i, feasible=False) for i in range(5)]
        self.assertIsNone(select_candidate(candidates, ppp))

    def test_step2_hard_threshold_discards_exceeding_candidate(self):
        """Hard threshold: candidates with risk_threshold_exceeded are discarded."""
        ppp = _ppp(risk_threshold=2.0, hard=True)
        bad = _rec(k=0, risk=3.0, threshold_exceeded=True)
        good = _rec(k=1, risk=3.0, threshold_exceeded=False)
        winner = select_candidate([bad, good], ppp)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_index, 1, "Hard threshold must discard k=0")

    def test_step2_soft_threshold_keeps_exceeding_candidate(self):
        """Soft threshold: risk_threshold_exceeded candidate is kept (lower risk wins)."""
        ppp = _ppp(risk_threshold=2.0, hard=False)
        over_threshold = _rec(k=0, risk=2.5, threshold_exceeded=True)
        under_threshold = _rec(k=1, risk=3.0, threshold_exceeded=False)
        # k=0 has lower risk — should win despite threshold breach
        winner = select_candidate([over_threshold, under_threshold], ppp)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_index, 0, "Soft threshold: lower risk must win")

    def test_step3_minimum_risk_wins(self):
        """Step 3: lower Risk_q is preferred over everything else."""
        ppp = _ppp()
        a = _rec(k=0, risk=2.0, nominal=0.1, tac=0.1)
        b = _rec(k=1, risk=1.0, nominal=9.9, tac=9.9)
        winner = select_candidate([a, b], ppp)
        self.assertEqual(winner.candidate_index, 1, "Lower risk (1.0) must win over (2.0)")

    def test_step4_nominal_error_tiebreak(self):
        """Step 4: equal risk → lower nominal_error wins."""
        ppp = _ppp()
        a = _rec(k=0, risk=1.0, nominal=2.0, tac=0.1)
        b = _rec(k=1, risk=1.0, nominal=1.0, tac=9.9)
        winner = select_candidate([a, b], ppp)
        self.assertEqual(winner.candidate_index, 1, "Lower nominal_error must be step-4 tiebreak")

    def test_step5_mean_tac_tiebreak(self):
        """Step 5: equal risk+nominal → lower mean_tac wins."""
        ppp = _ppp()
        a = _rec(k=0, risk=1.0, nominal=1.0, tac=2.0)
        b = _rec(k=1, risk=1.0, nominal=1.0, tac=1.5)
        winner = select_candidate([a, b], ppp)
        self.assertEqual(winner.candidate_index, 1, "Lower TAC must be step-5 tiebreak")

    def test_step6_lambda_conservative_lower_ogv_wins(self):
        """Step 6, λ<=0.5: lower mean_ogv is preferred (conservative mode)."""
        ppp = _ppp()
        a = _rec(k=0, lam=0.1, risk=1.0, nominal=1.0, tac=1.0, ogv=0.4)
        b = _rec(k=1, lam=0.1, risk=1.0, nominal=1.0, tac=1.0, ogv=0.2)
        winner = select_candidate([a, b], ppp)
        self.assertEqual(winner.candidate_index, 1, "Conservative (λ=0.1): lower OGV must win")

    def test_step6_lambda_saturated_higher_ogv_wins(self):
        """Step 6, λ>0.5: higher mean_ogv is preferred (saturated/chroma mode)."""
        ppp = _ppp()
        a = _rec(k=0, lam=0.9, risk=1.0, nominal=1.0, tac=1.0, ogv=0.4)
        b = _rec(k=1, lam=0.9, risk=1.0, nominal=1.0, tac=1.0, ogv=0.2)
        winner = select_candidate([a, b], ppp)
        self.assertEqual(winner.candidate_index, 0, "Saturated (λ=0.9): higher OGV must win")

    def test_step6_lambda_boundary_at_0_5_uses_lower_ogv(self):
        """Step 6, λ=0.5 (boundary): lower OGV is preferred (<=0.5 branch)."""
        ppp = _ppp()
        a = _rec(k=0, lam=0.5, risk=1.0, nominal=1.0, tac=1.0, ogv=0.4)
        b = _rec(k=1, lam=0.5, risk=1.0, nominal=1.0, tac=1.0, ogv=0.2)
        winner = select_candidate([a, b], ppp)
        self.assertEqual(winner.candidate_index, 1, "At λ=0.5 (boundary), lower OGV must win")

    def test_step7_lower_candidate_index_last_tiebreak(self):
        """Step 7: all else equal → lower candidate_index wins."""
        ppp = _ppp()
        a = _rec(k=2, lam=0.5, risk=1.0, nominal=1.0, tac=1.0, ogv=0.1)
        b = _rec(k=0, lam=0.5, risk=1.0, nominal=1.0, tac=1.0, ogv=0.1)
        c = _rec(k=1, lam=0.5, risk=1.0, nominal=1.0, tac=1.0, ogv=0.1)
        winner = select_candidate([a, b, c], ppp)
        self.assertEqual(winner.candidate_index, 0, "Lowest index (0) must win final tiebreak")

    def test_full_ordering_ladder(self):
        """Run through all 7 steps in one call to verify whole hierarchy."""
        ppp = _ppp(risk_threshold=5.0, hard=True)
        # Create 6 candidates, each winning one step of the tiebreak ladder
        infeasible = _rec(k=6, feasible=False, risk=0.1)  # eliminated by step 1
        hard_violator = _rec(k=5, feasible=True, threshold_exceeded=True, risk=0.2)  # step 2
        high_risk = _rec(k=4, risk=4.0, nominal=0.1, tac=0.1, ogv=0.1)
        high_nominal = _rec(k=3, risk=1.0, nominal=5.0, tac=0.1, ogv=0.1)
        high_tac = _rec(k=2, risk=1.0, nominal=1.0, tac=9.0, ogv=0.1)
        winner_ref = _rec(k=0, risk=1.0, nominal=1.0, tac=1.0, ogv=0.1, lam=0.5)
        loser = _rec(k=1, risk=1.0, nominal=1.0, tac=1.0, ogv=0.5, lam=0.5)  # higher OGV

        candidates = [infeasible, hard_violator, high_risk, high_nominal, high_tac, winner_ref, loser]
        winner = select_candidate(candidates, ppp)
        self.assertEqual(winner.candidate_index, 0)


# ---------------------------------------------------------------------------
# Test class: is_selectable
# ---------------------------------------------------------------------------

class IsSelectableTests(unittest.TestCase):

    def test_infeasible_never_selectable(self):
        ppp = _ppp(hard=False)
        self.assertFalse(is_selectable(_rec(feasible=False), ppp))

    def test_hard_threshold_exceedance_not_selectable(self):
        ppp = _ppp(hard=True)
        self.assertFalse(is_selectable(_rec(threshold_exceeded=True), ppp))

    def test_soft_threshold_exceedance_still_selectable(self):
        ppp = _ppp(hard=False)
        self.assertTrue(is_selectable(_rec(threshold_exceeded=True), ppp))

    def test_feasible_and_within_threshold_selectable(self):
        ppp = _ppp(hard=True)
        self.assertTrue(is_selectable(_rec(threshold_exceeded=False), ppp))


# ---------------------------------------------------------------------------
# Test class: compute_gating
# ---------------------------------------------------------------------------

class ComputeGatingTests(unittest.TestCase):
    """Spec §7 compute-gating formulas."""

    def test_brand_patch_k_cap_5(self):
        """w_B >= 0.25 → K_cap=5, N_fallback=64."""
        g = compute_gating({"brand": 0.25, "flat": 0.0, "gradient": 0.0})
        self.assertEqual(g.k_cap, 5)
        self.assertEqual(g.n_fallback, 64)
        self.assertEqual(g.k_base, K_BASE)

    def test_gradient_patch_k_cap_5(self):
        """w_G >= 0.35 → K_cap=5, N_fallback=64."""
        g = compute_gating({"brand": 0.0, "flat": 0.1, "gradient": 0.35})
        self.assertEqual(g.k_cap, 5)
        self.assertEqual(g.n_fallback, 64)

    def test_priority_1_gives_k_cap_4(self):
        """priority_u >= 1.0 (but not brand/gradient threshold) → K_cap=4."""
        # 2*0.0 + 1.5*0.0 + 0.5*2.0 = 1.0 — exactly at boundary
        g = compute_gating({"brand": 0.0, "flat": 2.0, "gradient": 0.0})
        self.assertEqual(g.k_cap, 4)
        self.assertEqual(g.n_fallback, g.n_default, "No N_fallback boost without high brand/grad")

    def test_low_priority_k_cap_3(self):
        """priority_u < 1.0 and no brand/gradient → K_cap=3, N_fallback=N_default."""
        g = compute_gating({"brand": 0.0, "flat": 0.1, "gradient": 0.1})
        self.assertEqual(g.k_cap, 3)
        self.assertEqual(g.n_fallback, g.n_default)

    def test_empty_weights_gives_k_base(self):
        """Completely flat patch (no intents) → K_cap=3."""
        g = compute_gating({})
        self.assertEqual(g.k_cap, K_BASE)

    def test_priority_calculation(self):
        """priority_u = 2*w_B + 1.5*w_G + 0.5*w_F."""
        g = compute_gating({"brand": 0.1, "gradient": 0.2, "flat": 0.3})
        expected_priority = 2.0 * 0.1 + 1.5 * 0.2 + 0.5 * 0.3
        self.assertAlmostEqual(g.priority, expected_priority, places=6)

    def test_n_default_matches_drift_config(self):
        """n_default always equals drift_config.sample_count."""
        dc = DriftConfig(sample_count=48, fallback_sample_count=96)
        g = compute_gating({"brand": 0.0}, dc)
        self.assertEqual(g.n_default, 48)

    def test_n_fallback_for_high_priority_uses_drift_config(self):
        """N_fallback for brand patches matches drift_config.fallback_sample_count."""
        dc = DriftConfig(sample_count=32, fallback_sample_count=64)
        g = compute_gating({"brand": 0.30}, dc)  # w_B >= 0.25
        self.assertEqual(g.n_fallback, 64)


# ---------------------------------------------------------------------------
# Test class: aggregate_risk
# ---------------------------------------------------------------------------

class AggregateRiskTests(unittest.TestCase):
    """Risk_q = sort(errors)[ceil(q*N)-1] — spec §7."""

    def test_q90_n32_index_28(self):
        """For q=0.90, N=32: index=ceil(0.90*32)-1=ceil(28.8)-1=29-1=28."""
        errors = np.arange(32, dtype=np.float32)  # sorted: 0..31
        risk = aggregate_risk(errors, q=0.90)
        self.assertAlmostEqual(risk, 28.0, places=5)

    def test_q50_n10_index_4(self):
        """q=0.50, N=10: ceil(5.0)-1=4 → index 4 of sorted errors."""
        errors = np.array([9, 7, 2, 4, 1, 5, 8, 3, 6, 0], dtype=np.float32)
        risk = aggregate_risk(errors, q=0.50)
        # sorted: [0,1,2,3,4,5,6,7,8,9]; index=ceil(5)-1=4; value=4
        self.assertAlmostEqual(risk, 4.0, places=5)

    def test_q1_n1_index_0(self):
        """q=1.0, N=1: ceil(1.0)-1=0 → only element."""
        errors = np.array([7.5], dtype=np.float32)
        self.assertAlmostEqual(aggregate_risk(errors, q=1.0), 7.5, places=5)

    def test_q90_n64_index_56(self):
        """q=0.90, N=64: ceil(57.6)-1=58-1=57 → index 57."""
        errors = np.arange(64, dtype=np.float32)
        risk = aggregate_risk(errors, q=0.90)
        # ceil(0.90*64)=ceil(57.6)=58; 58-1=57
        self.assertAlmostEqual(risk, 57.0, places=5)

    def test_unsorted_input_gives_same_result(self):
        """Aggregate_risk sorts internally; order of input doesn't matter."""
        errors_sorted = np.arange(10, dtype=np.float32)
        errors_shuffled = errors_sorted.copy()
        np.random.default_rng(0).shuffle(errors_shuffled)
        self.assertAlmostEqual(
            aggregate_risk(errors_sorted, 0.90),
            aggregate_risk(errors_shuffled, 0.90),
            places=5,
        )


# ---------------------------------------------------------------------------
# Test class: Lambda schedule
# ---------------------------------------------------------------------------

class LambdaScheduleTests(unittest.TestCase):
    """Spec §4 lambda schedules."""

    def test_lambda_3_values(self):
        self.assertEqual(LAMBDA_SCHEDULE[3], [0.1, 0.5, 0.9])

    def test_lambda_4_values(self):
        self.assertEqual(LAMBDA_SCHEDULE[4], [0.1, 0.5, 0.9, 0.0])

    def test_lambda_5_values(self):
        self.assertEqual(LAMBDA_SCHEDULE[5], [0.1, 0.5, 0.9, 0.0, 1.0])

    def test_lambda_3_prefixes_lambda_4(self):
        """Escalating from K=3 to K=4 only adds a new candidate (index 3)."""
        self.assertEqual(LAMBDA_SCHEDULE[4][:3], LAMBDA_SCHEDULE[3])

    def test_lambda_4_prefixes_lambda_5(self):
        self.assertEqual(LAMBDA_SCHEDULE[5][:4], LAMBDA_SCHEDULE[4])


# ---------------------------------------------------------------------------
# Test class: run_patch_engine (escalation loop)
# ---------------------------------------------------------------------------

class RunPatchEngineTests(unittest.TestCase):
    """Integration tests for the full escalation policy."""

    def _make_propose_fn(
        self,
        risk_per_candidate: list[float],
        feasible_per_candidate: list[bool] | None = None,
        threshold_value: float = 3.0,
    ):
        """Return a propose_fn that always uses pre-baked risk values."""
        if feasible_per_candidate is None:
            feasible_per_candidate = [True] * len(risk_per_candidate)

        lam_sched = LAMBDA_SCHEDULE[5]

        def propose_fn(k: int, n_drift: int) -> CandidateRecord:
            risk = risk_per_candidate[k] if k < len(risk_per_candidate) else 99.0
            feasible = feasible_per_candidate[k] if k < len(feasible_per_candidate) else False
            lam = lam_sched[k] if k < len(lam_sched) else 0.0
            errors = np.full(n_drift, risk, dtype=np.float32)
            return CandidateRecord(
                candidate_index=k,
                lambda_val=lam,
                values=_VALUES.copy(),
                is_feasible=feasible,
                risk=risk,
                nominal_error=risk,
                mean_tac=1.0,
                mean_ogv=0.1,
                risk_threshold_exceeded=risk > threshold_value,
                per_drift_errors=errors,
            )

        return propose_fn

    # ── 14. Normal selection at K_base ───────────────────────────────────

    def test_selects_at_k_base_no_escalation(self):
        """All 3 base candidates are fine → winner from first round, no escalation."""
        ppp = _ppp(risk_threshold=5.0, hard=False)
        propose_fn = self._make_propose_fn([1.0, 2.0, 3.0])
        result = run_patch_engine(propose_fn, ppp, intent_weights={})
        self.assertTrue(result.succeeded())
        self.assertEqual(result.escalation_count, 0)
        self.assertEqual(result.final_k, K_BASE)
        # Winner should be k=0 (lowest risk=1.0)
        self.assertEqual(result.selected.candidate_index, 0)

    # ── 15. Escalation to K=4 ────────────────────────────────────────────

    def test_escalates_once_to_k4_when_first3_fail(self):
        """First 3 candidates infeasible → escalate, 4th succeeds."""
        ppp = _ppp(risk_threshold=5.0, hard=False)
        feasible = [False, False, False, True, True]
        risks = [0.1, 0.2, 0.3, 1.0, 2.0]
        propose_fn = self._make_propose_fn(risks, feasible)
        # Need at least K_cap=4 to escalate from 3->4
        result = run_patch_engine(
            propose_fn, ppp,
            # gradient weight to unlock K_cap=4
            intent_weights={"gradient": 0.0, "flat": 2.0},  # priority=1.0 -> k_cap=4
        )
        self.assertTrue(result.succeeded())
        self.assertGreaterEqual(result.escalation_count, 1)
        self.assertEqual(result.selected.candidate_index, 3)

    # ── 16. N_fallback re-evaluation ─────────────────────────────────────

    def test_uses_fallback_n_after_primary_escalation_exhausted(self):
        """Primary escalation exhausted → N_fallback re-evaluation → success."""
        ppp = _ppp(risk_threshold=0.5, hard=True)  # hard threshold = 0.5
        # All candidates have risk=1.0 which exceeds threshold=0.5 in first round
        # But the propose_fn is called with n_fallback=64 in the second round
        # We simulate: first call (n=32) suggests over-threshold; second call (n=64) suggests ok

        call_counts = {"n32": 0, "n64": 0}
        lam_sched = LAMBDA_SCHEDULE[5]

        def propose_fn(k: int, n_drift: int) -> CandidateRecord:
            # When called with n_drift=64 (fallback), return lower risk
            if n_drift >= 64:
                call_counts["n64"] += 1
                risk = 0.1  # under threshold
                threshold_exceeded = False
            else:
                call_counts["n32"] += 1
                risk = 1.0  # over threshold
                threshold_exceeded = True
            lam = lam_sched[k] if k < len(lam_sched) else 0.0
            errors = np.full(n_drift, risk, dtype=np.float32)
            return CandidateRecord(
                candidate_index=k,
                lambda_val=lam,
                values=_VALUES.copy(),
                is_feasible=True,
                risk=risk,
                nominal_error=risk,
                mean_tac=1.0,
                mean_ogv=0.1,
                risk_threshold_exceeded=threshold_exceeded,
                per_drift_errors=errors,
            )

        result = run_patch_engine(
            propose_fn, ppp,
            intent_weights={"brand": 0.30},  # brand >= 0.25 -> K_cap=5, N_fallback=64
        )
        self.assertTrue(result.succeeded(), msg=f"Should succeed via N_fallback: {result.failure_reason}")
        self.assertTrue(result.used_fallback_n, "Must have used N_fallback")
        self.assertGreater(call_counts["n64"], 0, "propose_fn must be called with n=64")

    # ── 17. Hard failure ─────────────────────────────────────────────────

    def test_hard_failure_when_all_infeasible(self):
        """All infeasible + hard threshold → PatchResult.succeeded() == False."""
        ppp = _ppp(risk_threshold=5.0, hard=True)
        propose_fn = self._make_propose_fn(
            [1.0] * 5, feasible_per_candidate=[False] * 5
        )
        result = run_patch_engine(
            propose_fn, ppp,
            intent_weights={"brand": 0.30},  # K_cap=5
        )
        self.assertFalse(result.succeeded())
        self.assertIsNone(result.selected)
        self.assertIn("no_feasible", result.failure_reason)

    # ── 18. Soft fallback ────────────────────────────────────────────────

    def test_soft_fallback_returns_minimum_risk_feasible(self):
        """Soft threshold: exceed threshold → return minimum-risk feasible."""
        ppp = _ppp(risk_threshold=0.5, hard=False)  # soft threshold
        # All risks exceed threshold (>0.5), but soft mode allows returning minimum
        risks = [2.0, 1.0, 3.0]
        propose_fn = self._make_propose_fn(risks, threshold_value=0.5)
        result = run_patch_engine(propose_fn, ppp, intent_weights={})
        self.assertTrue(result.succeeded(), "Soft fallback must return a winner")
        # Must return minimum-risk feasible = k=1 (risk=1.0)
        self.assertEqual(result.selected.candidate_index, 1,
                         "Soft fallback must return minimum-risk candidate")

    # ── Candidate caching / independence ─────────────────────────────────

    def test_propose_fn_called_once_per_new_candidate(self):
        """Candidates are cached: propose_fn called exactly once per new k."""
        call_log = []
        lam_sched = LAMBDA_SCHEDULE[5]

        def propose_fn(k: int, n_drift: int) -> CandidateRecord:
            call_log.append(k)
            lam = lam_sched[k] if k < len(lam_sched) else 0.0
            return CandidateRecord(
                candidate_index=k, lambda_val=lam,
                values=_VALUES.copy(), is_feasible=True,
                risk=float(k + 1), nominal_error=1.0,
                mean_tac=1.0, mean_ogv=0.1,
                risk_threshold_exceeded=False,
                per_drift_errors=np.array([1.0], dtype=np.float32),
            )

        result = run_patch_engine(propose_fn, _ppp(), intent_weights={})
        # With K_base=3 and a winner at round 1, propose_fn called for k=0,1,2
        self.assertEqual(sorted(call_log), [0, 1, 2])

    # ── PatchResult fields ────────────────────────────────────────────────

    def test_patch_result_to_dict_serialisable(self):
        """PatchResult.to_dict() is JSON-serialisable."""
        import json
        ppp = _ppp()
        propose_fn = self._make_propose_fn([1.0, 2.0, 3.0])
        result = run_patch_engine(propose_fn, ppp, intent_weights={})
        d = result.to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# Test class: CandidateRecord
# ---------------------------------------------------------------------------

class CandidateRecordTests(unittest.TestCase):

    def test_to_dict_round_trip(self):
        rec = _rec(k=2, lam=0.5, risk=1.5, nominal=1.0, tac=2.0, ogv=0.3)
        d = rec.to_dict()
        self.assertEqual(d["candidate_index"], 2)
        self.assertAlmostEqual(d["risk"], 1.5)
        self.assertAlmostEqual(d["mean_ogv"], 0.3)

    def test_extra_fields_preserved(self):
        rec = _rec()
        rec.extra["tv_score"] = 0.012
        d = rec.to_dict()
        self.assertIn("tv_score", d)

    def test_mean_cmyk_tac_property(self):
        rec = _rec()
        # values is all zeros -> mean_cmyk_tac = 0
        self.assertAlmostEqual(rec.mean_cmyk_tac, 0.0)


if __name__ == "__main__":
    unittest.main()
