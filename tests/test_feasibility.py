"""test_feasibility.py — Mathematical correctness of PPP feasibility projection.

Tests every constraint in K(PPP) and every step of Pi_K, including:
  - Per-channel caps
  - Pair caps (CO, MG, YV) with proportional scaling
  - OGV total cap
  - Neutral/dark OGV cap (both chroma-triggered and L-triggered)
  - TAC capped-simplex (verified to land exactly on tac_max)
  - Idempotency (feasible pixels unchanged)
  - Projection ordering (pair → OGV → neutral-OGV → TAC)
  - feasibility_violations() counting
  - validate() key rejection
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from robustsep_pkg.models.conditioning.ppp import (
    PPP,
    feasibility_violations,
    project_to_feasible,
)


# ---------------------------------------------------------------------------
# Helper: build a PPP with known caps for predictable verification
# ---------------------------------------------------------------------------

def _ppp(
    *,
    tac_max: float = 3.0,
    ogv_max: float = 0.85,
    neutral_ogv_max: float = 0.05,
    neutral_chroma_threshold: float = 5.0,
    dark_l_threshold: float = 18.0,
    pair_caps: dict | None = None,
    channel_caps: dict | None = None,
    risk_threshold_hard: bool = False,
) -> PPP:
    caps = {
        "C": 1.0, "M": 1.0, "Y": 1.0, "K": 1.0,
        "O": 0.60, "G": 0.60, "V": 0.60,
    }
    if channel_caps:
        caps.update(channel_caps)
    return PPP(
        base_family="film_generic_conservative",
        caps=caps,
        tac_max=tac_max,
        ogv_max=ogv_max,
        pair_caps=pair_caps if pair_caps is not None else {"CO": 1.10, "MG": 1.10, "YV": 1.10},
        neutral_ogv_max=neutral_ogv_max,
        neutral_chroma_threshold=neutral_chroma_threshold,
        dark_l_threshold=dark_l_threshold,
        risk_threshold=3.0,
        risk_threshold_hard=risk_threshold_hard,
        override_mask={},
    )


def _patch(values: list[float], shape: tuple = (1, 1, 7)) -> np.ndarray:
    """Return a (H,W,7) float32 array filled with the given 7 channel values."""
    z = np.zeros(shape, dtype=np.float32)
    z[..., :] = values
    return z


class FeasibilityProjectionTests(unittest.TestCase):
    """Mathematical correctness of project_to_feasible and K(PPP)."""

    # ── 1. Per-channel caps ──────────────────────────────────────────────

    def test_channel_cap_clipping_each_individually(self):
        """Every channel is clipped to its cap independently."""
        ppp = _ppp(channel_caps={"C": 0.7, "O": 0.4})
        # C=0.9>0.7, O=0.5>0.4; M,Y,K,G,V all at 0.1 (fine)
        z = _patch([0.9, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1])
        out = project_to_feasible(z, ppp)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.7, places=5, msg="C must be clipped to 0.7")
        self.assertAlmostEqual(float(out[0, 0, 4]), 0.4, places=5, msg="O must be clipped to 0.4")
        # Other channels unchanged (sum is safe)
        self.assertAlmostEqual(float(out[0, 0, 1]), 0.1, places=5)

    def test_channel_cap_below_zero_clamped(self):
        """Negative values are clamped to zero."""
        ppp = _ppp()
        z = _patch([-0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        out = project_to_feasible(z, ppp)
        self.assertGreaterEqual(float(out[0, 0, 0]), 0.0)

    # ── 2. Pair caps (CO, MG, YV) ────────────────────────────────────────

    def test_pair_cap_co_proportional_scaling(self):
        """C+O > pair_CO_max: both scaled proportionally so sum equals pair_CO_max."""
        pair_co = 1.10
        ppp = _ppp(pair_caps={"CO": pair_co, "MG": 1.10, "YV": 1.10})
        # C=0.70, O=0.60 -> sum=1.30 > 1.10; scale=1.10/1.30 ≈ 0.8462
        z = _patch([0.70, 0.10, 0.10, 0.10, 0.60, 0.10, 0.10])
        out = project_to_feasible(z, ppp)
        c_out = float(out[0, 0, 0])
        o_out = float(out[0, 0, 4])
        co_sum = c_out + o_out
        expected_scale = pair_co / 1.30
        # Proportional: C' / O' == C / O
        self.assertAlmostEqual(co_sum, pair_co, places=5, msg="CO sum must equal pair cap")
        self.assertAlmostEqual(c_out / o_out, 0.70 / 0.60, places=4, msg="Proportion C/O preserved")

    def test_pair_cap_mg_proportional_scaling(self):
        """M+G > pair_MG_max: proportional scaling, M/G ratio preserved."""
        pair_mg = 1.05
        ppp = _ppp(pair_caps={"CO": 1.10, "MG": pair_mg, "YV": 1.10})
        # M=0.65, G=0.55 -> sum=1.20 > 1.05
        z = _patch([0.10, 0.65, 0.10, 0.10, 0.10, 0.55, 0.10])
        out = project_to_feasible(z, ppp)
        m_out = float(out[0, 0, 1])
        g_out = float(out[0, 0, 5])
        self.assertAlmostEqual(m_out + g_out, pair_mg, places=5)
        self.assertAlmostEqual(m_out / g_out, 0.65 / 0.55, places=4)

    def test_pair_cap_yv_proportional_scaling(self):
        """Y+V > pair_YV_max: proportional scaling."""
        pair_yv = 1.00
        ppp = _ppp(pair_caps={"CO": 1.10, "MG": 1.10, "YV": pair_yv})
        # Y=0.55, V=0.55 -> sum=1.10 > 1.00
        z = _patch([0.10, 0.10, 0.55, 0.10, 0.10, 0.10, 0.55])
        out = project_to_feasible(z, ppp)
        y_out = float(out[0, 0, 2])
        v_out = float(out[0, 0, 6])
        self.assertAlmostEqual(y_out + v_out, pair_yv, places=5)
        self.assertAlmostEqual(y_out / v_out, 1.0, places=4)

    def test_pair_cap_within_limit_unchanged(self):
        """Pair caps don't fire when already within limit."""
        ppp = _ppp(pair_caps={"CO": 1.10, "MG": 1.10, "YV": 1.10})
        # All pairs well within caps, TAC=0.6 ok
        z = _patch([0.5, 0.1, 0.0, 0.0, 0.5, 0.0, 0.0])
        out = project_to_feasible(z, ppp)
        # C+O = 1.0 <= 1.10 so unchanged
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.5, places=5)
        self.assertAlmostEqual(float(out[0, 0, 4]), 0.5, places=5)

    def test_pair_caps_empty_dict_no_effect(self):
        """Empty pair_caps dict means no pair constraints are enforced."""
        ppp = _ppp(pair_caps={})
        # C=0.9, O=0.9 -> sum=1.8, but no pair cap -> only OGV cap applies
        z = _patch([0.9, 0.1, 0.1, 0.1, 0.9, 0.0, 0.0])
        out = project_to_feasible(z, ppp)
        # C unchanged (no CO pair constraint), O may be reduced by OGV cap
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.9, places=5, msg="C must not change without CO pair cap")

    # ── 3. OGV total cap ─────────────────────────────────────────────────

    def test_ogv_cap_scales_all_three_proportionally(self):
        """O+G+V > ogv_max: all three scaled by the same factor."""
        ogv_max = 0.85
        ppp = _ppp(ogv_max=ogv_max)
        # O=0.4, G=0.3, V=0.4 -> sum=1.10 > 0.85; scale=0.85/1.10
        z = _patch([0.10, 0.10, 0.10, 0.10, 0.40, 0.30, 0.40])
        out = project_to_feasible(z, ppp)
        o, g, v = float(out[0, 0, 4]), float(out[0, 0, 5]), float(out[0, 0, 6])
        ogv_sum = o + g + v
        self.assertAlmostEqual(ogv_sum, ogv_max, places=5, msg="OGV sum must equal ogv_max")
        expected_scale = ogv_max / 1.10
        self.assertAlmostEqual(o / 0.40, expected_scale, places=4, msg="O scaling factor must match")
        self.assertAlmostEqual(g / 0.30, expected_scale, places=4, msg="G scaling factor must match")
        self.assertAlmostEqual(v / 0.40, expected_scale, places=4, msg="V scaling factor must match")

    def test_ogv_within_cap_unchanged(self):
        """O+G+V within ogv_max -> OGV channels not touched."""
        ppp = _ppp(ogv_max=0.85)
        z = _patch([0.10, 0.10, 0.10, 0.10, 0.20, 0.20, 0.20])
        out = project_to_feasible(z, ppp)
        # OGV sum=0.60 < 0.85 -> no change
        for ch in [4, 5, 6]:
            self.assertAlmostEqual(float(out[0, 0, ch]), 0.20, places=5)

    # ── 4. Neutral/dark OGV cap ──────────────────────────────────────────

    def test_neutral_dark_ogv_cap_by_chroma(self):
        """Low-chroma pixels (a*^2 + b*^2 <= threshold^2) get neutral_OGV_max cap."""
        neutral_ogv_max = 0.05
        neutral_chroma_threshold = 5.0  # C_lab <= 5 triggers
        ppp = _ppp(
            ogv_max=0.85, neutral_ogv_max=neutral_ogv_max,
            neutral_chroma_threshold=neutral_chroma_threshold,
        )
        # Lab = [50, 1, 1] -> chroma=sqrt(2)≈1.41 <= 5 -> is neutral
        # O=0.3, G=0.2, V=0.2 -> OGV=0.7 > neutral_ogv_max=0.05
        lab_ref = np.zeros((1, 1, 3), dtype=np.float32)
        lab_ref[0, 0] = [50.0, 1.0, 1.0]  # low chroma
        z = _patch([0.10, 0.10, 0.10, 0.10, 0.30, 0.20, 0.20])
        out = project_to_feasible(z, ppp, lab_ref=lab_ref)
        ogv_out = float(out[0, 0, 4]) + float(out[0, 0, 5]) + float(out[0, 0, 6])
        self.assertLessEqual(
            ogv_out, neutral_ogv_max + 1e-5,
            msg=f"OGV must be <= neutral_ogv_max={neutral_ogv_max} for neutral pixel; got {ogv_out}",
        )
        # Verify proportional scaling preserved relative OGV ratios
        o, g, v = float(out[0, 0, 4]), float(out[0, 0, 5]), float(out[0, 0, 6])
        self.assertAlmostEqual(o / g, 0.30 / 0.20, places=3, msg="O/G ratio preserved")
        self.assertAlmostEqual(o / v, 0.30 / 0.20, places=3, msg="O/V ratio preserved")

    def test_neutral_dark_ogv_cap_by_luminance(self):
        """Low-L pixels (L <= dark_L_threshold) get neutral_OGV_max cap."""
        neutral_ogv_max = 0.05
        dark_l_threshold = 18.0
        ppp = _ppp(
            ogv_max=0.85, neutral_ogv_max=neutral_ogv_max,
            dark_l_threshold=dark_l_threshold,
        )
        # Lab = [10, 0, 0] -> L=10 <= 18 -> is dark
        lab_ref = np.zeros((1, 1, 3), dtype=np.float32)
        lab_ref[0, 0] = [10.0, 0.0, 0.0]
        z = _patch([0.10, 0.10, 0.10, 0.30, 0.25, 0.25, 0.25])
        out = project_to_feasible(z, ppp, lab_ref=lab_ref)
        ogv_out = float(out[0, 0, 4]) + float(out[0, 0, 5]) + float(out[0, 0, 6])
        self.assertLessEqual(
            ogv_out, neutral_ogv_max + 1e-5,
            msg=f"Dark pixel OGV must be capped to {neutral_ogv_max}; got {ogv_out}",
        )

    def test_non_neutral_pixel_not_capped_by_neutral_ogv_max(self):
        """High-chroma, bright pixels are NOT affected by the neutral OGV cap."""
        ppp = _ppp(ogv_max=0.85, neutral_ogv_max=0.05)
        # Lab = [60, 40, -30] -> chroma=sqrt(1600+900)=50 > 5 -> NOT neutral
        lab_ref = np.zeros((1, 1, 3), dtype=np.float32)
        lab_ref[0, 0] = [60.0, 40.0, -30.0]
        # OGV=0.3+0.2+0.2=0.70 <= 0.85 (within ogv_max, but > neutral_ogv_max)
        z = _patch([0.10, 0.10, 0.10, 0.10, 0.30, 0.20, 0.20])
        out = project_to_feasible(z, ppp, lab_ref=lab_ref)
        ogv_out = float(out[0, 0, 4]) + float(out[0, 0, 5]) + float(out[0, 0, 6])
        # Should NOT be capped to neutral_ogv_max (0.05), since pixel is not neutral
        self.assertGreater(
            ogv_out, 0.05 + 1e-4,
            msg=f"Non-neutral pixel must not be limited to neutral_ogv_max; got {ogv_out}",
        )

    def test_neutral_dark_ogv_cap_pixel_map(self):
        """Neutral/dark cap applies pixel-wise; a non-neutral pixel in the same patch is unaffected."""
        ppp = _ppp(ogv_max=0.85, neutral_ogv_max=0.05)
        # 2x1 patch: pixel[0] is neutral (low chroma), pixel[1] is colourful
        z = np.zeros((2, 1, 7), dtype=np.float32)
        z[0, 0] = [0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.2]  # neutral -> OGV capped
        z[1, 0] = [0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.2]  # colourful -> OGV NOT capped

        lab_ref = np.zeros((2, 1, 3), dtype=np.float32)
        lab_ref[0, 0] = [50.0, 1.0, 1.0]   # neutral  (chroma=sqrt(2)≈1.41 <=5)
        lab_ref[1, 0] = [60.0, 40.0, -30.0]  # colourful (chroma=50)

        out = project_to_feasible(z, ppp, lab_ref=lab_ref)
        ogv_neutral = out[0, 0, 4] + out[0, 0, 5] + out[0, 0, 6]
        ogv_colorful = out[1, 0, 4] + out[1, 0, 5] + out[1, 0, 6]

        self.assertLessEqual(float(ogv_neutral), 0.05 + 1e-5, "Neutral pixel must be OGV capped")
        self.assertGreater(float(ogv_colorful), 0.05, "Colourful pixel must NOT be OGV-neutral capped")

    # ── 5. TAC capped-simplex ─────────────────────────────────────────────

    def test_tac_capped_simplex_lands_exactly_on_tac_max(self):
        """After projection, pixel TAC = tac_max (within tol)."""
        tac_max = 3.0
        ppp = _ppp(tac_max=tac_max, pair_caps={})
        # TAC = 7 (all channels at 1.0) -> needs aggressive reduction
        z = _patch([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        out = project_to_feasible(z, ppp)
        tac_out = float(out[0, 0].sum())
        self.assertAlmostEqual(tac_out, tac_max, places=4, msg="TAC must equal tac_max after projection")

    def test_tac_proportional_reduction_preserves_relative_order(self):
        """The capped-simplex reduces all channels proportionally (no channel increases)."""
        ppp = _ppp(tac_max=2.5, pair_caps={})
        z = _patch([0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])  # TAC=2.9 > 2.5
        out = project_to_feasible(z, ppp)
        for j in range(7):
            self.assertLessEqual(
                float(out[0, 0, j]), float(z[0, 0, j]) + 1e-6,
                msg=f"Channel {j} must not increase during TAC projection",
            )
        tac_out = float(out[0, 0].sum())
        self.assertAlmostEqual(tac_out, 2.5, places=4)

    def test_tac_within_limit_unchanged(self):
        """Pixels with TAC <= tac_max are not touched by the TAC step."""
        ppp = _ppp(tac_max=3.0, pair_caps={})
        z = _patch([0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1])  # TAC=1.4
        out = project_to_feasible(z, ppp)
        for j in range(7):
            self.assertAlmostEqual(float(out[0, 0, j]), float(z[0, 0, j]), places=5)

    # ── 6. Idempotency (feasibility preservation) ─────────────────────────

    def test_feasible_pixel_is_exactly_preserved(self):
        """A pixel already in K(PPP) comes out unchanged (idempotency of Pi_K)."""
        ppp = _ppp(tac_max=3.0, ogv_max=0.85, pair_caps={"CO": 1.10, "MG": 1.10, "YV": 1.10})
        # Carefully chosen feasible pixel
        z = _patch([0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.1])
        # TAC=1.4, OGV=0.4, C+O=0.6, M+G=0.4, Y+V=0.3 — all within caps
        out1 = project_to_feasible(z, ppp)
        out2 = project_to_feasible(out1, ppp)
        np.testing.assert_allclose(
            out1, out2, atol=1e-6,
            err_msg="Pi_K must be idempotent: applying twice gives same result",
        )

    def test_projection_idempotent_after_neutral_dark_cap(self):
        """Applying Pi_K twice to a neutral pixel gives identical output."""
        ppp = _ppp(neutral_ogv_max=0.05)
        lab_ref = np.zeros((1, 1, 3), dtype=np.float32)
        lab_ref[0, 0] = [50.0, 1.0, 1.0]  # neutral
        z = _patch([0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.2])
        out1 = project_to_feasible(z, ppp, lab_ref=lab_ref)
        out2 = project_to_feasible(out1, ppp, lab_ref=lab_ref)
        np.testing.assert_allclose(out1, out2, atol=1e-6)

    # ── 7. Projection ordering invariant ─────────────────────────────────

    def test_pair_cap_then_ogv_then_tac_ordering(self):
        """Projection order: pair caps applied BEFORE OGV cap BEFORE TAC.

        The key invariant is that *all* constraints are satisfied after the
        projection — not that each individual step lands at its boundary.
        After the CO pair step, O is reduced to pair_cap/C+O * O_prev.  The
        subsequent OGV step may further reduce O (because O, G, V together
        still exceed ogv_max), so the final CO sum will be < pair_CO_max
        (which is still valid — the constraint is <=, not ==).

        Pixel: C=0.60, O=0.60 (pair_CO=1.20 > cap=1.10), G=0.50, V=0.50

        Step 1 (CO pair): scale=1.10/1.20; C'=0.55, O'=0.55
        Step 2 (OGV):     O'+G+V = 0.55+0.50+0.50 = 1.55 > 0.85
                          scale=0.85/1.55; O''≈0.302, G'≈0.274, V'≈0.274
        Step 3 (TAC):     after step 2, TAC=0.55+0+0+0+0.302+0.274+0.274≈1.40 <= 3.0 → no change.

        All constraints must be satisfied; CO pair sum will be < 1.10 (not ==).
        """
        ppp = _ppp(
            tac_max=3.0, ogv_max=0.85, neutral_ogv_max=0.05,
            pair_caps={"CO": 1.10, "MG": 1.10, "YV": 1.10},
        )
        z = _patch([0.60, 0.0, 0.0, 0.0, 0.60, 0.50, 0.50])
        out = project_to_feasible(z, ppp)
        c_out = float(out[0, 0, 0])
        o_out = float(out[0, 0, 4])
        g_out = float(out[0, 0, 5])
        v_out = float(out[0, 0, 6])

        # CO pair cap satisfied (pair step satisfied it; OGV step can only lower O further)
        self.assertLessEqual(c_out + o_out, 1.10 + 1e-5,
                             msg=f"CO sum {c_out+o_out:.5f} must be <= pair_CO_max=1.10")
        # C must not increase beyond the pair-capped value (0.55)
        self.assertAlmostEqual(c_out, 0.55, places=4,
                               msg="C must equal pair-capped value 0.55 (OGV step doesn't change C)")
        # OGV cap satisfied
        self.assertLessEqual(o_out + g_out + v_out, 0.85 + 1e-5,
                             msg=f"OGV sum {o_out+g_out+v_out:.5f} must be <= ogv_max=0.85")
        # TAC satisfied
        tac_out = float(out[0, 0].sum())
        self.assertLessEqual(tac_out, 3.0 + 1e-5,
                             msg=f"TAC {tac_out:.5f} must be <= tac_max=3.0")


    # ── 8. feasibility_violations() counting ─────────────────────────────

    def test_feasibility_violations_all_types(self):
        """feasibility_violations detects all constraint types independently."""
        ppp = _ppp(
            tac_max=2.0, ogv_max=0.70,
            pair_caps={"CO": 0.90, "MG": 0.90, "YV": 0.90},
        )
        # Deliberately violate every constraint at once
        # C=0.6, M=0.6, Y=0.6, K=0.2, O=0.6, G=0.6, V=0.6
        # TAC=3.8 > 2.0; OGV=1.8 > 0.70; CO=1.2 > 0.90; MG=1.2 > 0.90; YV=1.2 > 0.90
        z = _patch([0.6, 0.6, 0.6, 0.2, 0.6, 0.6, 0.6])
        viols = feasibility_violations(z, ppp)
        self.assertGreater(viols["tac"], 0, "TAC violation not detected")
        self.assertGreater(viols["ogv"], 0, "OGV violation not detected")
        self.assertGreater(viols["pair_co"], 0, "CO pair violation not detected")
        self.assertGreater(viols["pair_mg"], 0, "MG pair violation not detected")
        self.assertGreater(viols["pair_yv"], 0, "YV pair violation not detected")

    def test_feasibility_violations_zero_after_projection(self):
        """After project_to_feasible all violation counts are zero."""
        ppp = _ppp(tac_max=2.0, ogv_max=0.70, pair_caps={"CO": 0.90, "MG": 0.90, "YV": 0.90})
        z = _patch([0.6, 0.6, 0.6, 0.2, 0.6, 0.6, 0.6])
        out = project_to_feasible(z, ppp)
        viols = feasibility_violations(out, ppp)
        for name, count in viols.items():
            self.assertEqual(
                count, 0,
                msg=f"Violation '{name}'={count} after projection (should be 0)",
            )

    # ── 9. validate() rejects invalid pair key ────────────────────────────

    def test_validate_rejects_invalid_pair_key(self):
        """PPP.validate() raises ValueError for unknown pair_caps keys."""
        with self.assertRaises(ValueError, msg="'INVALID_KEY' should be rejected"):
            p = PPP(
                base_family="film_generic_conservative",
                caps={"C": 1.0, "M": 1.0, "Y": 1.0, "K": 1.0,
                      "O": 0.55, "G": 0.55, "V": 0.55},
                tac_max=3.0, ogv_max=0.85,
                pair_caps={"INVALID_KEY": 1.0},
                neutral_ogv_max=0.05, neutral_chroma_threshold=5.0,
                dark_l_threshold=18.0, risk_threshold=3.0,
                risk_threshold_hard=False, override_mask={},
            )
            p.validate()

    def test_validate_accepts_valid_pair_keys(self):
        """PPP.validate() succeeds with CO, MG, YV pair keys."""
        p = _ppp(pair_caps={"CO": 1.10, "MG": 1.10, "YV": 1.10})
        self.assertIsNone(p.validate())  # no exception

    def test_validate_accepts_empty_pair_caps(self):
        """Empty pair_caps is valid (no pair constraints)."""
        p = _ppp(pair_caps={})
        self.assertIsNone(p.validate())

    # ── 10. Batch (multi-pixel patch) correctness ────────────────────────

    def test_projection_broadcast_multi_pixel(self):
        """Projection works correctly on a full (H,W,7) patch."""
        ppp = _ppp(tac_max=2.5)
        z = np.random.default_rng(42).uniform(0, 1, (16, 16, 7)).astype(np.float32)
        out = project_to_feasible(z, ppp)
        # All pixels must satisfy constraints
        self.assertTrue(np.all(out >= -1e-6), "No negative values after projection")
        self.assertTrue(np.all(out <= 1.0 + 1e-5), "No values > 1 after projection")
        tac = out.sum(axis=-1)
        self.assertTrue(
            np.all(tac <= 2.5 + 1e-4),
            f"TAC violation after projection: max={tac.max():.5f}",
        )
        ogv = out[..., 4:].sum(axis=-1)
        self.assertTrue(np.all(ogv <= ppp.ogv_max + 1e-5))


if __name__ == "__main__":
    unittest.main()
