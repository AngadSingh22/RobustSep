"""robustsep_pkg.data.enrichment — optional per-sample intent/structure enrichment.

This module attaches intent weights, structure tokens, and a low-resolution
intent raster to a :class:`~robustsep_pkg.data.shard_record.ShardSample`
when they are not pre-stored in the shard.

It is *optional* in the data pipeline — shards that already carry validated
intent/structure annotations can be consumed directly from the loader without
re-computing them.

Alpha fallback policy
---------------------
Current staged shards synthesise alpha as all-ones (``ShardArrays`` sets
``alpha = np.ones(...)`` when the ``.npz`` has no ``alpha`` key).  Three
fallback policies are supported:

``"ones"``
    Keep the all-ones synthetic alpha as-is (current behaviour).
``"visible_threshold"``
    Re-threshold: pixels with Lab L* > ``alpha_l_min`` are treated as fully
    visible (alpha=1).  Everything else gets alpha=0.  This heuristic is
    useful when a light background bleeds into the staging filter.
``"passthrough"``
    Use whatever alpha is stored in the shard with no modification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robustsep_pkg.core.config import PreprocessConfig
from robustsep_pkg.data.intent_adapter import (
    compute_intent_weights,
    compute_low_res_intent_raster,
    compute_structure_token,
)
from robustsep_pkg.data.shard_record import ShardSample


# ---------------------------------------------------------------------------
# Alpha fallback
# ---------------------------------------------------------------------------

_ALPHA_POLICIES = frozenset({"ones", "visible_threshold", "passthrough"})


def apply_alpha_fallback(
    sample: ShardSample,
    policy: str = "ones",
    *,
    alpha_l_min: float = 10.0,
) -> np.ndarray:
    """Return an alpha array for *sample* according to *policy*.

    Parameters
    ----------
    sample:
        The loaded :class:`~robustsep_pkg.data.shard_record.ShardSample`.
    policy:
        One of ``"ones"``, ``"visible_threshold"``, ``"passthrough"``.
    alpha_l_min:
        L* threshold used only when ``policy == "visible_threshold"``.

    Returns
    -------
    ``np.ndarray`` shape ``(16, 16)`` float32, values in [0, 1].
    """
    if policy not in _ALPHA_POLICIES:
        raise ValueError(f"Unknown alpha fallback policy {policy!r}; "
                         f"valid options: {sorted(_ALPHA_POLICIES)}")
    if policy == "ones":
        return np.ones(sample.rgb.shape[:2], dtype=np.float32)
    elif policy == "visible_threshold":
        l_star = sample.lab[..., 0]
        return (l_star > alpha_l_min).astype(np.float32)
    else:  # "passthrough"
        return np.clip(sample.alpha.astype(np.float32), 0.0, 1.0)


# ---------------------------------------------------------------------------
# EnrichmentConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentConfig:
    """Configuration for optional per-sample enrichment.

    Parameters
    ----------
    recompute_intent:
        If ``True``, (re)compute intent weights and raster from RGB/alpha.
        If ``False``, intent_weights and intent_raster are ``None`` in the
        output unless the record already has them.
    recompute_structure:
        If ``True``, (re)compute structure token.  If ``False``, uses the
        ``structure`` field from :class:`~robustsep_pkg.data.shard_record.ShardRecord`.
    alpha_policy:
        Alpha fallback policy — ``"ones"``, ``"visible_threshold"``,
        or ``"passthrough"``.
    alpha_l_min:
        L* threshold for ``"visible_threshold"`` policy.
    intent_raster_size:
        Grid size for the low-resolution intent raster (4 → 4×4 cells).
    preprocess_config:
        Forwarded to intent/structure computation.
    """

    recompute_intent: bool = False
    recompute_structure: bool = False
    alpha_policy: str = "ones"
    alpha_l_min: float = 10.0
    intent_raster_size: int = 4
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)


# ---------------------------------------------------------------------------
# EnrichedSample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichedSample:
    """A :class:`~robustsep_pkg.data.shard_record.ShardSample` plus enrichment fields.

    Parameters
    ----------
    sample:
        The original loader sample (RGB, alpha, Lab, CMYK, CMYKOGV, record).
    alpha_effective:
        Alpha after applying the configured fallback policy.
    intent_weights:
        Per-patch aggregated intent scalars ``{brand, gradient, flat}``.
        ``None`` if ``recompute_intent=False`` and no cached value is available.
    intent_raster:
        ``(raster_size, raster_size, 3)`` float32 intent raster.
        ``None`` under the same conditions as *intent_weights*.
    structure_token:
        ``"edge"``, ``"flat"``, or ``"textured"``.  Taken from the shard
        record when ``recompute_structure=False``.
    """

    sample: ShardSample
    alpha_effective: np.ndarray
    intent_weights: dict[str, float] | None
    intent_raster: np.ndarray | None
    structure_token: str

    # Convenience delegations
    @property
    def rgb(self) -> np.ndarray:
        return self.sample.rgb

    @property
    def lab(self) -> np.ndarray:
        return self.sample.lab

    @property
    def cmykogv_baseline(self) -> np.ndarray:
        return self.sample.cmykogv_baseline

    @property
    def record(self):
        return self.sample.record

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = self.sample.record.to_dict()
        out["intent_weights"] = self.intent_weights
        out["structure_token"] = self.structure_token
        if self.intent_raster is not None:
            out["intent_raster_shape"] = list(self.intent_raster.shape)
        return out


# ---------------------------------------------------------------------------
# enrich_sample
# ---------------------------------------------------------------------------


def enrich_sample(
    sample: ShardSample,
    config: EnrichmentConfig = EnrichmentConfig(),
) -> EnrichedSample:
    """Attach intent/structure annotations to a loaded sample.

    Parameters
    ----------
    sample:
        Raw sample from :meth:`~robustsep_pkg.data.dataset.RobustSepDataset.iter_samples`.
    config:
        Enrichment options.

    Returns
    -------
    :class:`EnrichedSample` with all configured fields populated.
    """
    pc = config.preprocess_config
    alpha_eff = apply_alpha_fallback(sample, config.alpha_policy, alpha_l_min=config.alpha_l_min)

    # ── Structure token ────────────────────────────────────────────────
    if config.recompute_structure:
        structure = compute_structure_token(
            sample.rgb, alpha_eff, x=0, y=0, config=pc
        )
    else:
        structure = sample.record.structure

    # ── Intent weights + raster ────────────────────────────────────────
    intent_weights: dict[str, float] | None = None
    intent_raster: np.ndarray | None = None

    if config.recompute_intent:
        intent_weights = compute_intent_weights(
            sample.rgb, alpha_eff, x=0, y=0, config=pc
        )
        intent_raster = compute_low_res_intent_raster(
            sample.rgb, alpha_eff,
            x=0, y=0,
            patch_size=pc.patch_size,
            raster_size=config.intent_raster_size,
            config=pc,
        )

    return EnrichedSample(
        sample=sample,
        alpha_effective=alpha_eff,
        intent_weights=intent_weights,
        intent_raster=intent_raster,
        structure_token=structure,
    )
