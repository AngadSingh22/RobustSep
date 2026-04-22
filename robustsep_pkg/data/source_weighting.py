"""robustsep_pkg.data.source_weighting — per-family shard interleaving.

Weight policy
-------------
Each shard *family* (e.g. ``"robustsep"``, ``"doclaynet"``, ``"sku110k"``)
has a non-negative weight.  The weighted shard schedule is built by a
deterministic reservoir-style algorithm: each shard is assigned a priority
value ``U ~ Uniform(0,1)`` drawn from a seeded RNG, raised to the power
``1 / weight``.  Sorting by descending priority produces a weighted ordering
without replacement.

This is the "A-Res" algorithm (Vitter 1985) and is fully deterministic for a
given ``(root_seed, epoch)``.

Weight semantics
----------------
Weights are **relative priorities** — only their ratios matter::

    prefix_priority(family F) ∝ weight_F

Full-epoch iteration includes every nonzero-weight shard exactly once.  The
ratios affect interleaving order and partial-epoch/prefix exposure; they do
not downsample counts unless the caller consumes only part of the schedule.

``default_weight=0.0`` (opt-in model): families not listed in *weights* are
excluded.  Set to ``1.0`` for opt-out behaviour (unlisted = fully included).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from robustsep_pkg.data.shard_reader import ShardReader


# ---------------------------------------------------------------------------
# SourceWeightPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceWeightPolicy:
    """Per-family priority weights for multi-source training.

    Weights are **relative priorities** — only their ratios matter, not their
    absolute values.  Conceptually::

        prefix_priority(family F) ∝ weight_F

    This is enforced via the A-Res (Vitter 1985) weighted reservoir algorithm
    without replacement.  Full schedules include every nonzero-weight shard
    once; the weights bias ordering and therefore any prefix/partial epoch.
    The resulting shard schedule is deterministic for a given
    ``(root_seed, epoch)``.

    Parameters
    ----------
    weights:
        Dict mapping family name -> non-negative float weight.
        Examples:

        - ``{"robustsep": 1.0, "doclaynet": 0.5}`` — RobustSep shards are
          prioritized ahead of DocLayNet shards in schedule prefixes.
        - ``{"robustsep": 2.0, "sku110k": 1.0}`` — equivalent if only two
          families are present (only ratios matter).
        - Weight ``0.0`` or absence from the dict excludes the family.
    default_weight:
        Weight for families **not** listed in *weights*.
        Defaults to ``0.0`` (**opt-in** model — unlisted families are excluded).
        Set ``default_weight=1.0`` for opt-out behaviour.
    """

    weights: dict[str, float] = field(default_factory=lambda: {
        "robustsep": 1.0,
        "doclaynet": 0.5,
        "sku110k": 0.5,
    })
    default_weight: float = 0.0  # opt-in: unlisted families are excluded

    def get(self, family: str) -> float:
        """Return the weight for *family*, using ``default_weight`` if absent."""
        w = self.weights.get(family, self.default_weight)
        if w < 0:
            raise ValueError(f"weight must be non-negative, got {w} for family {family!r}")
        return w

    def normalize_weights(self, families: list[str]) -> dict[str, float]:
        """Return effective relative priority fractions summing to 1.

        Useful for logging and manifest embedding.  These fractions describe
        priority ratios, not full-epoch counts.  Zero-weight families are
        included as ``0.0`` in the output.

        Parameters
        ----------
        families:
            Ordered list of family names to normalise over.
        """
        raw = {f: self.get(f) for f in families}
        total = sum(raw.values())
        if total == 0.0:
            return {f: 0.0 for f in families}
        return {f: raw[f] / total for f in families}

    def policy_summary(self, families: list[str]) -> dict:
        """Return a JSON-serialisable summary for embedding in manifests.

        Codex's target-manifest generator must embed this dict under the key
        ``"source_weight_policy"`` so that schedule ordering semantics are
        fully reproducible from the manifest alone.

        Parameters
        ----------
        families:
            All registered family names present in the adapter.
        """
        return {
            "algorithm": "A-Res (Vitter 1985)",
            "semantics": "weighted_priority_without_replacement",
            "default_weight": self.default_weight,
            "raw_weights": {f: self.get(f) for f in families},
            "normalized_priority_fractions": self.normalize_weights(families),
            # Backward-compatible v1.1 key.  Fractions are priorities, not
            # guaranteed full-epoch source counts.
            "normalized_fractions": self.normalize_weights(families),
        }


# ---------------------------------------------------------------------------
# Weighted shard schedule
# ---------------------------------------------------------------------------


def weighted_shard_schedule(
    readers_by_family: dict[str, list[ShardReader]],
    policy: SourceWeightPolicy,
    epoch: int,
    root_seed: int = 20260422,
) -> list[ShardReader]:
    """Return a deterministic interleaved list of ShardReaders.

    Uses the A-Res (reservoir) algorithm: each shard *i* in family *f*
    receives a priority::

        key_i = U_i ^ (1 / weight_f)      where U_i ~ Uniform(0, 1)

    Shards are returned sorted by descending key (highest priority first).
    With ``weight_f = 1.0`` this is plain random shuffle.  Smaller weights
    reduce a family's priority in early schedule prefixes.  Full schedules
    still contain every nonzero-weight shard exactly once.

    Parameters
    ----------
    readers_by_family:
        ``{family_name: [ShardReader, ...]}``.  Families with zero weight
        or empty reader lists are silently skipped.
    policy:
        Sampling weight specification.
    epoch:
        Current epoch number.  Combined with *root_seed* to give a unique
        RNG state per epoch so each epoch sees a different shard order.
    root_seed:
        Master seed for reproducibility.
    """
    # Combine epoch and root_seed into a single uint64 seed deterministically
    combined_seed = int(root_seed) ^ (int(epoch) * 2654435761)  # Knuth multiplicative hash
    combined_seed &= 0xFFFF_FFFF_FFFF_FFFF
    rng = np.random.default_rng(combined_seed)

    items: list[tuple[float, ShardReader]] = []
    for family, readers in readers_by_family.items():
        weight = policy.get(family)
        if weight == 0.0 or not readers:
            continue
        inv_w = 1.0 / weight
        for reader in readers:
            u = float(rng.uniform())
            # Avoid log(0) — u is always > 0 from numpy uniform(0,1) exclusive
            key = math.pow(u, inv_w) if u > 0 else 0.0
            items.append((key, reader))

    # Sort descending by key — deterministic because keys are unique with overwhelming probability
    items.sort(key=lambda t: t[0], reverse=True)
    return [reader for _, reader in items]
