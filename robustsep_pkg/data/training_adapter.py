"""robustsep_pkg.data.training_adapter — multi-source epoch iterator.

The :class:`TrainingAdapter` wraps one or more :class:`RobustSepDataset`
objects (which may come from different shard families: ``robustsep``,
``doclaynet``, ``sku110k``) and presents a single unified epoch-level
interface for training loops.

Epoch iteration
---------------
Each call to :meth:`iter_epoch` produces a deterministic sequence of
:class:`~robustsep_pkg.data.batching.ShardBatch` objects:

1. Collect all :class:`~robustsep_pkg.data.shard_reader.ShardReader` objects
   from every registered dataset, grouped by family name.
2. Run :func:`~robustsep_pkg.data.source_weighting.weighted_shard_schedule`
   to produce a deterministic interleaved shard order for this epoch.
3. For each shard in order: load arrays + records, then emit batches via
   :func:`~robustsep_pkg.data.batching.iter_batches`.

Manifest export
---------------
:meth:`export_split_manifest` writes a JSON file summarising which shard
files belong to which split, suitable for audit and replay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from robustsep_pkg.data.batching import ShardBatch, iter_batches
from robustsep_pkg.data.dataset import RobustSepDataset
from robustsep_pkg.data.shard_reader import ShardReader
from robustsep_pkg.data.source_weighting import SourceWeightPolicy, weighted_shard_schedule


# ---------------------------------------------------------------------------
# FamilyDataset — dataset tagged with a family name
# ---------------------------------------------------------------------------


@dataclass
class FamilyDataset:
    """A :class:`RobustSepDataset` tagged with its source-family name.

    Parameters
    ----------
    name:
        Short identifier used for weighting (e.g. ``"robustsep"``,
        ``"doclaynet"``, ``"sku110k"``).
    dataset:
        The underlying dataset for a specific split.
    """
    name: str
    dataset: RobustSepDataset


# ---------------------------------------------------------------------------
# TrainingAdapter
# ---------------------------------------------------------------------------


class TrainingAdapter:
    """Multi-source, epoch-level training iterator.

    Parameters
    ----------
    families:
        Named dataset sources.  Each is a :class:`FamilyDataset` (name +
        underlying :class:`RobustSepDataset`).
    weight_policy:
        Source priority weights for interleaving shard schedules.  Full
        epochs still visit every nonzero-weight shard once.
    batch_size:
        Number of patches per :class:`~robustsep_pkg.data.batching.ShardBatch`.
    drop_last:
        Drop the final incomplete batch from each shard.
    root_seed:
        Master seed for epoch-level shuffle determinism.

    Examples
    --------
    >>> from robustsep_pkg.data import RobustSepDataset
    >>> from robustsep_pkg.data.training_adapter import TrainingAdapter, FamilyDataset
    >>> ds = RobustSepDataset(["manifests/robustsep_run_manifest.json"], split="train")
    >>> adapter = TrainingAdapter([FamilyDataset("robustsep", ds)], batch_size=64)
    >>> for epoch in range(10):
    ...     for batch in adapter.iter_epoch(epoch):
    ...         train_step(batch.tensors())  # noqa: F821
    """

    def __init__(
        self,
        families: list[FamilyDataset],
        *,
        weight_policy: SourceWeightPolicy | None = None,
        batch_size: int = 64,
        drop_last: bool = False,
        root_seed: int = 20260422,
    ) -> None:
        if not families:
            raise ValueError("TrainingAdapter requires at least one FamilyDataset")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._families = families
        self._policy = weight_policy or SourceWeightPolicy()
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._root_seed = root_seed

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def total_patches(self) -> int:
        """Total number of patches across all registered families."""
        return sum(len(fd.dataset) for fd in self._families)

    @property
    def num_shards(self) -> int:
        """Total number of shards across all registered families."""
        return sum(fd.dataset.num_shards for fd in self._families)

    def family_names(self) -> list[str]:
        """Names of registered source families."""
        return [fd.name for fd in self._families]

    # ------------------------------------------------------------------
    # Epoch iteration
    # ------------------------------------------------------------------

    def _readers_by_family(self) -> dict[str, list[ShardReader]]:
        out: dict[str, list[ShardReader]] = {}
        for fd in self._families:
            out.setdefault(fd.name, []).extend(fd.dataset.shard_readers())
        return out

    def iter_epoch(
        self,
        epoch: int,
        *,
        batch_size: int | None = None,
    ) -> Iterator[ShardBatch]:
        """Yield :class:`~robustsep_pkg.data.batching.ShardBatch` for one epoch.

        The shard visit order is deterministic for a given ``(epoch, root_seed)``
        pair and respects the :class:`~robustsep_pkg.data.source_weighting.SourceWeightPolicy`.
        Arrays are loaded one shard at a time (memory-bounded).

        Parameters
        ----------
        epoch:
            Current epoch index (0-based).  Different epochs produce different
            shard orders; the same epoch index always produces the same order.
        batch_size:
            Override the adapter's default batch size for this call only.
        """
        bs = batch_size if batch_size is not None else self._batch_size
        schedule = weighted_shard_schedule(
            self._readers_by_family(),
            self._policy,
            epoch=epoch,
            root_seed=self._root_seed,
        )
        for reader in schedule:
            arrays = reader.load_arrays()
            records = reader.load_records()
            yield from iter_batches(arrays, records, bs, drop_last=self._drop_last)

    # ------------------------------------------------------------------
    # Manifest export
    # ------------------------------------------------------------------

    def export_split_manifest(
        self,
        path: str | Path,
        *,
        alpha_policy: str = "ones",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write a JSON summary of all families, weight policy, and alpha policy.

        This is the canonical handoff document for Codex's target-manifest
        generator.  It contains every parameter needed to reproduce the exact
        weighted shard ordering semantics and enrichment settings without
        instantiating a :class:`TrainingAdapter`.

        Parameters
        ----------
        path:
            Output ``.json`` path.
        alpha_policy:
            The alpha fallback policy used during training (``\"ones\"``,
            ``\"visible_threshold\"``, or ``\"passthrough\"``).  Embedded in the
            manifest so the target generator can apply the same alpha policy
            when calling :func:`~robustsep_pkg.data.enrichment.enrich_sample`.
        extra:
            Additional key-value pairs merged into the top-level JSON object.
        """
        _VALID_ALPHA = {"ones", "visible_threshold", "passthrough"}
        if alpha_policy not in _VALID_ALPHA:
            raise ValueError(
                f"alpha_policy must be one of {sorted(_VALID_ALPHA)}, "
                f"got {alpha_policy!r}"
            )

        all_family_names = self.family_names()
        families_out = []
        for fd in self._families:
            shards_out = []
            for reader in fd.dataset.shard_readers():
                shards_out.append({
                    "npz": reader.entry.npz,
                    "jsonl": reader.entry.jsonl,
                    "count": reader.entry.count,
                    "npz_sha256": reader.entry.npz_sha256,
                    "jsonl_sha256": reader.entry.jsonl_sha256,
                })
            families_out.append({
                "name": fd.name,
                "split": fd.dataset.split,
                "num_shards": fd.dataset.num_shards,
                "total_patches": len(fd.dataset),
                "shards": shards_out,
            })

        payload: dict[str, Any] = {
            "split_manifest_version": "1.1",
            "root_seed": self._root_seed,
            "batch_size": self._batch_size,
            "drop_last": self._drop_last,
            "total_patches": self.total_patches,
            "total_shards": self.num_shards,
            # Weight policy: full summary for manifest reproducibility.
            # Codex's target-manifest generator must read this to reconstruct
            # schedule ordering semantics without instantiating a TrainingAdapter.
            "source_weight_policy": self._policy.policy_summary(all_family_names),
            # Alpha policy: the enrichment policy applied during this run.
            # Codex's generator must pass the same value to enrich_sample().
            "alpha_policy": alpha_policy,
            "families": families_out,
            **(extra or {}),
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def summary(self) -> dict[str, Any]:
        """Compact summary dict for logging."""
        return {
            "families": [
                {"name": fd.name, "split": fd.dataset.split,
                 "num_shards": fd.dataset.num_shards, "total_patches": len(fd.dataset)}
                for fd in self._families
            ],
            "total_patches": self.total_patches,
            "total_shards": self.num_shards,
            "batch_size": self._batch_size,
            "root_seed": self._root_seed,
        }
