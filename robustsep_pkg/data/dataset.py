from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from robustsep_pkg.core.artifact_io import read_json
from robustsep_pkg.data.shard_record import ShardEntry, ShardRecord, ShardSample
from robustsep_pkg.data.shard_reader import ShardArrays, ShardReader
from robustsep_pkg.data.split import SPLIT_TRAIN, SPLITS, deterministic_split


class RobustSepDataset:
    """Multi-shard dataset backed by staged .npz/.jsonl shard pairs.

    Supports multiple shard *families* (e.g. ``robustsep_patches``,
    ``doclaynet_patches``, ``sku110k_patches``) loaded from a shared list
    of run-manifest JSON files.  Shards across all families are pooled
    and then split deterministically into train / val / test.

    Parameters
    ----------
    manifest_paths:
        One or more ``*_run_manifest.json`` paths produced by the
        preparation scripts.
    root:
        Filesystem root prepended to relative paths stored inside
        manifests.  Defaults to the current working directory when
        ``None``.
    split:
        Which split to expose — ``"train"``, ``"val"``, or ``"test"``.
    val_frac, test_frac:
        Fractional sizes for validation and test splits (per shard family).
    root_seed:
        Master seed forwarded to :func:`~robustsep_pkg.data.split.deterministic_split`.

    Examples
    --------
    >>> ds = RobustSepDataset(
    ...     ["data/external/manifests/robustsep_patches_run_manifest.json"],
    ...     split="train",
    ... )
    >>> print(len(ds))   # number of patches in train split
    >>> for arrays, records in ds.iter_shards():
    ...     # arrays: ShardArrays  — rgb, lab, icc_cmyk, cmyk_baseline, cmykogv_baseline
    ...     # records: list[ShardRecord]
    ...     pass
    """

    def __init__(
        self,
        manifest_paths: list[str | Path],
        *,
        root: str | Path | None = None,
        split: str = SPLIT_TRAIN,
        val_frac: float = 0.10,
        test_frac: float = 0.05,
        root_seed: int = 20260422,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self._root = Path(root) if root is not None else Path(".")
        self._split = split
        self._val_frac = val_frac
        self._test_frac = test_frac
        self._root_seed = root_seed

        self._readers: list[ShardReader] = []
        self._manifest_paths: list[Path] = []
        self._total_patches: int = 0

        for mp in manifest_paths:
            self._load_manifest(Path(mp))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: Path) -> None:
        """Parse a run-manifest JSON and register the relevant shard readers."""
        if not manifest_path.is_absolute() and not manifest_path.exists():
            candidate = self._root / manifest_path
            if candidate.exists():
                manifest_path = candidate
        manifest: dict[str, Any] = read_json(manifest_path)
        self._manifest_paths.append(manifest_path)
        raw_shards: list[dict[str, Any]] = manifest.get("shards", [])
        if not raw_shards:
            return

        entries = [ShardEntry.from_dict(s) for s in raw_shards]
        # Scope the split by the output-directory basename so that
        # independent families with the same n_shards get different splits.
        scope = Path(manifest.get("out_dir", manifest_path.stem)).name
        split_map = deterministic_split(
            len(entries),
            val_frac=self._val_frac,
            test_frac=self._test_frac,
            root_seed=self._root_seed,
            scope=scope,
        )
        chosen_indices = split_map[self._split]
        for idx in chosen_indices:
            reader = ShardReader(entries[idx], root=self._root)
            self._readers.append(reader)
            self._total_patches += entries[idx].count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def split(self) -> str:
        """The active data split (``"train"``, ``"val"``, or ``"test"``)."""
        return self._split

    @property
    def num_shards(self) -> int:
        """Number of shards in this split."""
        return len(self._readers)

    def __len__(self) -> int:
        """Total number of patches across all shards in this split."""
        return self._total_patches

    def iter_shards(self) -> Iterator[tuple[ShardArrays, list[ShardRecord]]]:
        """Yield ``(ShardArrays, list[ShardRecord])`` for each shard.

        Arrays are loaded from disk on demand (one shard at a time) so
        memory usage is bounded by the largest single shard.
        """
        for reader in self._readers:
            arrays = reader.load_arrays()
            records = list(reader.iter_records())
            yield arrays, records

    def iter_samples(self) -> Iterator[ShardSample]:
        """Yield individual patch samples across selected shards.

        This is convenient for smoke tests and simple CPU pipelines. Training
        code should usually consume `iter_shards()` to avoid repeatedly loading
        the same shard.
        """
        for arrays, records in self.iter_shards():
            if len(records) != arrays.rgb.shape[0]:
                raise ValueError(f"metadata count {len(records)} does not match array count {arrays.rgb.shape[0]}")
            for i, record in enumerate(records):
                yield ShardSample(record=record, **arrays.sample(i))

    def __iter__(self) -> Iterator[ShardSample]:
        return self.iter_samples()

    def __getitem__(self, index: int) -> ShardSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        offset = 0
        for reader in self._readers:
            next_offset = offset + reader.count
            if offset <= index < next_offset:
                local_index = index - offset
                arrays = reader.load_arrays()
                records = reader.load_records()
                if local_index >= len(records):
                    raise IndexError(f"metadata for local index {local_index} missing in {reader.entry.jsonl}")
                return ShardSample(record=records[local_index], **arrays.sample(local_index))
            offset = next_offset
        raise IndexError(index)

    def shard_readers(self) -> list[ShardReader]:
        """Return the list of :class:`~robustsep_pkg.data.shard_reader.ShardReader` objects."""
        return list(self._readers)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict suitable for logging."""
        return {
            "split": self._split,
            "num_shards": self.num_shards,
            "total_patches": self._total_patches,
            "val_frac": self._val_frac,
            "test_frac": self._test_frac,
            "root_seed": self._root_seed,
            "manifests": [str(p) for p in self._manifest_paths],
        }

    @classmethod
    def from_manifest_dir(
        cls,
        manifest_dir: str | Path,
        *,
        glob: str = "*_run_manifest.json",
        split: str = SPLIT_TRAIN,
        **kwargs: Any,
    ) -> "RobustSepDataset":
        """Construct from all manifests matching *glob* in *manifest_dir*.

        Parameters
        ----------
        manifest_dir:
            Directory to search for run-manifest JSON files.
        glob:
            Filename glob pattern.  Defaults to ``"*_run_manifest.json"``.
        split:
            Which split to expose.
        **kwargs:
            Forwarded to :class:`RobustSepDataset.__init__`.
        """
        manifest_dir = Path(manifest_dir)
        paths = sorted(manifest_dir.glob(glob))
        if not paths:
            raise FileNotFoundError(f"No manifests matching {glob!r} in {manifest_dir}")
        return cls(paths, split=split, **kwargs)
