from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from robustsep_pkg.core.artifact_io import read_jsonl
from robustsep_pkg.data.shard_record import ShardEntry, ShardRecord
from robustsep_pkg.preprocess.color import cmyk_to_cmykogv


class ShardArrays:
    """Lazily loaded dense arrays for one shard.

    Attributes
    ----------
    rgb : np.ndarray
        Shape ``(N, 16, 16, 3)``, dtype ``float32``.  sRGB in ``[0, 1]``.
    alpha : np.ndarray
        Shape ``(N, 16, 16)``, dtype ``float32``.  Uniform ``1.0`` for
        non-RGBA sources; alpha-aware shards will expose the real value
        once the staging pipeline stores it.  Currently shards only store
        RGB/CMYK, so alpha is synthesised as all-ones here.
    lab : np.ndarray
        Shape ``(N, 16, 16, 3)``, dtype ``float32``.  CIE L*a*b* D50.
    icc_cmyk : np.ndarray
        Shape ``(N, 16, 16, 4)``, dtype ``float32``.  Raw CMYK baseline
        (deterministic GCR or ICC profile, depending on shard family).
    cmyk_baseline : np.ndarray
        Shape ``(N, 16, 16, 4)``, dtype ``float32``.  PPP-projected CMYK.
    cmykogv_baseline : np.ndarray
        Shape ``(N, 16, 16, 7)``, dtype ``float32``.  OGV appended as zeros
        to ``cmyk_baseline`` — the canonical CMYKOGV starting point before
        the constrained solver is run.
    """

    __slots__ = ("rgb", "alpha", "lab", "icc_cmyk", "cmyk_baseline", "cmykogv_baseline")

    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(str(npz_path))
        self.rgb: np.ndarray = data["rgb"].astype(np.float32)
        self.lab: np.ndarray = data["lab"].astype(np.float32)
        # ``cmyk`` is the raw baseline (GCR or ICC); ``cmyk_projected`` is
        # the PPP-feasibility-projected view of it.
        self.icc_cmyk: np.ndarray = data["cmyk"].astype(np.float32)
        self.cmyk_baseline: np.ndarray = data["cmyk_projected"].astype(np.float32)
        # Synthesise alpha: shards do not store an explicit alpha channel yet;
        # staged images were all RGBA but only RGB+targets were persisted.
        self.alpha: np.ndarray = np.ones(self.rgb.shape[:3], dtype=np.float32)
        # CMYKOGV baseline: cmyk_projected with OGV = 0.
        self.cmykogv_baseline: np.ndarray = cmyk_to_cmykogv(self.cmyk_baseline)


class ShardReader:
    """Reader for a single staged shard pair (.npz + .jsonl).

    Parameters
    ----------
    entry:
        :class:`~robustsep_pkg.data.shard_record.ShardEntry` pointing to
        the ``.npz`` and ``.jsonl`` files.
    root:
        Optional filesystem root prepended to relative paths stored in
        manifest entries.  Leave as ``None`` if paths are already absolute
        or the working directory is correct.

    Examples
    --------
    >>> entry = ShardEntry(npz="patches-00000.npz", jsonl="patches-00000.jsonl",
    ...                    count=4096, npz_sha256="...", jsonl_sha256="...")
    >>> reader = ShardReader(entry)
    >>> arrays = reader.load_arrays()
    >>> for rec in reader.iter_records():
    ...     pass  # rec is a ShardRecord
    """

    def __init__(self, entry: ShardEntry, root: str | Path | None = None) -> None:
        self._entry = entry
        self._root = Path(root) if root is not None else None

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        if self._root is not None and not p.is_absolute():
            return self._root / p
        return p

    @property
    def entry(self) -> ShardEntry:
        return self._entry

    @property
    def count(self) -> int:
        return self._entry.count

    def load_arrays(self) -> ShardArrays:
        """Load and return all dense arrays for this shard."""
        return ShardArrays(self._resolve(self._entry.npz))

    def iter_records(self) -> Iterator[ShardRecord]:
        """Yield one :class:`~robustsep_pkg.data.shard_record.ShardRecord` per patch."""
        for raw in read_jsonl(self._resolve(self._entry.jsonl)):
            yield ShardRecord.from_dict(raw)

    def load_record(self, shard_index: int) -> ShardRecord:
        """Load the metadata record at ``shard_index`` by sequential scan.

        For random access to many indices prefer :meth:`iter_records` with
        a dict comprehension.
        """
        for rec in self.iter_records():
            if rec.shard_index == shard_index:
                return rec
        raise IndexError(f"shard_index {shard_index} not found in {self._entry.jsonl}")
