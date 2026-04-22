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
        with np.load(str(npz_path)) as data:
            self.rgb = data["rgb"].astype(np.float32)
            self.lab = data["lab"].astype(np.float32)
            # ``cmyk`` is the raw baseline (GCR or ICC); ``cmyk_projected`` is
            # the PPP-feasibility-projected view of it.
            self.icc_cmyk = data["cmyk"].astype(np.float32)
            self.cmyk_baseline = data.get("cmyk_projected", data["cmyk"]).astype(np.float32)
            if "alpha" in data:
                alpha = data["alpha"].astype(np.float32)
                if alpha.ndim == 4 and alpha.shape[-1] == 1:
                    alpha = alpha[..., 0]
                self.alpha = np.clip(alpha, 0.0, 1.0)
            else:
                # Existing staged shards do not persist alpha. They were
                # alpha-filtered during sampling, so expose an all-visible
                # tensor until the staging format is upgraded.
                self.alpha = np.ones(self.rgb.shape[:3], dtype=np.float32)
            # CMYKOGV baseline: projected CMYK with OGV = 0.
            self.cmykogv_baseline = cmyk_to_cmykogv(self.cmyk_baseline)
        self._validate_shapes(npz_path)

    def _validate_shapes(self, npz_path: str | Path) -> None:
        n = self.rgb.shape[0]
        expected = {
            "rgb": (n, 16, 16, 3),
            "alpha": (n, 16, 16),
            "lab": (n, 16, 16, 3),
            "icc_cmyk": (n, 16, 16, 4),
            "cmyk_baseline": (n, 16, 16, 4),
            "cmykogv_baseline": (n, 16, 16, 7),
        }
        actual = {
            "rgb": self.rgb.shape,
            "alpha": self.alpha.shape,
            "lab": self.lab.shape,
            "icc_cmyk": self.icc_cmyk.shape,
            "cmyk_baseline": self.cmyk_baseline.shape,
            "cmykogv_baseline": self.cmykogv_baseline.shape,
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{npz_path}: expected {name} shape {shape}, got {actual[name]}")

    def sample(self, index: int) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb[index],
            "alpha": self.alpha[index],
            "lab": self.lab[index],
            "icc_cmyk": self.icc_cmyk[index],
            "cmyk_baseline": self.cmyk_baseline[index],
            "cmykogv_baseline": self.cmykogv_baseline[index],
        }


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

    def load_records(self) -> list[ShardRecord]:
        return list(self.iter_records())

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
