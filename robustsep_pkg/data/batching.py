"""robustsep_pkg.data.batching — contiguous mini-batch slices over a shard.

A *batch* here is a contiguous slice of one shard's dense arrays plus the
corresponding metadata records.  This is the basic unit fed into a training
step.  Shuffling across shards happens at the :class:`TrainingAdapter` level;
within a single shard patches are served in storage order (shuffled at the
shard level between epochs by the weighted schedule).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from robustsep_pkg.data.shard_reader import ShardArrays
from robustsep_pkg.data.shard_record import ShardRecord


# ---------------------------------------------------------------------------
# ShardBatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardBatch:
    """A slice of contiguous patches from one shard.

    All array attributes have shape ``(B, ...)`` where ``B`` is the actual
    batch size (may be smaller than requested for the final batch in a shard).

    Attributes
    ----------
    rgb : np.ndarray
        ``(B, 16, 16, 3)`` float32 — sRGB in [0, 1].
    alpha : np.ndarray
        ``(B, 16, 16)`` float32.
    lab : np.ndarray
        ``(B, 16, 16, 3)`` float32 — CIE L*a*b* D50.
    icc_cmyk : np.ndarray
        ``(B, 16, 16, 4)`` float32 — raw ICC/GCR CMYK baseline.
    cmyk_baseline : np.ndarray
        ``(B, 16, 16, 4)`` float32 — PPP-projected CMYK.
    cmykogv_baseline : np.ndarray
        ``(B, 16, 16, 7)`` float32 — OGV=0 expansion.
    records : list[ShardRecord]
        Length ``B`` — per-patch metadata.
    batch_index : int
        Zero-based index of this batch within its shard.
    """

    rgb: np.ndarray
    alpha: np.ndarray
    lab: np.ndarray
    icc_cmyk: np.ndarray
    cmyk_baseline: np.ndarray
    cmykogv_baseline: np.ndarray
    records: list[ShardRecord]
    batch_index: int

    def __len__(self) -> int:
        return self.rgb.shape[0]

    def tensors(self) -> dict[str, np.ndarray]:
        """Return all array fields as a dict (excludes records and batch_index)."""
        return {
            "rgb": self.rgb,
            "alpha": self.alpha,
            "lab": self.lab,
            "icc_cmyk": self.icc_cmyk,
            "cmyk_baseline": self.cmyk_baseline,
            "cmykogv_baseline": self.cmykogv_baseline,
        }


# ---------------------------------------------------------------------------
# iter_batches
# ---------------------------------------------------------------------------


def iter_batches(
    arrays: ShardArrays,
    records: list[ShardRecord],
    batch_size: int,
    *,
    drop_last: bool = False,
) -> Iterator[ShardBatch]:
    """Yield contiguous :class:`ShardBatch` slices from one shard.

    Parameters
    ----------
    arrays:
        Dense arrays for the full shard (loaded by :meth:`ShardReader.load_arrays`).
    records:
        Metadata records matching ``arrays`` row-for-row.
    batch_size:
        Number of patches per batch.  Must be >= 1.
    drop_last:
        If ``True``, discard the final incomplete batch (useful when the
        training loop requires uniform batch sizes).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    n = arrays.rgb.shape[0]
    if len(records) != n:
        raise ValueError(
            f"arrays have {n} rows but {len(records)} records — cannot batch"
        )

    batch_idx = 0
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        if drop_last and (end - start) < batch_size:
            break
        sl = np.s_[start:end]
        yield ShardBatch(
            rgb=arrays.rgb[sl],
            alpha=arrays.alpha[sl],
            lab=arrays.lab[sl],
            icc_cmyk=arrays.icc_cmyk[sl],
            cmyk_baseline=arrays.cmyk_baseline[sl],
            cmykogv_baseline=arrays.cmykogv_baseline[sl],
            records=records[start:end],
            batch_index=batch_idx,
        )
        start = end
        batch_idx += 1
