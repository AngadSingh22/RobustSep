"""robustsep_pkg.data.torch_dataset — optional PyTorch Dataset wrapper.

This module is imported only when ``torch`` is available.  If ``torch`` is
not installed, importing this module raises :class:`ImportError` with a clear
message; the rest of the data pipeline works without PyTorch.

Usage
-----
::

    from robustsep_pkg.data.torch_dataset import RobustSepTorchDataset
    dataset = RobustSepTorchDataset(adapter, tensor_keys=["rgb", "cmykogv_baseline"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

Design
------
- ``__len__`` returns the total number of patches across all families in the adapter.
- ``__getitem__`` delegates to ``adapter``'s underlying datasets via a flat
  index into the concatenated patch sequence.  This is O(n_families) not O(n_patches).
- Returned items are dicts of ``torch.Tensor`` (float32) for the requested keys,
  plus ``"record_struct"`` string (JSON-serialised :class:`ShardRecord`) for
  debugging.
"""
from __future__ import annotations

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _TorchDataset = object  # type: ignore[misc, assignment]

from typing import Any

import numpy as np

from robustsep_pkg.data.training_adapter import FamilyDataset, TrainingAdapter

# Valid tensor keys that can be requested from a ShardSample
_VALID_KEYS = frozenset({
    "rgb", "alpha", "lab", "icc_cmyk", "cmyk_baseline", "cmykogv_baseline",
})


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed.  Install it with:\n"
            "  pip install torch\n"
            "or use the CPU-only data pipeline (TrainingAdapter.iter_epoch) instead."
        )


class RobustSepTorchDataset(_TorchDataset):  # type: ignore[misc]
    """PyTorch :class:`~torch.utils.data.Dataset` backed by a :class:`TrainingAdapter`.

    Parameters
    ----------
    adapter:
        A pre-built :class:`~robustsep_pkg.data.training_adapter.TrainingAdapter`
        wrapping one or more :class:`~robustsep_pkg.data.dataset.RobustSepDataset`
        objects.
    tensor_keys:
        Which array fields to include in the returned dict.  Must be a subset
        of ``{"rgb", "alpha", "lab", "icc_cmyk", "cmyk_baseline", "cmykogv_baseline"}``.
        Defaults to all keys.
    include_record:
        If ``True``, the dict also contains ``"source_path"``, ``"x"``,
        ``"y"``, ``"structure"``, ``"color"`` as plain Python strings/ints
        (not tensors) for diagnostics.
    """

    def __init__(
        self,
        adapter: TrainingAdapter,
        *,
        tensor_keys: list[str] | None = None,
        include_record: bool = False,
    ) -> None:
        _require_torch()
        self._adapter = adapter
        self._families = adapter._families  # list[FamilyDataset]
        self._tensor_keys = list(tensor_keys or sorted(_VALID_KEYS))
        self._include_record = include_record

        bad = set(self._tensor_keys) - _VALID_KEYS
        if bad:
            raise ValueError(f"Unknown tensor keys: {sorted(bad)}")

        # Build flat cumulative patch count per family for O(families) __getitem__
        self._cum_counts: list[int] = []
        total = 0
        for fd in self._families:
            total += len(fd.dataset)
            self._cum_counts.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> dict[str, Any]:
        _require_torch()
        if index < 0:
            index += self._total
        if index < 0 or index >= self._total:
            raise IndexError(index)

        # Locate which family and local index
        prev = 0
        for fd, cum in zip(self._families, self._cum_counts):
            if index < cum:
                local_idx = index - prev
                sample = fd.dataset[local_idx]
                break
            prev = cum
        else:
            raise IndexError(index)

        out: dict[str, Any] = {}
        all_tensors = sample.tensors()
        for key in self._tensor_keys:
            arr = all_tensors[key].astype(np.float32)
            out[key] = torch.from_numpy(arr)

        if self._include_record:
            rec = sample.record
            out["source_path"] = rec.source_path
            out["x"] = rec.x
            out["y"] = rec.y
            out["structure"] = rec.structure
            out["color"] = rec.color

        return out
