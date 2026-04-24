from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("ProposerTrainingDataset requires PyTorch. Install torch to use this module.") from exc

from robustsep_pkg.core.artifact_io import read_json, read_jsonl
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.proposer.model import ProposerModelConfig
from robustsep_pkg.models.surrogate.data import STRUCTURE_TO_INDEX, ppp_condition_arrays


class ProposerTrainingDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, model_config: ProposerModelConfig = ProposerModelConfig()) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = read_json(self.manifest_path)
        self.ppp = PPP.from_dict(self.manifest["ppp"])
        self.model_config = model_config
        self.schema_version = int(self.manifest.get("schema", {}).get("version", 1))
        self._ppp_numeric, self._ppp_mask, self._base_family_index = ppp_condition_arrays(self.ppp, model_config=model_config)
        self._shards = list(self.manifest["shards"])
        self._index: list[tuple[int, int]] = []
        for shard_idx, shard in enumerate(self._shards):
            for local_idx in range(int(shard["count"])):
                self._index.append((shard_idx, local_idx))
        self._cache_shard_idx: int | None = None
        self._cache_arrays: dict[str, np.ndarray] | None = None
        self._cache_records: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx, local_idx = self._index[index]
        arrays, records = self._load_shard(shard_idx)
        record = records[local_idx]
        return {
            "rgb_patch": torch.from_numpy(arrays["rgb_patch"][local_idx].astype(np.float32)),
            "alpha": torch.from_numpy(arrays["alpha"][local_idx].astype(np.float32)),
            "lab_ref_center": torch.from_numpy(arrays["lab_ref_center"][local_idx].astype(np.float32)),
            "target_cmykogv": torch.from_numpy(arrays["target_cmykogv"][local_idx].astype(np.float32)),
            "intent_raster": torch.from_numpy(arrays["intent_raster"][local_idx].astype(np.float32)),
            "intent_weights": torch.from_numpy(arrays["intent_weights"][local_idx].astype(np.float32)),
            "base_family_index": torch.tensor(self._base_family_index, dtype=torch.long),
            "ppp_numeric": torch.from_numpy(self._ppp_numeric),
            "ppp_override_mask": torch.from_numpy(self._ppp_mask),
            "structure_index": torch.tensor(STRUCTURE_TO_INDEX.get(record.get("structure_token", "flat"), 1), dtype=torch.long),
            "source_id": record.get("source_id", ""),
            "target_hash": record.get("target_hash", ""),
            "schema_version": self.schema_version,
        }

    def _load_shard(self, shard_idx: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if self._cache_shard_idx == shard_idx and self._cache_arrays is not None and self._cache_records is not None:
            return self._cache_arrays, self._cache_records
        shard = self._shards[shard_idx]
        with np.load(shard["npz"]) as data:
            arrays = {key: data[key].copy() for key in data.files}
        records = read_jsonl(shard["jsonl"])
        self._cache_shard_idx = shard_idx
        self._cache_arrays = arrays
        self._cache_records = records
        return arrays, records


def iter_proposer_shard_batches(
    manifest_path: str | Path,
    *,
    batch_size: int,
    model_config: ProposerModelConfig = ProposerModelConfig(),
    seed: int = 20260422,
    epoch: int = 0,
    shuffle_shards: bool = True,
    shuffle_within_shard: bool = True,
    drop_last: bool = False,
) -> Iterator[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    manifest = read_json(Path(manifest_path))
    ppp = PPP.from_dict(manifest["ppp"])
    schema_version = int(manifest.get("schema", {}).get("version", 1))
    ppp_numeric, ppp_mask, base_family_index = ppp_condition_arrays(ppp, model_config=model_config)
    shards = list(manifest["shards"])
    rng = np.random.default_rng(int(seed) + 1000003 * int(epoch))
    shard_order = np.arange(len(shards), dtype=np.int64)
    if shuffle_shards:
        rng.shuffle(shard_order)
    for shard_idx in shard_order:
        shard = shards[int(shard_idx)]
        count = int(shard["count"])
        with np.load(shard["npz"]) as data:
            arrays = {key: data[key].copy() for key in data.files}
        records = read_jsonl(shard["jsonl"])
        order = np.arange(count, dtype=np.int64)
        if shuffle_within_shard:
            rng.shuffle(order)
        for start in range(0, count, batch_size):
            take = order[start : start + batch_size]
            if drop_last and take.size < batch_size:
                continue
            yield _batch_from_shard_arrays(
                arrays,
                records,
                take,
                ppp_numeric=ppp_numeric,
                ppp_mask=ppp_mask,
                base_family_index=base_family_index,
                schema_version=schema_version,
            )


def _batch_from_shard_arrays(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    ppp_numeric: np.ndarray,
    ppp_mask: np.ndarray,
    base_family_index: int,
    schema_version: int,
) -> dict[str, Any]:
    size = int(indices.size)
    structure_index = np.asarray(
        [STRUCTURE_TO_INDEX.get(records[int(i)].get("structure_token", "flat"), 1) for i in indices],
        dtype=np.int64,
    )
    return {
        "rgb_patch": torch.from_numpy(arrays["rgb_patch"][indices].astype(np.float32)),
        "alpha": torch.from_numpy(arrays["alpha"][indices].astype(np.float32)),
        "lab_ref_center": torch.from_numpy(arrays["lab_ref_center"][indices].astype(np.float32)),
        "target_cmykogv": torch.from_numpy(arrays["target_cmykogv"][indices].astype(np.float32)),
        "intent_raster": torch.from_numpy(arrays["intent_raster"][indices].astype(np.float32)),
        "intent_weights": torch.from_numpy(arrays["intent_weights"][indices].astype(np.float32)),
        "base_family_index": torch.full((size,), int(base_family_index), dtype=torch.long),
        "ppp_numeric": torch.from_numpy(np.broadcast_to(ppp_numeric, (size, ppp_numeric.size)).copy()),
        "ppp_override_mask": torch.from_numpy(np.broadcast_to(ppp_mask, (size, ppp_mask.size)).copy()),
        "structure_index": torch.from_numpy(structure_index),
        "source_id": [records[int(i)].get("source_id", "") for i in indices],
        "target_hash": [records[int(i)].get("target_hash", "") for i in indices],
        "schema_version": schema_version,
    }
