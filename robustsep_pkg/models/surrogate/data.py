from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("SurrogateTrainingDataset requires PyTorch. Install torch to use this module.") from exc

from robustsep_pkg.core.artifact_io import read_json, read_jsonl
from robustsep_pkg.core.channels import CHANNELS_CMYKOGV
from robustsep_pkg.models.conditioning.ppp import BASE_FAMILIES, PPP
from robustsep_pkg.models.surrogate.model import SurrogateModelConfig

BASE_FAMILY_TO_INDEX = {name: idx for idx, name in enumerate(BASE_FAMILIES)}
STRUCTURE_TO_INDEX = {"edge": 0, "flat": 1, "textured": 2}
PAIR_KEYS = ("CO", "MG", "YV")


class SurrogateTrainingDataset(Dataset):
    """Map-style PyTorch dataset over surrogate training shard manifests."""

    def __init__(self, manifest_path: str | Path, *, model_config: SurrogateModelConfig = SurrogateModelConfig()) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = read_json(self.manifest_path)
        self.ppp = PPP.from_dict(self.manifest["ppp"])
        self.model_config = model_config
        self._ppp_numeric, self._ppp_mask, self._base_family_index = ppp_condition_arrays(self.ppp, model_config)
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
        drift_vector = np.concatenate(
            [
                arrays["drift_multipliers"][local_idx].reshape(-1),
                arrays["drift_trc_y"][local_idx, :, 1:-1].reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "cmykogv_context": torch.from_numpy(arrays["cmykogv_context"][local_idx].astype(np.float32)),
            "lab_center": torch.from_numpy(arrays["lab_center"][local_idx].astype(np.float32)),
            "intent_raster": torch.from_numpy(arrays["intent_raster"][local_idx].astype(np.float32)),
            "intent_weights": torch.from_numpy(arrays["intent_weights"][local_idx].astype(np.float32)),
            "lambda_value": torch.tensor(float(arrays["lambda_value"][local_idx]), dtype=torch.float32),
            "drift_multipliers": torch.from_numpy(arrays["drift_multipliers"][local_idx].astype(np.float32)),
            "drift_trc_x": torch.from_numpy(arrays["drift_trc_x"][local_idx].astype(np.float32)),
            "drift_trc_y": torch.from_numpy(arrays["drift_trc_y"][local_idx].astype(np.float32)),
            "drift_vector": torch.from_numpy(drift_vector),
            "base_family_index": torch.tensor(self._base_family_index, dtype=torch.long),
            "ppp_numeric": torch.from_numpy(self._ppp_numeric),
            "ppp_override_mask": torch.from_numpy(self._ppp_mask),
            "structure_index": torch.tensor(STRUCTURE_TO_INDEX.get(record.get("structure_token", "flat"), 1), dtype=torch.long),
            "source_id": record.get("source_id", ""),
            "target_hash": record.get("target_hash", ""),
            "drift_hash": record.get("drift_hash", ""),
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


def iter_surrogate_shard_batches(
    manifest_path: str | Path,
    *,
    batch_size: int,
    model_config: SurrogateModelConfig = SurrogateModelConfig(),
    seed: int = 20260422,
    epoch: int = 0,
    shuffle_shards: bool = True,
    shuffle_within_shard: bool = True,
    drop_last: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield training batches while loading each surrogate shard at most once.

    The map-style dataset is intentionally kept for validation/probing, where
    access is sequential. Training with global random shuffling is much more
    expensive because every random sample can force a different compressed
    ``.npz`` shard to be loaded. This iterator shuffles shard order and then
    shuffles examples inside each loaded shard, preserving stochastic training
    without repeated whole-shard reloads.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    manifest = read_json(Path(manifest_path))
    ppp = PPP.from_dict(manifest["ppp"])
    ppp_numeric, ppp_mask, base_family_index = ppp_condition_arrays(ppp, model_config)
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
        if len(records) != count:
            raise ValueError(f"{shard['jsonl']} has {len(records)} records, expected {count}")

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
            )


def _batch_from_shard_arrays(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    ppp_numeric: np.ndarray,
    ppp_mask: np.ndarray,
    base_family_index: int,
) -> dict[str, Any]:
    size = int(indices.size)
    drift_vector = np.concatenate(
        [
            arrays["drift_multipliers"][indices].reshape(size, -1),
            arrays["drift_trc_y"][indices, :, 1:-1].reshape(size, -1),
        ],
        axis=1,
    ).astype(np.float32)
    structure_index = np.asarray(
        [STRUCTURE_TO_INDEX.get(records[int(i)].get("structure_token", "flat"), 1) for i in indices],
        dtype=np.int64,
    )
    return {
        "cmykogv_context": torch.from_numpy(arrays["cmykogv_context"][indices].astype(np.float32)),
        "lab_center": torch.from_numpy(arrays["lab_center"][indices].astype(np.float32)),
        "intent_raster": torch.from_numpy(arrays["intent_raster"][indices].astype(np.float32)),
        "intent_weights": torch.from_numpy(arrays["intent_weights"][indices].astype(np.float32)),
        "lambda_value": torch.from_numpy(arrays["lambda_value"][indices].astype(np.float32)),
        "drift_multipliers": torch.from_numpy(arrays["drift_multipliers"][indices].astype(np.float32)),
        "drift_trc_x": torch.from_numpy(arrays["drift_trc_x"][indices].astype(np.float32)),
        "drift_trc_y": torch.from_numpy(arrays["drift_trc_y"][indices].astype(np.float32)),
        "drift_vector": torch.from_numpy(drift_vector),
        "base_family_index": torch.full((size,), int(base_family_index), dtype=torch.long),
        "ppp_numeric": torch.from_numpy(np.broadcast_to(ppp_numeric, (size, ppp_numeric.size)).copy()),
        "ppp_override_mask": torch.from_numpy(np.broadcast_to(ppp_mask, (size, ppp_mask.size)).copy()),
        "structure_index": torch.from_numpy(structure_index),
        "source_id": [records[int(i)].get("source_id", "") for i in indices],
        "target_hash": [records[int(i)].get("target_hash", "") for i in indices],
        "drift_hash": [records[int(i)].get("drift_hash", "") for i in indices],
    }


def ppp_condition_arrays(ppp: PPP, model_config: SurrogateModelConfig = SurrogateModelConfig()) -> tuple[np.ndarray, np.ndarray, int]:
    raw = np.asarray(
        [float(ppp.caps[channel]) for channel in CHANNELS_CMYKOGV]
        + [
            float(ppp.tac_max),
            float(ppp.ogv_max),
            float(ppp.neutral_ogv_max),
            float(ppp.neutral_chroma_threshold),
            float(ppp.dark_l_threshold),
            float(ppp.risk_threshold),
            float(ppp.risk_threshold_hard),
        ]
        + [float(ppp.pair_caps.get(pair, 0.0) or 0.0) for pair in PAIR_KEYS],
        dtype=np.float32,
    )
    if raw.size != model_config.ppp_numeric_dim:
        raise ValueError(f"PPP numeric vector has {raw.size} fields, model expects {model_config.ppp_numeric_dim}")
    mask_keys = [f"caps.{channel}" for channel in CHANNELS_CMYKOGV] + [
        "tac_max",
        "ogv_max",
        "neutral_ogv_max",
        "neutral_chroma_threshold",
        "dark_l_threshold",
        "risk_threshold",
        "risk_threshold_hard",
    ] + [f"pair_caps.{pair}" for pair in PAIR_KEYS]
    mask = np.asarray([1.0 if ppp.override_mask.get(key, False) else 0.0 for key in mask_keys], dtype=np.float32)
    base_index = BASE_FAMILY_TO_INDEX.get(ppp.base_family, 0)
    return raw, mask, base_index
