from __future__ import annotations

from pathlib import Path
from typing import Any

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
