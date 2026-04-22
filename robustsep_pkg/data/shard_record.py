from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShardEntry:
    """One shard row in a run-manifest ``shards`` list.

    Corresponds directly to the dict written by ``write_shard`` in the
    preparation scripts (keys: ``npz``, ``jsonl``, ``count``,
    ``npz_sha256``, ``jsonl_sha256``).
    """

    npz: str
    jsonl: str
    count: int
    npz_sha256: str
    jsonl_sha256: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShardEntry":
        return cls(
            npz=d["npz"],
            jsonl=d["jsonl"],
            count=int(d["count"]),
            npz_sha256=d["npz_sha256"],
            jsonl_sha256=d["jsonl_sha256"],
        )


@dataclass
class ShardRecord:
    """Metadata for a single patch record as stored in a ``.jsonl`` file.

    All fields mirror the keys written by the preparation scripts.  The
    ``shard_index`` locates the corresponding row in the companion ``.npz``
    array batch.
    """

    shard_index: int
    source_path: str
    x: int
    y: int
    structure: str          # "flat" | "edge" | "textured"
    color: str              # "dark" | "neutral" | "saturated" | "normal"
    stats: dict[str, float] = field(default_factory=dict)
    crop_meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShardRecord":
        known = {"shard_index", "source_path", "x", "y", "structure", "color", "stats", "crop_meta"}
        return cls(
            shard_index=int(d["shard_index"]),
            source_path=str(d["source_path"]),
            x=int(d["x"]),
            y=int(d["y"]),
            structure=str(d["structure"]),
            color=str(d["color"]),
            stats=dict(d.get("stats", {})),
            crop_meta=dict(d.get("crop_meta", {})),
            extra={k: v for k, v in d.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shard_index": self.shard_index,
            "source_path": self.source_path,
            "x": self.x,
            "y": self.y,
            "structure": self.structure,
            "color": self.color,
            "stats": self.stats,
            "crop_meta": self.crop_meta,
        }
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class ShardSample:
    rgb: np.ndarray
    alpha: np.ndarray
    lab: np.ndarray
    icc_cmyk: np.ndarray
    cmyk_baseline: np.ndarray
    cmykogv_baseline: np.ndarray
    record: ShardRecord

    def tensors(self) -> dict[str, np.ndarray]:
        return {
            "rgb": self.rgb,
            "alpha": self.alpha,
            "lab": self.lab,
            "icc_cmyk": self.icc_cmyk,
            "cmyk_baseline": self.cmyk_baseline,
            "cmykogv_baseline": self.cmykogv_baseline,
        }
