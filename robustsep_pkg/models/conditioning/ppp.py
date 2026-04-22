from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np

from robustsep_pkg.core.artifact_io import canonical_json_hash
from robustsep_pkg.core.channels import CHANNELS_CMYKOGV, CHANNEL_INDEX, ensure_cmykogv_last_axis


BASE_FAMILIES: dict[str, dict[str, Any]] = {
    "film_generic_conservative": {
        "caps": {"C": 1.0, "M": 1.0, "Y": 1.0, "K": 1.0, "O": 0.55, "G": 0.55, "V": 0.55},
        "tac_max": 3.0,
        "ogv_max": 0.85,
        "pair_caps": {"CO": 1.25, "MG": 1.25, "YV": 1.25},
        "neutral_ogv_max": 0.05,
        "neutral_chroma_threshold": 5.0,
        "dark_l_threshold": 18.0,
        "risk_threshold": 3.0,
        "risk_threshold_hard": False,
    },
    "film_gravure_generic": {
        "caps": {"C": 1.0, "M": 1.0, "Y": 1.0, "K": 1.0, "O": 0.65, "G": 0.65, "V": 0.65},
        "tac_max": 3.2,
        "ogv_max": 1.0,
        "pair_caps": {"CO": 1.35, "MG": 1.35, "YV": 1.35},
        "neutral_ogv_max": 0.05,
        "neutral_chroma_threshold": 5.0,
        "dark_l_threshold": 16.0,
        "risk_threshold": 3.0,
        "risk_threshold_hard": False,
    },
    "film_flexo_generic": {
        "caps": {"C": 0.95, "M": 0.95, "Y": 0.95, "K": 0.95, "O": 0.50, "G": 0.50, "V": 0.50},
        "tac_max": 2.8,
        "ogv_max": 0.75,
        "pair_caps": {"CO": 1.15, "MG": 1.15, "YV": 1.15},
        "neutral_ogv_max": 0.04,
        "neutral_chroma_threshold": 5.0,
        "dark_l_threshold": 20.0,
        "risk_threshold": 3.2,
        "risk_threshold_hard": False,
    },
    "paperboard_generic": {
        "caps": {"C": 0.95, "M": 0.95, "Y": 0.95, "K": 1.0, "O": 0.45, "G": 0.45, "V": 0.45},
        "tac_max": 2.7,
        "ogv_max": 0.65,
        "pair_caps": {"CO": 1.10, "MG": 1.10, "YV": 1.10},
        "neutral_ogv_max": 0.035,
        "neutral_chroma_threshold": 6.0,
        "dark_l_threshold": 22.0,
        "risk_threshold": 3.5,
        "risk_threshold_hard": False,
    },
    "label_stock_generic": {
        "caps": {"C": 1.0, "M": 1.0, "Y": 1.0, "K": 1.0, "O": 0.60, "G": 0.60, "V": 0.60},
        "tac_max": 3.0,
        "ogv_max": 0.90,
        "pair_caps": {"CO": 1.25, "MG": 1.25, "YV": 1.25},
        "neutral_ogv_max": 0.05,
        "neutral_chroma_threshold": 5.0,
        "dark_l_threshold": 18.0,
        "risk_threshold": 3.0,
        "risk_threshold_hard": False,
    },
}


@dataclass(frozen=True)
class PPP:
    base_family: str = "film_generic_conservative"
    caps: dict[str, float] = field(default_factory=dict)
    tac_max: float = 3.0
    ogv_max: float = 0.85
    pair_caps: dict[str, float | None] = field(default_factory=dict)
    neutral_ogv_max: float = 0.05
    neutral_chroma_threshold: float = 5.0
    dark_l_threshold: float = 18.0
    risk_threshold: float = 3.0
    risk_threshold_hard: bool = False
    override_mask: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_base(cls, base_family: str = "film_generic_conservative", overrides: dict[str, Any] | None = None) -> "PPP":
        if base_family not in BASE_FAMILIES:
            raise ValueError(f"unknown PPP base family: {base_family}")
        data = dict(BASE_FAMILIES[base_family])
        data["caps"] = dict(data["caps"])
        data["pair_caps"] = dict(data["pair_caps"])
        overrides = overrides or {}
        override_mask: dict[str, bool] = {}
        for key, value in overrides.items():
            if key == "caps":
                data["caps"].update(value)
                for cap_key in value:
                    override_mask[f"caps.{cap_key}"] = True
            elif key == "pair_caps":
                data["pair_caps"].update(value)
                for pair_key in value:
                    override_mask[f"pair_caps.{pair_key}"] = True
            else:
                data[key] = value
                override_mask[key] = True
        ppp = cls(base_family=base_family, override_mask=override_mask, **data)
        ppp.validate()
        return ppp

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PPP":
        base_family = payload.get("base_family", "film_generic_conservative")
        overrides = {k: v for k, v in payload.items() if k not in {"base_family", "override_mask"}}
        return cls.from_base(base_family, overrides)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    _VALID_PAIR_KEYS: ClassVar[frozenset[str]] = frozenset({"CO", "MG", "YV"})

    def validate(self) -> None:
        missing = [c for c in CHANNELS_CMYKOGV if c not in self.caps]
        if missing:
            raise ValueError(f"PPP missing channel caps for: {missing}")
        if self.tac_max <= 0:
            raise ValueError("PPP tac_max must be positive")
        if self.ogv_max < 0 or self.neutral_ogv_max < 0:
            raise ValueError("PPP OGV caps must be nonnegative")
        for channel, cap in self.caps.items():
            if channel not in CHANNEL_INDEX:
                raise ValueError(f"unknown channel in caps: {channel}")
            if cap < 0:
                raise ValueError(f"channel cap must be nonnegative: {channel}")
        invalid_pairs = set(self.pair_caps.keys()) - self._VALID_PAIR_KEYS
        if invalid_pairs:
            raise ValueError(f"PPP pair_caps contains unknown keys: {sorted(invalid_pairs)}; valid keys are {sorted(self._VALID_PAIR_KEYS)}")

    @property
    def hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    @property
    def cap_vector(self) -> np.ndarray:
        return np.asarray([self.caps.get(c, 1.0) for c in CHANNELS_CMYKOGV], dtype=np.float32)


def neutral_or_dark_mask(lab_ref: np.ndarray, ppp: PPP) -> np.ndarray:
    chroma = np.sqrt(lab_ref[..., 1] ** 2 + lab_ref[..., 2] ** 2)
    return (chroma <= ppp.neutral_chroma_threshold) | (lab_ref[..., 0] <= ppp.dark_l_threshold)


def _enforce_pair_cap(z: np.ndarray, i: int, j: int, cap: float | None, eps: float) -> None:
    if cap is None:
        return
    s = z[..., i] + z[..., j]
    scale = np.minimum(1.0, cap / np.maximum(s, eps))
    z[..., i] *= scale
    z[..., j] *= scale


def capped_simplex_lower(z: np.ndarray, target_sum: np.ndarray, iterations: int = 40) -> np.ndarray:
    """Project each last-axis vector downward to sum <= target_sum.

    Solves min ||x-z||^2 subject to 0 <= x <= z and sum(x) = target_sum for rows whose
    current sum exceeds target_sum. Rows already under target are returned unchanged.
    """
    out = z.copy()
    sums = out.sum(axis=-1, keepdims=True)
    active = sums > target_sum
    if not np.any(active):
        return out
    lo = np.zeros_like(target_sum, dtype=np.float32)
    hi = np.max(out, axis=-1, keepdims=True)
    for _ in range(iterations):
        mid = (lo + hi) * 0.5
        reduced_sum = np.maximum(out - mid, 0.0).sum(axis=-1, keepdims=True)
        lo = np.where(reduced_sum > target_sum, mid, lo)
        hi = np.where(reduced_sum > target_sum, hi, mid)
    projected = np.maximum(out - hi, 0.0)
    return np.where(active, projected, out)


def project_to_feasible(
    values: np.ndarray,
    ppp: PPP,
    lab_ref: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    ensure_cmykogv_last_axis(values.shape)
    z = np.clip(values.astype(np.float32, copy=True), 0.0, ppp.cap_vector)
    _enforce_pair_cap(z, CHANNEL_INDEX["C"], CHANNEL_INDEX["O"], ppp.pair_caps.get("CO"), eps)
    _enforce_pair_cap(z, CHANNEL_INDEX["M"], CHANNEL_INDEX["G"], ppp.pair_caps.get("MG"), eps)
    _enforce_pair_cap(z, CHANNEL_INDEX["Y"], CHANNEL_INDEX["V"], ppp.pair_caps.get("YV"), eps)

    ogv = z[..., 4:7].sum(axis=-1, keepdims=True)
    scale = np.minimum(1.0, ppp.ogv_max / np.maximum(ogv, eps))
    z[..., 4:7] *= scale

    if lab_ref is not None:
        mask = neutral_or_dark_mask(lab_ref, ppp)[..., None]
        ogv = z[..., 4:7].sum(axis=-1, keepdims=True)
        neutral_scale = np.minimum(1.0, ppp.neutral_ogv_max / np.maximum(ogv, eps))
        z[..., 4:7] = np.where(mask, z[..., 4:7] * neutral_scale, z[..., 4:7])

    target = np.full(z.shape[:-1] + (1,), ppp.tac_max, dtype=np.float32)
    z = capped_simplex_lower(z, target)
    return z.astype(np.float32)


def feasibility_violations(values: np.ndarray, ppp: PPP, lab_ref: np.ndarray | None = None, tol: float = 1e-6) -> dict[str, int]:
    ensure_cmykogv_last_axis(values.shape)
    z = values.astype(np.float32, copy=False)
    counts = {
        "below_zero": int(np.count_nonzero(z < -tol)),
        "channel_cap": int(np.count_nonzero(z > ppp.cap_vector + tol)),
        "tac": int(np.count_nonzero(z.sum(axis=-1) > ppp.tac_max + tol)),
        "ogv": int(np.count_nonzero(z[..., 4:7].sum(axis=-1) > ppp.ogv_max + tol)),
        "pair_co": int(np.count_nonzero(z[..., 0] + z[..., 4] > ppp.pair_caps.get("CO", np.inf) + tol)),
        "pair_mg": int(np.count_nonzero(z[..., 1] + z[..., 5] > ppp.pair_caps.get("MG", np.inf) + tol)),
        "pair_yv": int(np.count_nonzero(z[..., 2] + z[..., 6] > ppp.pair_caps.get("YV", np.inf) + tol)),
    }
    if lab_ref is not None:
        mask = neutral_or_dark_mask(lab_ref, ppp)
        counts["neutral_ogv"] = int(np.count_nonzero((z[..., 4:7].sum(axis=-1) > ppp.neutral_ogv_max + tol) & mask))
    return counts
