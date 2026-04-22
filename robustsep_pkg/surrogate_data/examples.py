from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robustsep_pkg.models.conditioning.drift import DriftSample, apply_drift
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data.context import pad_patch_to_context


@dataclass(frozen=True)
class SurrogateExample:
    cmykogv_context: np.ndarray
    lab_center: np.ndarray
    ppp: PPP
    structure_token: str
    intent_weights: dict[str, float]
    intent_raster: np.ndarray
    drift: DriftSample
    lambda_value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def drifted_context(self) -> np.ndarray:
        return apply_drift(self.cmykogv_context, self.drift)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ppp_hash": self.ppp.hash,
            "structure_token": self.structure_token,
            "intent_weights": dict(self.intent_weights),
            "intent_raster_shape": tuple(self.intent_raster.shape),
            "lambda_value": self.lambda_value,
            **self.metadata,
        }


def build_surrogate_example(
    center_cmykogv_patch: np.ndarray,
    lab_center_patch: np.ndarray,
    *,
    ppp: PPP,
    drift: DriftSample,
    structure_token: str,
    intent_weights: dict[str, float],
    intent_raster: np.ndarray | None = None,
    lambda_value: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> SurrogateExample:
    """Build one surrogate training/eval example from a center 16x16 patch."""
    context = pad_patch_to_context(center_cmykogv_patch, context_size=32)
    lab = np.asarray(lab_center_patch, dtype=np.float32)
    if lab.shape != (16, 16, 3):
        raise ValueError(f"expected lab_center_patch shape (16,16,3), got {lab.shape}")
    if intent_raster is None:
        raster = np.zeros((4, 4, 3), dtype=np.float32)
        raster[..., 0] = float(intent_weights.get("brand", 0.0))
        raster[..., 1] = float(intent_weights.get("gradient", 0.0))
        raster[..., 2] = float(intent_weights.get("flat", 0.0))
    else:
        raster = np.asarray(intent_raster, dtype=np.float32)
    return SurrogateExample(
        cmykogv_context=context,
        lab_center=lab,
        ppp=ppp,
        structure_token=structure_token,
        intent_weights=dict(intent_weights),
        intent_raster=raster,
        drift=drift,
        lambda_value=float(lambda_value),
        metadata=metadata or {},
    )
