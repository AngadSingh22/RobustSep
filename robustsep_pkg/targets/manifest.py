from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from robustsep_pkg.core.artifact_io import canonical_json_hash
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.targets.config import TargetSolverConfig


@dataclass(frozen=True)
class TargetManifestRecord:
    target_hash: str
    initial_hash: str
    ppp_hash: str
    teacher_mode: str
    solver_config: dict[str, Any]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    ogv_initial_policy: str = "zero"
    projection_policy: str = "pi_k"
    source_id: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        initial: np.ndarray,
        target: np.ndarray,
        ppp: PPP,
        config: TargetSolverConfig,
        source_id: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> "TargetManifestRecord":
        initial_hash = canonical_json_hash(_array_digest_payload(initial))
        target_hash = canonical_json_hash(_array_digest_payload(target))
        return cls(
            target_hash=target_hash,
            initial_hash=initial_hash,
            ppp_hash=ppp.hash,
            teacher_mode=config.teacher_mode.value,
            solver_config=config.to_dict(),
            input_shape=tuple(initial.shape),
            output_shape=tuple(target.shape),
            source_id=source_id,
            diagnostics=diagnostics or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array_digest_payload(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(array)
    # Hash the raw bytes through canonical_json_hash's stable JSON payload.
    import hashlib

    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
