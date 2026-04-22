from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TeacherMode(str, Enum):
    ICC_ONLY = "icc_only"
    SURROGATE_ONLY = "surrogate_only"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class TargetSolverConfig:
    teacher_mode: TeacherMode = TeacherMode.ICC_ONLY
    stage1_steps: int = 0
    stage2_steps: int = 0
    stage1_step_size: float = 0.0
    stage2_step_size: float = 0.0
    alpha_app: float = 1.0
    alpha_anchor: float = 0.25
    alpha_ogv: float = 0.05
    alpha_neutral: float = 0.10
    beta_risk: float = 1.0
    beta_trust: float = 0.50
    beta_ogv: float = 0.10
    alpha_icc: float = 1.0
    version: str = "target-solver-skeleton-v1"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["teacher_mode"] = self.teacher_mode.value
        return data
