from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.refiner.solver import pi_k

GradientFn = Callable[[np.ndarray, int, str], np.ndarray]


@dataclass(frozen=True)
class ProjectedGradientTrace:
    stage1_steps: int
    stage2_steps: int
    projected_each_step: bool
    max_abs_update: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | bool | dict[str, float]]:
        return {
            "stage1_steps": self.stage1_steps,
            "stage2_steps": self.stage2_steps,
            "projected_each_step": self.projected_each_step,
            "max_abs_update": self.max_abs_update,
            "diagnostics": self.diagnostics,
        }


def projected_gradient_solve(
    initial: np.ndarray,
    ppp: PPP,
    *,
    lab_ref: np.ndarray | None = None,
    stage1_steps: int = 0,
    stage2_steps: int = 0,
    stage1_step_size: float = 0.0,
    stage2_step_size: float = 0.0,
    gradient_fn: GradientFn | None = None,
) -> tuple[np.ndarray, ProjectedGradientTrace]:
    """Fixed-step projected-gradient skeleton for CMYKOGV targets.

    If `gradient_fn` is omitted, this is intentionally a no-op solver plus
    mandatory feasibility closure. That gives a deterministic target baseline
    until the ICC/surrogate teacher gradient is implemented.
    """
    y = pi_k(initial.astype(np.float32, copy=True), ppp, lab_ref=lab_ref)
    max_abs_update = 0.0

    for stage, steps, step_size in (
        ("stage1", stage1_steps, stage1_step_size),
        ("stage2", stage2_steps, stage2_step_size),
    ):
        for step in range(steps):
            if gradient_fn is None or step_size == 0.0:
                grad = np.zeros_like(y, dtype=np.float32)
            else:
                grad = gradient_fn(y.copy(), step, stage).astype(np.float32)
            update = step_size * grad
            max_abs_update = max(max_abs_update, float(np.max(np.abs(update))) if update.size else 0.0)
            y = pi_k(y - update, ppp, lab_ref=lab_ref)

    trace = ProjectedGradientTrace(
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        projected_each_step=True,
        max_abs_update=max_abs_update,
    )
    return y.astype(np.float32), trace
