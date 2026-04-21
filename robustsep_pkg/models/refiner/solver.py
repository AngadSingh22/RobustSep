from __future__ import annotations

import numpy as np

from robustsep_pkg.models.conditioning.ppp import PPP, feasibility_violations, project_to_feasible


def pi_k(values: np.ndarray, ppp: PPP, lab_ref: np.ndarray | None = None) -> np.ndarray:
    return project_to_feasible(values, ppp, lab_ref=lab_ref)


def is_feasible(values: np.ndarray, ppp: PPP, lab_ref: np.ndarray | None = None, tol: float = 1e-6) -> bool:
    return not any(feasibility_violations(values, ppp, lab_ref=lab_ref, tol=tol).values())
