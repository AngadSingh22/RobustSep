from __future__ import annotations

import numpy as np

from robustsep_pkg.targets.solver import render_cmykogv_lab_proxy


def calibrated_cmykogv_lab(
    values: np.ndarray,
    *,
    anchor_cmykogv: np.ndarray,
    anchor_lab: np.ndarray,
) -> np.ndarray:
    """Return a Lab teacher proxy calibrated to an ICC/RGB anchor.

    The raw seven-channel proxy is useful for local deltas but is not an ICC
    characterization.  This function anchors the proxy so that
    ``anchor_cmykogv`` renders exactly as ``anchor_lab`` and nearby ink changes
    are interpreted as deterministic Lab deltas around that staged reference.
    """
    y = np.asarray(values, dtype=np.float32)
    anchor_y = np.asarray(anchor_cmykogv, dtype=np.float32)
    lab = np.asarray(anchor_lab, dtype=np.float32)
    if y.shape != anchor_y.shape:
        raise ValueError(f"values and anchor_cmykogv shapes must match, got {y.shape} and {anchor_y.shape}")
    if lab.shape != y.shape[:-1] + (3,):
        raise ValueError(f"anchor_lab shape must be {y.shape[:-1] + (3,)}, got {lab.shape}")
    if np.array_equal(y, anchor_y):
        return lab.copy()
    return (lab + render_cmykogv_lab_proxy(y) - render_cmykogv_lab_proxy(anchor_y)).astype(np.float32)
