"""robustsep_pkg.data.intent_adapter — intent/structure computation helpers.

Thin wrapper over the preprocess primitives that presents a stable API for
the enrichment layer.  All heavy computation is delegated to the existing
package functions; this module handles shape negotiation and the
intent-raster downsampling contract.

Intent raster (low-resolution intent map)
------------------------------------------
For the surrogate model and the proposer conditioning, we need a compact
low-resolution representation of the per-pixel intent maps.  The spec
Algorithm 1 refers to ``intent_raster_u`` as an input to the proposer:

    G_phi: (x_u, alpha_u, tau_u, w_u, intent_raster_u, PPP, lambda_k, z_k)

We produce this as a ``(raster_size, raster_size, 3)`` array where the three
channels are [w_brand, w_gradient, w_flat] — the alpha-weighted mean of each
raw intent map over a ``(raster_size x raster_size)`` grid of sub-regions.

For a 16x16 patch with raster_size=4, each sub-region covers 4x4 pixels.
"""
from __future__ import annotations

import numpy as np

from robustsep_pkg.core.config import PreprocessConfig
from robustsep_pkg.preprocess.intent import aggregate_patch_intents, compute_intent_maps
from robustsep_pkg.preprocess.structure import structure_token_for_patch


# ---------------------------------------------------------------------------
# Intent weights (aggregate patch scalars)
# ---------------------------------------------------------------------------


def compute_intent_weights(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    x: int = 0,
    y: int = 0,
    config: PreprocessConfig = PreprocessConfig(),
) -> dict[str, float]:
    """Compute per-patch intent weights from a patch's RGB and alpha.

    Parameters
    ----------
    rgb:
        Full image or patch RGB, shape ``(H, W, 3)`` float32, sRGB [0,1].
        When passing a cropped 16x16 patch set ``x=0, y=0``.
    alpha:
        Full image or patch alpha, shape ``(H, W)`` float32 [0,1].
    x, y:
        Top-left corner of the patch within *rgb* (0-based pixel coords).
        Use ``x=0, y=0`` when *rgb* is already the patch itself.
    config:
        Preprocessing configuration.

    Returns
    -------
    Dict with keys ``"brand"``, ``"gradient"``, ``"flat"`` — normalised
    per-patch scalar weights summing to approximately 1.
    """
    maps, _ = compute_intent_maps(rgb, config=config)
    return aggregate_patch_intents(
        maps,
        alpha=alpha,
        x=x,
        y=y,
        patch_size=config.patch_size,
        alpha_gamma=config.alpha_gamma,
    )


def compute_structure_token(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    x: int = 0,
    y: int = 0,
    config: PreprocessConfig = PreprocessConfig(),
) -> str:
    """Compute the structure token for one patch.

    Parameters
    ----------
    rgb:
        Full image or patch, shape ``(H, W, 3)`` float32, sRGB [0,1].
    alpha:
        Full image or patch alpha, shape ``(H, W)`` float32.
    x, y:
        Top-left corner within *rgb*.  ``0, 0`` when *rgb* is the patch.
    config:
        Preprocessing configuration.

    Returns
    -------
    One of ``"edge"``, ``"flat"``, ``"textured"``.
    """
    _, feature_maps = compute_intent_maps(rgb, config=config)
    return structure_token_for_patch(feature_maps, alpha, x=x, y=y,
                                     patch_size=config.patch_size, config=config)


# ---------------------------------------------------------------------------
# Low-resolution intent raster
# ---------------------------------------------------------------------------


def compute_low_res_intent_raster(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    x: int = 0,
    y: int = 0,
    patch_size: int = 16,
    raster_size: int = 4,
    config: PreprocessConfig = PreprocessConfig(),
) -> np.ndarray:
    """Compute a ``(raster_size, raster_size, 3)`` intent raster for one patch.

    The three channels encode ``[brand, gradient, flat]`` intent densities on
    a coarse grid.  Each cell ``(ri, rj)`` covers a ``(cell_h x cell_w)``
    sub-region and stores the alpha-weighted mean intent value over that region.

    This is the ``intent_raster_u`` input required by the proposer conditioning
    stack (spec §4, Algorithm 1).

    Parameters
    ----------
    rgb:
        Full image or patch RGB ``(H, W, 3)`` float32.
    alpha:
        Alpha ``(H, W)`` float32.
    x, y:
        Top-left corner of the patch within *rgb*.
    patch_size:
        Patch edge length in pixels (16 by default).
    raster_size:
        Grid size for the low-resolution raster (4 → 4×4 cells for 16×16).
    config:
        Preprocessing configuration.

    Returns
    -------
    ``np.ndarray`` shape ``(raster_size, raster_size, 3)`` float32.
    Channels: [brand_density, gradient_density, flat_density].
    """
    intent_maps, _ = compute_intent_maps(rgb, config=config)
    brand_map = intent_maps["brand"]   # (H, W) full image maps
    grad_map   = intent_maps["gradient"]
    flat_map   = intent_maps["flat"]

    alpha_patch = np.clip(alpha[y:y + patch_size, x:x + patch_size], 0.0, 1.0)
    alpha_w = np.power(alpha_patch, config.alpha_gamma)

    # Extract patch-level maps
    sl = np.s_[y:y + patch_size, x:x + patch_size]
    b = brand_map[sl] * alpha_w
    g = grad_map[sl] * alpha_w
    f = flat_map[sl] * alpha_w
    w = alpha_w  # denominator

    cell_h = patch_size // raster_size
    cell_w = patch_size // raster_size

    raster = np.zeros((raster_size, raster_size, 3), dtype=np.float32)
    for ri in range(raster_size):
        for rj in range(raster_size):
            r_sl = np.s_[ri * cell_h:(ri + 1) * cell_h, rj * cell_w:(rj + 1) * cell_w]
            denom = float(w[r_sl].sum()) + config.eps
            raster[ri, rj, 0] = float(b[r_sl].sum()) / denom
            raster[ri, rj, 1] = float(g[r_sl].sum()) / denom
            raster[ri, rj, 2] = float(f[r_sl].sum()) / denom

    return raster
