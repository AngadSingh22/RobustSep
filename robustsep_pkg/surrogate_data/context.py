from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustsep_pkg.core.channels import ensure_cmykogv_last_axis


@dataclass(frozen=True)
class ContextWindow:
    cmykogv_context: np.ndarray
    lab_center: np.ndarray
    center_slice: tuple[slice, slice]


def pad_patch_to_context(patch: np.ndarray, context_size: int = 32) -> np.ndarray:
    """Pad a square patch to a square context by edge replication."""
    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC patch, got shape {arr.shape}")
    h, w, _ = arr.shape
    if h > context_size or w > context_size:
        raise ValueError(f"patch shape {arr.shape} larger than context_size={context_size}")
    pad_y = context_size - h
    pad_x = context_size - w
    before_y = pad_y // 2
    after_y = pad_y - before_y
    before_x = pad_x // 2
    after_x = pad_x - before_x
    return np.pad(arr, ((before_y, after_y), (before_x, after_x), (0, 0)), mode="edge").astype(np.float32)


def extract_center_context(
    cmykogv_image: np.ndarray,
    lab_image: np.ndarray,
    *,
    center_x: int,
    center_y: int,
    patch_size: int = 16,
    context_size: int = 32,
) -> ContextWindow:
    """Extract a 32x32 CMYKOGV context and scored 16x16 Lab center."""
    ensure_cmykogv_last_axis(cmykogv_image.shape)
    if lab_image.shape[-1] != 3:
        raise ValueError(f"expected Lab last axis 3, got shape {lab_image.shape}")
    half_ctx = context_size // 2
    half_patch = patch_size // 2
    padded_cmykogv = np.pad(
        cmykogv_image.astype(np.float32),
        ((half_ctx, half_ctx), (half_ctx, half_ctx), (0, 0)),
        mode="edge",
    )
    padded_lab = np.pad(
        lab_image.astype(np.float32),
        ((half_ctx, half_ctx), (half_ctx, half_ctx), (0, 0)),
        mode="edge",
    )
    cx = center_x + half_ctx
    cy = center_y + half_ctx
    context = padded_cmykogv[cy - half_ctx : cy + half_ctx, cx - half_ctx : cx + half_ctx]
    lab_center = padded_lab[cy - half_patch : cy + half_patch, cx - half_patch : cx + half_patch]
    center_start = (context_size - patch_size) // 2
    center_slice = (slice(center_start, center_start + patch_size), slice(center_start, center_start + patch_size))
    return ContextWindow(cmykogv_context=context.astype(np.float32), lab_center=lab_center.astype(np.float32), center_slice=center_slice)
