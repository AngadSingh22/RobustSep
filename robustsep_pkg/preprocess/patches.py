from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class AlphaPatch:
    rgb: np.ndarray
    alpha: np.ndarray
    x: int
    y: int

    @property
    def visible_fraction(self) -> float:
        return float(np.mean(self.alpha > 0.0))


def alpha_from_rgba(rgba: np.ndarray) -> np.ndarray:
    rgba = np.asarray(rgba)
    if rgba.shape[-1] == 4:
        return (rgba[..., 3].astype(np.float32) / 255.0).clip(0.0, 1.0)
    return np.ones(rgba.shape[:2], dtype=np.float32)


def rgb_from_image_array(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype.kind in {"u", "i"}:
        rgb = image[..., :3].astype(np.float32) / 255.0
    else:
        rgb = image[..., :3].astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def deterministic_patch_grid(height: int, width: int, patch_size: int = 16, stride: int = 8, include_edges: bool = True) -> list[tuple[int, int]]:
    if height < patch_size or width < patch_size:
        return []
    ys = list(range(0, height - patch_size + 1, stride))
    xs = list(range(0, width - patch_size + 1, stride))
    if include_edges:
        if ys[-1] != height - patch_size:
            ys.append(height - patch_size)
        if xs[-1] != width - patch_size:
            xs.append(width - patch_size)
    return [(x, y) for y in ys for x in xs]


def extract_alpha_patches(
    image: np.ndarray,
    patch_size: int = 16,
    stride: int = 8,
    include_edges: bool = True,
) -> Iterator[AlphaPatch]:
    rgb = rgb_from_image_array(image)
    alpha = alpha_from_rgba(image)
    for x, y in deterministic_patch_grid(rgb.shape[0], rgb.shape[1], patch_size, stride, include_edges=include_edges):
        yield AlphaPatch(
            rgb=rgb[y : y + patch_size, x : x + patch_size].copy(),
            alpha=alpha[y : y + patch_size, x : x + patch_size].copy(),
            x=x,
            y=y,
        )


def raised_cosine_window(size: int = 16, floor: float = 0.1) -> np.ndarray:
    i = np.arange(size, dtype=np.float32)
    h = floor + (1.0 - floor) * np.sin(np.pi * (i + 0.5) / size) ** 2
    return np.outer(h, h).astype(np.float32)
