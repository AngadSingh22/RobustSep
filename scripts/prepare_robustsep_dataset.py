#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


PATCH = 16
D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
D50 = np.array([0.96422, 1.0, 0.82521], dtype=np.float32)
SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)
BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float32,
)
BRADFORD_INV = np.linalg.inv(BRADFORD).astype(np.float32)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_lab_d50(linear_rgb: np.ndarray) -> np.ndarray:
    xyz_d65 = np.tensordot(linear_rgb, SRGB_TO_XYZ_D65.T, axes=1)
    src = BRADFORD @ D65
    dst = BRADFORD @ D50
    adapt = BRADFORD_INV @ np.diag(dst / src) @ BRADFORD
    xyz_d50 = np.tensordot(xyz_d65, adapt.T, axes=1)
    scaled = xyz_d50 / D50
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(scaled > eps, np.cbrt(np.maximum(scaled, 0)), (kappa * scaled + 16) / 116)
    l = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1).astype(np.float32)


def rgb_to_cmyk_baseline(rgb: np.ndarray) -> np.ndarray:
    cmy = 1.0 - rgb
    k = np.min(cmy, axis=-1, keepdims=True)
    denom = np.maximum(1.0 - k, 1e-6)
    cmy_out = np.where(k >= 0.999, 0.0, (cmy - k) / denom)
    return np.concatenate([cmy_out, k], axis=-1).astype(np.float32)


def project_ppp(cmyk: np.ndarray, tac_max: float = 3.0) -> np.ndarray:
    y = np.clip(cmyk, 0.0, 1.0).astype(np.float32)
    tac = np.sum(y, axis=-1, keepdims=True)
    scale = np.minimum(1.0, tac_max / np.maximum(tac, 1e-6))
    return y * scale


def edge_score(rgba: np.ndarray) -> float:
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * alpha
    dx = np.abs(lum[:, 1:] - lum[:, :-1]).mean()
    dy = np.abs(lum[1:, :] - lum[:-1, :]).mean()
    da = np.abs(alpha[:, 1:] - alpha[:, :-1]).mean() + np.abs(alpha[1:, :] - alpha[:-1, :]).mean()
    return float(dx + dy + 0.5 * da)


def classify_patch(rgba: np.ndarray) -> Tuple[str, str, Dict[str, float]]:
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    mask = alpha >= 0.0625
    visible = float(mask.mean())
    if mask.any():
        vals = rgb[mask]
    else:
        vals = rgb.reshape(-1, 3)
    mx = vals.max(axis=1)
    mn = vals.min(axis=1)
    sat = np.zeros_like(mx)
    np.divide(mx - mn, mx, out=sat, where=mx > 1e-6)
    brightness = vals.mean(axis=1)
    mean_sat = float(sat.mean())
    mean_brightness = float(brightness.mean())
    e = edge_score(rgba)
    rgb_std = float(vals.std())
    if e < 0.015 and rgb_std < 0.035:
        structure = "flat"
    elif e > 0.07 or (0.10 <= visible <= 0.92 and e > 0.025):
        structure = "edge"
    else:
        structure = "textured"
    if mean_brightness < 0.18:
        color = "dark"
    elif mean_sat < 0.08:
        color = "neutral"
    elif mean_sat > 0.55:
        color = "saturated"
    else:
        color = "normal"
    stats = {
        "visible_alpha": visible,
        "mean_saturation": mean_sat,
        "mean_brightness": mean_brightness,
        "edge_score": e,
        "rgb_std": rgb_std,
    }
    return structure, color, stats


def alpha_bbox(im: Image.Image, threshold: int) -> Tuple[int, int, int, int] | None:
    a = im.getchannel("A").point(lambda v: 255 if v >= threshold else 0)
    return a.getbbox()


def iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.exists():
            for pattern in ("*.png", "*.jpg", "*.jpeg"):
                yield from sorted(root.glob(pattern))


def quarantine_corrupt(paths: Iterable[Path], quarantine_dir: Path) -> List[dict]:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for p in iter_images(paths):
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError) as exc:
            target = quarantine_dir / p.name
            if target.exists():
                target = quarantine_dir / f"{p.stem}.{int(time.time())}{p.suffix}"
            shutil.move(str(p), str(target))
            records.append(
                {
                    "source": str(p),
                    "quarantine_path": str(target),
                    "reason": repr(exc),
                    "sha256": sha256_file(target),
                }
            )
    return records


def maybe_crop(im: Image.Image, bbox: Tuple[int, int, int, int] | None, min_gain: float) -> Tuple[Image.Image, dict]:
    if not bbox:
        return im, {"cropped": False, "bbox": None}
    w, h = im.size
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    area_ratio = (bw * bh) / max(1, w * h)
    if area_ratio <= min_gain:
        return im.crop(bbox), {"cropped": True, "bbox": list(bbox), "bbox_area_ratio": area_ratio}
    return im, {"cropped": False, "bbox": list(bbox), "bbox_area_ratio": area_ratio}


def sample_from_image(
    path: Path,
    rng: random.Random,
    min_visible: float,
    stride: int,
    max_per_image: int,
    bucket_counts: Counter,
    bucket_cap: int,
) -> List[dict]:
    with Image.open(path) as opened:
        im = opened.convert("RGBA")
    bbox = alpha_bbox(im, 16)
    im, crop_meta = maybe_crop(im, bbox, 0.82)
    w, h = im.size
    if w < PATCH or h < PATCH:
        return []
    candidates = []
    coords = [(x, y) for y in range(0, h - PATCH + 1, stride) for x in range(0, w - PATCH + 1, stride)]
    rng.shuffle(coords)
    for x, y in coords:
        patch = np.asarray(im.crop((x, y, x + PATCH, y + PATCH)), dtype=np.uint8)
        visible = float((patch[..., 3] >= 16).mean())
        if visible < min_visible:
            continue
        structure, color, stats = classify_patch(patch)
        bucket = f"{structure}/{color}"
        if bucket_counts[bucket] >= bucket_cap:
            continue
        patch_rgb = patch[..., :3].astype(np.float32) / 255.0
        linear = srgb_to_linear(patch_rgb)
        lab = linear_to_lab_d50(linear)
        cmyk = rgb_to_cmyk_baseline(patch_rgb)
        cmyk_projected = project_ppp(cmyk)
        bucket_counts[bucket] += 1
        candidates.append(
            {
                "source_path": str(path),
                "x": x,
                "y": y,
                "crop_meta": crop_meta,
                "structure": structure,
                "color": color,
                "stats": stats,
                "rgb": patch_rgb.astype(np.float16),
                "lab": lab.astype(np.float16),
                "cmyk": cmyk.astype(np.float16),
                "cmyk_projected": cmyk_projected.astype(np.float16),
            }
        )
        if len(candidates) >= max_per_image:
            break
    return candidates


def write_shard(records: List[dict], out_dir: Path, shard_id: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "rgb": np.stack([r.pop("rgb") for r in records]),
        "lab": np.stack([r.pop("lab") for r in records]),
        "cmyk": np.stack([r.pop("cmyk") for r in records]),
        "cmyk_projected": np.stack([r.pop("cmyk_projected") for r in records]),
    }
    npz_path = out_dir / f"patches-{shard_id:05d}.npz"
    meta_path = out_dir / f"patches-{shard_id:05d}.jsonl"
    np.savez_compressed(npz_path, **arrays)
    with meta_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            r["shard_index"] = i
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return {
        "npz": str(npz_path),
        "jsonl": str(meta_path),
        "count": int(arrays["rgb"].shape[0]),
        "npz_sha256": sha256_file(npz_path),
        "jsonl_sha256": sha256_file(meta_path),
    }


def generate_color_supplements(out_dir: Path, tile_size: int = 64) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    colors = []
    for r in np.linspace(0, 255, 17, dtype=np.uint8):
        for g in np.linspace(0, 255, 17, dtype=np.uint8):
            for b in np.linspace(0, 255, 17, dtype=np.uint8):
                colors.append((int(r), int(g), int(b)))
    for i, color in enumerate(colors):
        arr = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        arr[..., :3] = color
        arr[..., 3] = 255
        p = out_dir / f"rgb_grid_{i:04d}_{color[0]:02x}{color[1]:02x}{color[2]:02x}.png"
        Image.fromarray(arr, "RGBA").save(p)
        outputs.append(p)
    for name, start, end in [
        ("neutral_ramp", (0, 0, 0), (255, 255, 255)),
        ("dark_blue_ramp", (0, 0, 0), (0, 20, 80)),
        ("neon_rg_ramp", (0, 255, 60), (255, 0, 220)),
        ("warm_cool_ramp", (255, 70, 0), (0, 180, 255)),
    ]:
        arr = np.zeros((256, 256, 4), dtype=np.uint8)
        for x in range(256):
            t = x / 255
            c = [round((1 - t) * start[j] + t * end[j]) for j in range(3)]
            arr[:, x, :3] = c
        arr[..., 3] = 255
        p = out_dir / f"{name}.png"
        Image.fromarray(arr, "RGBA").save(p)
        outputs.append(p)
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument("--out-dir", default="data/external/processed/robustsep_patches")
    ap.add_argument("--manifest-dir", default="data/external/manifests")
    ap.add_argument("--quarantine-dir", default="data/quarantine/corrupt_png")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--max-per-image", type=int, default=24)
    ap.add_argument("--bucket-cap", type=int, default=25000)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--min-visible", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260422)
    ap.add_argument("--skip-quarantine", action="store_true")
    ap.add_argument("--no-supplements", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sources = args.source or ["data/png_full/png", "data/png_sample"]
    source_dirs = [Path(s) for s in sources]
    out_dir = Path(args.out_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    quarantined = []
    if not args.skip_quarantine:
        quarantined = quarantine_corrupt(source_dirs, Path(args.quarantine_dir))

    supplement_dir = Path("data/external/processed/color_supplements")
    supplement_paths = [] if args.no_supplements else generate_color_supplements(supplement_dir)
    all_sources = list(iter_images(source_dirs)) + supplement_paths
    all_sources = [p for p in all_sources if p.exists()]
    rng.shuffle(all_sources)
    if args.max_images > 0:
        all_sources = all_sources[: args.max_images]

    bucket_counts: Counter = Counter()
    shard_records: List[dict] = []
    shard_manifests = []
    image_errors = []
    shard_id = 0
    total = 0

    for p in all_sources:
        try:
            records = sample_from_image(
                p,
                rng,
                args.min_visible,
                args.stride,
                args.max_per_image,
                bucket_counts,
                args.bucket_cap,
            )
        except Exception as exc:
            image_errors.append({"path": str(p), "error": repr(exc)})
            continue
        shard_records.extend(records)
        while len(shard_records) >= args.shard_size:
            chunk = shard_records[: args.shard_size]
            shard_records = shard_records[args.shard_size :]
            shard_manifests.append(write_shard(chunk, out_dir, shard_id))
            total += shard_manifests[-1]["count"]
            shard_id += 1

    if shard_records:
        shard_manifests.append(write_shard(shard_records, out_dir, shard_id))
        total += shard_manifests[-1]["count"]

    run_manifest = {
        "created_unix": time.time(),
        "seed": args.seed,
        "sources": [str(p) for p in source_dirs],
        "supplement_dir": str(supplement_dir),
        "out_dir": str(out_dir),
        "patch_size": PATCH,
        "stride": args.stride,
        "min_visible": args.min_visible,
        "max_per_image": args.max_per_image,
        "bucket_cap": args.bucket_cap,
        "source_images_considered": len(all_sources),
        "patches_written": total,
        "bucket_counts": dict(bucket_counts),
        "quarantined": quarantined,
        "image_errors": image_errors[:1000],
        "image_error_count": len(image_errors),
        "shards": shard_manifests,
        "target_generation": {
            "lab": "sRGB inverse transfer -> XYZ D65 -> Bradford D50 -> CIE Lab",
            "cmyk_baseline": "deterministic GCR-style RGB to CMYK baseline",
            "ppp_projection": {"channel_caps": [1.0, 1.0, 1.0, 1.0], "tac_max": 3.0},
        },
    }
    manifest_name = f"{out_dir.name}_run_manifest.json"
    manifest_path = manifest_dir / manifest_name
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"patches_written": total, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
