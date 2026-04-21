#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_robustsep_dataset import (  # noqa: E402
    PATCH,
    classify_patch,
    project_ppp,
    rgb_to_cmyk_baseline,
    sha256_file,
    srgb_to_linear,
    linear_to_lab_d50,
    write_shard,
)


def sample_page(
    image_bytes: bytes,
    source_id: str,
    metadata: dict,
    rng: random.Random,
    stride: int,
    max_per_page: int,
    bucket_counts: Counter,
    bucket_cap: int,
) -> list[dict]:
    with Image.open(io.BytesIO(image_bytes)) as opened:
        im = opened.convert("RGBA")
    w, h = im.size
    if w < PATCH or h < PATCH:
        return []
    coords = [(x, y) for y in range(0, h - PATCH + 1, stride) for x in range(0, w - PATCH + 1, stride)]
    rng.shuffle(coords)
    out = []
    for x, y in coords:
        patch = np.asarray(im.crop((x, y, x + PATCH, y + PATCH)), dtype=np.uint8)
        structure, color, stats = classify_patch(patch)
        bucket = f"{structure}/{color}"
        if bucket_counts[bucket] >= bucket_cap:
            continue
        rgb = patch[..., :3].astype(np.float32) / 255.0
        lab = linear_to_lab_d50(srgb_to_linear(rgb))
        cmyk = rgb_to_cmyk_baseline(rgb)
        cmyk_projected = project_ppp(cmyk)
        bucket_counts[bucket] += 1
        out.append(
            {
                "source_path": source_id,
                "x": x,
                "y": y,
                "crop_meta": {"cropped": False, "bbox": None},
                "structure": structure,
                "color": color,
                "stats": stats,
                "doclaynet_metadata": metadata,
                "rgb": rgb.astype(np.float16),
                "lab": lab.astype(np.float16),
                "cmyk": cmyk.astype(np.float16),
                "cmyk_projected": cmyk_projected.astype(np.float16),
            }
        )
        if len(out) >= max_per_page:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="data/external/raw/doclaynet")
    ap.add_argument("--out-dir", default="data/external/processed/doclaynet_patches")
    ap.add_argument("--manifest-dir", default="data/external/manifests")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--max-per-page", type=int, default=8)
    ap.add_argument("--bucket-cap", type=int, default=25000)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260424)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for split in args.splits:
        files.extend(sorted(parquet_dir.glob(f"{split}-*.parquet")))

    bucket_counts: Counter = Counter()
    shard_records = []
    shard_manifests = []
    errors = []
    pages_seen = 0
    total_patches = 0
    shard_id = 0

    for parquet_path in files:
        pf = pq.ParquetFile(parquet_path)
        for row_group in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(row_group, columns=["image", "metadata"])
            data = table.to_pydict()
            for image_obj, metadata in zip(data["image"], data["metadata"]):
                pages_seen += 1
                source_id = image_obj.get("path") or f"{parquet_path.name}:row{pages_seen}"
                try:
                    records = sample_page(
                        image_obj["bytes"],
                        source_id,
                        metadata,
                        rng,
                        args.stride,
                        args.max_per_page,
                        bucket_counts,
                        args.bucket_cap,
                    )
                except Exception as exc:
                    errors.append({"source": source_id, "error": repr(exc)})
                    continue
                shard_records.extend(records)
                while len(shard_records) >= args.shard_size:
                    chunk = shard_records[: args.shard_size]
                    shard_records = shard_records[args.shard_size :]
                    shard_manifests.append(write_shard(chunk, out_dir, shard_id))
                    total_patches += shard_manifests[-1]["count"]
                    shard_id += 1
                if args.max_pages and pages_seen >= args.max_pages:
                    break
            if args.max_pages and pages_seen >= args.max_pages:
                break
        if args.max_pages and pages_seen >= args.max_pages:
            break

    if shard_records:
        shard_manifests.append(write_shard(shard_records, out_dir, shard_id))
        total_patches += shard_manifests[-1]["count"]

    manifest = {
        "created_unix": time.time(),
        "seed": args.seed,
        "parquet_dir": str(parquet_dir),
        "out_dir": str(out_dir),
        "splits": args.splits,
        "parquet_files": [str(p) for p in files],
        "pages_seen": pages_seen,
        "patches_written": total_patches,
        "patch_size": PATCH,
        "stride": args.stride,
        "max_per_page": args.max_per_page,
        "bucket_cap": args.bucket_cap,
        "bucket_counts": dict(bucket_counts),
        "errors": errors[:1000],
        "error_count": len(errors),
        "shards": shard_manifests,
        "target_generation": {
            "lab": "sRGB inverse transfer -> XYZ D65 -> Bradford D50 -> CIE Lab",
            "cmyk_baseline": "deterministic GCR-style RGB to CMYK baseline",
            "ppp_projection": {"channel_caps": [1.0, 1.0, 1.0, 1.0], "tac_max": 3.0},
        },
    }
    manifest_path = manifest_dir / f"{out_dir.name}_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"patches_written": total_patches, "pages_seen": pages_seen, "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
