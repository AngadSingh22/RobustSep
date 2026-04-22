#!/usr/bin/env python3
"""Apply an ICC profile CMYK conversion to existing staged .npz shards.

Replaces the deterministic GCR baseline with a profile-derived CMYK and
re-projects to feasibility using :func:`robustsep_pkg.models.refiner.solver.pi_k`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

# ---------------------------------------------------------------------------
# All math from the package — no local duplicates.
# ---------------------------------------------------------------------------
from robustsep_pkg.core.artifact_io import read_json, sha256_file, write_json
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.models.refiner.solver import pi_k


def build_transform(src_profile: Path, dst_profile: Path):
    src = ImageCms.ImageCmsProfile(str(src_profile))
    dst = ImageCms.ImageCmsProfile(str(dst_profile))
    return ImageCms.buildTransform(
        src,
        dst,
        "RGB",
        "CMYK",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )


def rgb_patch_to_icc_cmyk(rgb_patch: np.ndarray, transform) -> np.ndarray:
    rgb8 = np.clip(np.rint(rgb_patch.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(rgb8, "RGB")
    cmyk = ImageCms.applyTransform(im, transform)
    return np.asarray(cmyk, dtype=np.float32) / 255.0


def convert_dir(input_dir: Path, output_dir: Path, src_profile: Path, dst_profile: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(src_profile, dst_profile)
    # Use a conservative film family PPP for the feasibility projection step.
    # The actual PPP used in model training will be passed at inference time;
    # this staging-time projection only enforces channel caps <= 1 and TAC <= 3.
    staging_ppp = PPP.from_base("film_generic_conservative")
    shard_manifests = []
    total = 0
    for npz_path in sorted(input_dir.glob("patches-*.npz")):
        data = np.load(npz_path)
        rgb = data["rgb"].astype(np.float32)
        lab = data["lab"]
        cmyk = np.empty((*rgb.shape[:3], 4), dtype=np.float16)
        cmyk_projected = np.empty_like(cmyk)
        for i in range(rgb.shape[0]):
            icc = rgb_patch_to_icc_cmyk(rgb[i], transform)
            cmyk[i] = icc.astype(np.float16)
            # pi_k requires last-axis = 7 (CMYKOGV); pad OGV with zeros for
            # staging projection, then extract only the CMYK channels.
            cmykogv = np.concatenate([icc, np.zeros((*icc.shape[:2], 3), dtype=np.float32)], axis=-1)
            projected_cmykogv = pi_k(cmykogv, staging_ppp)
            cmyk_projected[i] = projected_cmykogv[..., :4].astype(np.float16)
        out_npz = output_dir / npz_path.name
        np.savez_compressed(out_npz, rgb=rgb.astype(np.float16), lab=lab, cmyk=cmyk, cmyk_projected=cmyk_projected)
        in_meta = input_dir / npz_path.name.replace(".npz", ".jsonl")
        out_meta = output_dir / in_meta.name
        out_meta.write_bytes(in_meta.read_bytes())
        count = int(rgb.shape[0])
        total += count
        shard_manifests.append(
            {
                "npz": str(out_npz),
                "jsonl": str(out_meta),
                "count": count,
                "npz_sha256": sha256_file(out_npz),
                "jsonl_sha256": sha256_file(out_meta),
            }
        )
    return {"patches_written": total, "shards": shard_manifests}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--src-profile", default="/usr/share/color/icc/colord/sRGB.icc")
    ap.add_argument("--dst-profile", default="/usr/share/color/icc/colord/FOGRA39L_coated.icc")
    ap.add_argument("--manifest-dir", default="data/external/manifests")
    args = ap.parse_args()

    result = convert_dir(
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.src_profile),
        Path(args.dst_profile),
    )
    base = read_json(args.base_manifest)
    base["created_unix"] = time.time()
    base["out_dir"] = args.output_dir
    base["shards"] = result["shards"]
    base["patches_written"] = result["patches_written"]
    base["target_generation"]["cmyk_baseline"] = (
        "ICC RGB->CMYK Relative Colorimetric with Black Point Compensation"
    )
    base["target_generation"]["icc_profiles"] = {
        "source_rgb": args.src_profile,
        "destination_cmyk": args.dst_profile,
    }
    manifest_path = Path(args.manifest_dir) / f"{Path(args.output_dir).name}_run_manifest.json"
    write_json(manifest_path, base)
    print(json.dumps({"patches_written": result["patches_written"], "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
