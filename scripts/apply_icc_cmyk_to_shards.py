#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from prepare_robustsep_dataset import project_ppp, sha256_file


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
    arr = np.asarray(cmyk, dtype=np.float32) / 255.0
    return arr


def convert_dir(input_dir: Path, output_dir: Path, src_profile: Path, dst_profile: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform(src_profile, dst_profile)
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
            cmyk_projected[i] = project_ppp(icc).astype(np.float16)
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

    result = convert_dir(Path(args.input_dir), Path(args.output_dir), Path(args.src_profile), Path(args.dst_profile))
    base = json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    base["created_unix"] = time.time()
    base["out_dir"] = args.output_dir
    base["shards"] = result["shards"]
    base["patches_written"] = result["patches_written"]
    base["target_generation"]["cmyk_baseline"] = "ICC RGB->CMYK Relative Colorimetric with Black Point Compensation"
    base["target_generation"]["icc_profiles"] = {
        "source_rgb": args.src_profile,
        "destination_cmyk": args.dst_profile,
    }
    manifest_path = Path(args.manifest_dir) / f"{Path(args.output_dir).name}_run_manifest.json"
    manifest_path.write_text(json.dumps(base, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"patches_written": result["patches_written"], "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
