#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/external/raw"

mkdir -p "$RAW_DIR/doclaynet" "$RAW_DIR/sku110k" "$RAW_DIR/rgb_color"

download() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "skip existing: $out"
    return 0
  fi
  echo "download: $url"
  curl -C - -fL --retry 8 --retry-delay 10 --connect-timeout 30 -o "$out" "$url"
}

download_doclaynet_v12() {
  local base="https://huggingface.co/datasets/docling-project/DocLayNet-v1.2/resolve/main"
  download "$base/README.md" "$RAW_DIR/doclaynet/README.md"

  local i name
  for i in $(seq -f "%05g" 0 71); do
    name="train-${i}-of-00072.parquet"
    download "$base/data/$name" "$RAW_DIR/doclaynet/$name"
  done
  for i in $(seq -f "%05g" 0 6); do
    name="validation-${i}-of-00007.parquet"
    download "$base/data/$name" "$RAW_DIR/doclaynet/$name"
  done
  for i in $(seq -f "%05g" 0 5); do
    name="test-${i}-of-00006.parquet"
    download "$base/data/$name" "$RAW_DIR/doclaynet/$name"
  done
}

download_sku110k() {
  download "https://raw.githubusercontent.com/eg4000/SKU110K_CVPR19/master/README.md" \
    "$RAW_DIR/sku110k/README.md"
  download "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz" \
    "$RAW_DIR/sku110k/SKU110K_fixed.tar.gz"
}

write_rgb_color_note() {
  cat > "$RAW_DIR/rgb_color/README_DOWNLOAD.md" <<'EOF'
# RGB Color Coverage

The training plan references the Kaggle dataset:

https://www.kaggle.com/datasets/shuvokumarbasak4004/rgb-color-dataset-16777216-colors

Direct unauthenticated download is not available from this environment. The local preparation
pipeline therefore generates deterministic RGB color coverage supplements:

- uniform RGB grid tiles
- near-neutral ramps
- dark ramps
- saturated/neon ramps
- smooth 2D gradients

If Kaggle credentials are available later, place the downloaded archive in this directory and
record it in `data/external/manifests/download_manifest.jsonl`.
EOF
}

case "${1:-all}" in
  doclaynet) download_doclaynet_v12 ;;
  sku110k) download_sku110k ;;
  rgb-note) write_rgb_color_note ;;
  all)
    write_rgb_color_note
    download_doclaynet_v12
    download_sku110k
    ;;
  *)
    echo "usage: $0 [all|doclaynet|sku110k|rgb-note]" >&2
    exit 2
    ;;
esac
