# RobustSep-v1.1

RobustSep is a reproducible research/codebase for PPP-conditioned RGB-to-CMYK separation under realistic print-process constraints. The repo is organized to support publication: synthetic dataset generation with strict verification, trainable components (surrogate + proposer), evaluation/report generation, and a paper + supplementary bundle, all tied together with run manifests and hashes so every result can be traced back to an exact config and code commit.

## Code Organization
The repository has been restructured to be self-contained within `robustsep_pkg`:
*   `robustsep_pkg/core/`: artifact IO, channel conventions, config, deterministic seeding.
*   `robustsep_pkg/data/`: staged shard readers, split manifests, training adapter, enrichment.
*   `robustsep_pkg/preprocess/`: RGB/Lab conversion, alpha-aware patches, intent maps, structure tokens.
*   `robustsep_pkg/targets/`: CMYKOGV initialization, two-stage target generation, target manifests.
*   `robustsep_pkg/surrogate_data/`: forward-surrogate v2 local-delta examples and shard writer.
*   `robustsep_pkg/models/`: surrogate CNN, conditional VAE proposer, conditioning, refiner utilities.
*   `scripts/`: repository-level dataset and ICC preparation scripts.
*   `robustsep_pkg/docs/`: Documentation.

## Installation
```bash
pip install -e .
```

## Running Scripts
The package exposes a consolidated CLI:

```bash
PYTHONPATH=. python -m robustsep_pkg.cli --help
PYTHONPATH=. python -m robustsep_pkg.cli export-split-manifest --help
PYTHONPATH=. python -m robustsep_pkg.cli generate-targets --help
PYTHONPATH=. python -m robustsep_pkg.cli write-surrogate-shards --help
PYTHONPATH=. python -m robustsep_pkg.cli train-surrogate --help
PYTHONPATH=. python -m robustsep_pkg.cli eval-surrogate-gate --help
```

Repository-level data preparation scripts are still available directly:

```bash
PYTHONPATH=. python scripts/prepare_robustsep_dataset.py --help
PYTHONPATH=. python scripts/prepare_doclaynet_patches.py --help
PYTHONPATH=. python scripts/apply_icc_cmyk_to_shards.py --help
```

## Current Smoke Path
```bash
PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli export-split-manifest \
  --family robustsep=data/external/manifests/robustsep_patches_icc_run_manifest.json \
  --family doclaynet=data/external/manifests/doclaynet_patches_icc_run_manifest.json \
  --family sku110k=data/external/manifests/sku110k_patches_icc_run_manifest.json \
  --weight robustsep=1.0 --weight doclaynet=0.5 --weight sku110k=0.5 \
  --root . --split train --alpha-policy ones \
  --out artifacts/cli_smoke/train_split_manifest_v11.json

PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli write-surrogate-shards \
  --split-manifest artifacts/cli_smoke/train_split_manifest_v11.json \
  --root . --out-dir artifacts/cli_smoke/surrogate_shards \
  --max-records 32 --drift-samples-per-patch 2 --stage1-steps 2 --stage2-steps 1

PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli train-surrogate \
  --manifest artifacts/cli_smoke/surrogate_shards/surrogate_training_manifest.json \
  --out-dir artifacts/cli_smoke/surrogate_train_gpu --device cuda \
  --probe-drift-samples 2 --probe-max-patches 1

PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

New surrogate shards are `surrogate-shards-v2`: they store `lab_ref_center`,
`teacher_lab_nominal`, and `teacher_lab_drifted`, and train the Micro-CNN to predict local
`DeltaLab`. `surrogate-shards-v1` artifacts and checkpoints remain inspectable for debugging
but must not be used to unlock VAE training or hybrid target generation.

The VAE proposer training CLI is still a future step; the current CLI covers target
generation, surrogate shard generation, surrogate training, and surrogate gate evaluation.

## Reproducibility
*   **Source of truth**: `robustsep_pkg` + `pyproject.toml`.
*   **Artifacts**: Large files (checkpoints, datasets) go to `artifacts/` (gitignored).

## Architecture
See `robustsep_pkg/docs/final_arch.md` and `robustsep_pkg/docs/vae_fs_pargb2cmyk.md`
for the controlling architecture and mathematical specification.
