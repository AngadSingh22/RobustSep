# RobustSep-v1.1

RobustSep is a reproducible research/codebase for PPP-conditioned RGB-to-CMYK separation under realistic print-process constraints. The repo is organized to support publication: synthetic dataset generation with strict verification, trainable components (surrogate + proposer), evaluation/report generation, and a paper + supplementary bundle, all tied together with run manifests and hashes so every result can be traced back to an exact config and code commit.

## Code Organization
The repository has been restructured to be self-contained within `robustsep_pkg`:
*   `robustsep_pkg/`: Configuration, Models, Training, Eval, Utils.
*   `robustsep_pkg/scripts/`: Executable scripts (e.g., `train_vae.py`).
*   `robustsep_pkg/notebooks/`: Jupyter notebooks.
*   `robustsep_pkg/docs/`: Documentation.

## Installation
```bash
pip install -e .
```

## Running Scripts
Scripts should be run as modules to ensure proper import resolution:

```bash
# Data Generation (Future)
python -m robustsep_pkg.scripts.data_gen

# VAE Training
python -m robustsep_pkg.scripts.train_vae
```

Alternatively, running directly from the root with `PYTHONPATH`:
```bash
set PYTHONPATH=.
python robustsep_pkg/scripts/train_vae.py
```

## CPU Smoke Path (End-to-End)
```bash
set PYTHONPATH=.
python robustsep_pkg/scripts/ingest_rgb_patches.py --input_dir path\to\pngs --output_dir artifacts\rgb_cache
python robustsep_pkg/scripts/build_vae_targets_stub.py --input_cache_dir artifacts\rgb_cache --out_dir artifacts\train_shards
python robustsep_pkg/scripts/train_vae.py --data_dir artifacts\train_shards --smoke_test
```

## Reproducibility
*   **Source of truth**: `robustsep_pkg` + `pyproject.toml`.
*   **Artifacts**: Large files (checkpoints, datasets) go to `artifacts/` (gitignored).

## Architecture
See `robustsep_pkg/docs/design_specs/repo_blueprint.md` for the detailed layout.
