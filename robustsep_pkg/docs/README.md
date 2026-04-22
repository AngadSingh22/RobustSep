# RobustSep-v1.1

RobustSep is a reproducible research/codebase for PPP-conditioned RGB-to-CMYK separation under realistic print-process constraints. The repo is organized to support publication: synthetic dataset generation with strict verification, trainable components (surrogate + proposer), evaluation/report generation, and a paper + supplementary bundle, all tied together with run manifests and hashes so every result can be traced back to an exact config and code commit.

## Code Organization
The repository has been restructured to be self-contained within `robustsep_pkg`:
*   `robustsep_pkg/core/`: artifact IO, channel conventions, config, deterministic seeding.
*   `robustsep_pkg/data/`: staged shard readers, split manifests, training adapter, enrichment.
*   `robustsep_pkg/preprocess/`: RGB/Lab conversion, alpha-aware patches, intent maps, structure tokens.
*   `robustsep_pkg/targets/`: CMYKOGV initialization, two-stage target generation, target manifests.
*   `robustsep_pkg/surrogate_data/`: forward-surrogate example and shard writer.
*   `robustsep_pkg/models/`: surrogate CNN, conditional VAE proposer, conditioning, refiner utilities.
*   `scripts/`: repository-level dataset and ICC preparation scripts.
*   `robustsep_pkg/docs/`: Documentation.

## Installation
```bash
pip install -e .
```

## Running Scripts
Run scripts from the repository root with `PYTHONPATH` set to the checkout:

```bash
PYTHONPATH=. python scripts/prepare_robustsep_dataset.py --help
PYTHONPATH=. python scripts/prepare_doclaynet_patches.py --help
PYTHONPATH=. python scripts/apply_icc_cmyk_to_shards.py --help
```

## Current Smoke Path
```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

There is no checked-in `robustsep_pkg/scripts/train_vae.py` entrypoint yet. Model code,
surrogate data generation, and target-generation modules are available as package APIs and
should be wired into explicit CLIs after the surrogate quality probe is implemented.

## Reproducibility
*   **Source of truth**: `robustsep_pkg` + `pyproject.toml`.
*   **Artifacts**: Large files (checkpoints, datasets) go to `artifacts/` (gitignored).

## Architecture
See `robustsep_pkg/docs/final_arch.md` and `robustsep_pkg/docs/vae_fs_pargb2cmyk.md`
for the controlling architecture and mathematical specification.
