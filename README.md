# RobustSep: A Non-Deterministic Architecture for Robust, Drift-Aware Color Separation in Packaging

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of **RobustSep**, as detailed in the paper:
*"RobustSep: A Non-Deterministic Architecture for Robust, Drift-Aware Color Separation in Packaging"*.

## Abstract

RGB-to-CMYK color separation for packaging presents a complex inverse problem, characterized by the absence of stable press characterization data and notable physical process drift. Current deterministic approaches, relying on ICC profiles, presuppose idealized conditions and lack formal assurances against perceptual degradation, whereas unconstrained learned models frequently fail to adhere to stringent print-production standards. This paper introduces **RobustSep**, a non-deterministic architecture for packaging color separation that amalgamates stochastic candidate generation, physics-informed refinement, and drift-aware evaluation into a cohesive inference pipeline. At the core of our methodology is the Press Prior Package (PPP), a structured entity encoding process-family constraints and drift distributions, independent of proprietary calibration data. We employ a conditional variational autoencoder (CVAE) for ink candidate generation, which undergoes refinement through a two-pass feasibility pipeline, and is evaluated by a lightweight forward surrogate network under simulated drift conditions. Candidate selection is directed by a 90th-percentile tail-risk criterion across the intent-weighted CIEDE2000 error distribution.

## Repository Structure

```text
RobustSepTraining/
├── data/                  # Raw and processed datasets (DocLayNet, SKU110k, RobustSep splits)
├── paper/                 # LaTeX source files for the manuscript
├── robustsep_pkg/         # Core Python package module
│   ├── data/              # Adapters, intent parsing, batching, and source-weighting
│   ├── docs/              # Architectural specs and engineering documentation
│   ├── engine/            # Multi-shot escalation and pipeline loop logic
│   ├── surrogate_data/    # Surrogate training context extraction scripts
│   └── targets/           # Dual-stage projected gradient solver for ground-truth generation
├── scripts/               # Utility scripts for data preparation and pipeline execution
└── tests/                 # Comprehensive unit test suite (139+ tests passing)
```

## System Architecture

The core of the RobustSep paradigm is the *Propose $\rightarrow$ Refine $\rightarrow$ Evaluate $\rightarrow$ Select* pipeline:

1. **Proposer (CVAE):** Generates diverse CMYKOGV ink candidates conditioned on the Press Prior Package (PPP) and target RGB patch.
2. **Refiner (Deterministic):** Unconditionally enforces printability by applying the projection operator $\Pi_K$ for spatial regularity, neutral/dark stability, and hard TAC constraints.
3. **Evaluator (Micro-CNN Surrogate):** Approximates physical press rendering under explicitly parameterized drift instances to evaluate tail-risk robustness.
4. **Selector (Ranker):** Uses a multi-shot escalation policy to select the feasible candidate that minimizes the $90^{\text{th}}$ percentile intent-weighted CIEDE2000 perceptual deviation.

## Installation

RobustSep requires Python 3.10 or higher. All dependencies are managed seamlessly.

```bash
# Clone the repository
git clone https://github.com/your-org/RobustSepTraining.git
cd RobustSepTraining

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -e .
```

## Usage Pipeline

### 1. Data Preparation and Shard Generation

Export manifests and generate surrogate target shards explicitly capturing A-Res source weighting (Vitter 1985).

```bash
# Example for generating train-split surrogate data shards
PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli write-surrogate-shards \
  --split-manifest artifacts/full_shards/train_split_manifest_v11.json \
  --root . \
  --out-dir artifacts/full_shards/surrogate_train \
  --summary-out artifacts/full_shards/surrogate_train_summary.json \
  --drift-samples-per-patch 2 \
  --shard-size 4096 \
  --stage1-steps 8 \
  --stage2-steps 4
```

### 2. Forward Surrogate Training

Train the Micro-CNN surrogate to reliably evaluate perceptual behavior under simulated process drift conditions.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli train-surrogate \
  --manifest artifacts/full_shards/surrogate_train/surrogate_training_manifest.json \
  --out-dir artifacts/surrogate_training/full_v1 \
  --device cuda \
  --batch-size 128 \
  --epochs 1 \
  --learning-rate 0.001
```

### 3. Quality Gate Evaluation

Execute the surrogate quality gates to ensure adherence to drift stability constraints before committing to broader pipeline deployment.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli eval-surrogate-gate \
  --manifest artifacts/full_shards/surrogate_test/surrogate_training_manifest.json \
  --checkpoint artifacts/surrogate_training/full_v1/surrogate_checkpoint.pth \
  --out artifacts/surrogate_training/full_v1/gates/test_quality_gate.json \
  --device cuda \
  --batch-size 128
```

## Citation

If you find this work or code useful in your research, please consider citing our paper:

```bibtex
@article{srivastava2026robustsep,
  title={RobustSep: A Non-Deterministic Architecture for Robust, Drift-Aware Color Separation in Packaging},
  author={Srivastava et al.},
  journal={TBD},
  year={2026}
}
```

## Authors & Acknowledgements

- **Shreeya Srivastava$^\dagger$** - Constrained Image-Synthesis Lab
- **Angad Singh Ahuja$^\dagger$** - Constrained Image-Synthesis Lab
- **Pranjal Wala$^\dagger$** - Constrained Image-Synthesis Lab
- **Rushabh Lodha** - IIT Mandi

*$^\dagger$ Equal Contribution*

> Please refer to the `paper/main/` directory for the full LaTeX compilation of the manuscript.
