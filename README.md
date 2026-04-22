# RobustSep

**A Non-Deterministic Architecture for Robust, Drift-Aware Color Separation in Packaging**

---

## Abstract

RGB-to-CMYK color separation for packaging is an underdetermined inverse problem: for any target appearance, a continuous family of ink configurations exists that produces perceptually equivalent printed output. Existing approaches either assume a fully characterized, stable press an assumption that breaks down in practice or learn an unconstrained mapping that ignores the hard physical limits packaging production imposes.

RobustSep addresses both gaps within a single architecture. Rather than committing to a single deterministic output, the system generates a population of CMYKOGV ink candidates via a conditional variational autoencoder (CVAE), enforces printability through a two-pass physics-informed refiner, and selects the candidate that minimizes worst-case perceptual error across a distribution of simulated drift conditions. Selection uses a 90th-percentile tail-risk criterion over the intent-weighted CIEDE2000 error distribution : prioritizing candidates that remain acceptable under adverse process variation, not merely on average.

All press-specific knowledge is externalized into the **Press Prior Package (PPP)**: a structured inference-time object encoding process-family constraints and drift distributions without requiring proprietary calibration data. Swapping the PPP adapts the pipeline to a new press family without retraining.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   Press Prior Package (PPP)                 │
│       process-family constraints · drift distributions      │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
  ┌─────────────────┐      ┌──────────────────┐
  │   Proposer      │      │ Drift simulation │
  │   (CVAE)        │      │ (PPP-sampled)    │
  │                 │      └────────┬─────────┘
  │ CMYKOGV cands.  │               │
  └────────┬────────┘               │
           │                        │
           ▼                        │
  ┌─────────────────┐               │
  │    Refiner      │               │
  │ (Deterministic) │               │
  │                 │               │
  │ ΠK · TAC · neu- │               │
  │ tral stability  │               │
  └────────┬────────┘               │
           │                        │
           ▼                        │
  ┌────────────────────────────────-┘
  │    Evaluator                    │
  │  (Micro-CNN surrogate)          │
  │                                 │
  │  CIEDE2000 error distribution   │
  └────────┬────────────────────────┘
           │
           ▼
  ┌─────────────────┐
  │    Selector     │
  │                 │
  │  90th-pct tail- │
  │  risk criterion │
  └────────┬────────┘
           │
           ▼
   Optimal CMYK separation
   + decision report R
```

**Proposer.** A CVAE generates diverse CMYKOGV candidates conditioned on the PPP, target RGB patch, structure token, intent map, and a style knob λ ∈ {0.1, 0.5, 0.9} encoding safety-to-saturation intent. Three candidates are produced per patch by default (up to five in research mode), each with a distinct deterministic seed. Stochasticity here is deliberate: the selector can only achieve robustness over the diversity it is given.

**Refiner.** Every candidate passes unconditionally through a two-pass deterministic refiner. Pass 1 applies spatial regularity smoothing and neutral/dark stabilization to suppress high-frequency separation artifacts. Pass 2 applies the projection operator Π_K: each channel is clipped to its PPP cap, then total area coverage (TAC) is scaled back to the PPP ceiling minus a small numerical margin. The output is guaranteed to lie inside the PPP feasibility envelope.

**Evaluator.** A lightweight Micro-CNN surrogate (8-block dilated CNN with FiLM conditioning) approximates ICC-based press rendering over a 32×32 context window, scored on the inner 16×16 region. For each feasible candidate, 32 drift instances are sampled from the PPP drift distribution and applied channel-wise. The surrogate is queried under each drift instance, producing a per-candidate distribution of CIEDE2000 errors rather than a single scalar.

**Selector.** Candidates are ranked by 90th-percentile intent-weighted CIEDE2000 error — a tail-risk criterion that eliminates candidates with catastrophic worst-case behavior even when their median error is low. Ties are broken by nominal error under identity drift. If no candidate survives the robustness gate, an escalation policy first increases K up to five, then (as a secondary fallback) increases the drift sample count. All escalation events are recorded in the decision report R.

### Press Prior Package (PPP)

The PPP is a two-tier JSON object. The first tier is a base family token mapping to pre-calibrated ink constraints, drift parameters, and neutral detection thresholds. The second tier is a sparse override layer for job-specific values such as a measured TAC ceiling or tighter per-channel caps; a binary override mask distinguishes deliberately set fields from inherited defaults.

At inference time, the resolved PPP is encoded by a two-layer MLP into a 128-dimensional embedding, concatenated with a structure embedding (32-d), an intent embedding (128-d), and λ to form a unified conditioning vector c_u ∈ ℝ^289. This vector drives Feature-wise Linear Modulation (FiLM) throughout both the CVAE proposer and the forward surrogate.

Five base families are defined in v1:

| Base family | Process / Substrate |
|---|---|
| `film_generic_conservative` | Generic packaging films (default) |
| `film_gravure_generic` | Gravure on film |
| `film_flexo_generic` | Flexo on film |
| `paperboard_generic` | Paperboard and carton |
| `label_stock_generic` | Label stocks |

### Intent Classification

Each pixel is assigned to one of three intent classes from local image statistics, with no learned components:

- **Brand** — logos, wordmarks, and spot-color regions. Colorimetric accuracy is a contractual requirement.
- **Gradient** — smooth tonal transitions, vignettes, skin tones. Primary failure mode is contouring.
- **Flat** — uniform fields where tonal stability and absence of banding dominate.

Classification resolves in strict priority order (Brand > Gradient > Flat). Per-patch intent weights enter the pipeline both as spatial input channels to the proposer's first layer and as a coarse intent raster processed through a dedicated MLP, both propagating through FiLM into all downstream blocks.

### Complexity

Let H × W be image resolution, s = 16 the patch size, t the overlap stride, K the candidate count, N the drift samples per candidate, T_F the surrogate forward-pass cost, and T_R the refiner cost per candidate. The number of patches is approximately M ≈ ⌈H/t⌉ · ⌈W/t⌉. Total inference cost is:

```
T_total = O(M · K · (N · T_F + T_R))
```

Memory is dominated by patch buffering and candidate storage. Storing all candidates concurrently scales as O(M · K · s² · 7); a streaming implementation reduces this to O(K · s² · 7) per patch.

---

## Repository Structure

```
RobustSepTraining/
├── data/                  # Raw and processed datasets
│                          # (DocLayNet, SKU110k, RobustSep splits)
├── paper/                 # LaTeX source for the manuscript
│   └── main/              # Compiled paper (PDF)
├── robustsep_pkg/         # Core Python package
│   ├── data/              # Adapters, intent parsing, batching,
│   │                      # A-Res source weighting (Vitter 1985)
│   ├── docs/              # Architectural specs and engineering docs
│   ├── engine/            # Multi-shot escalation and pipeline loop
│   ├── surrogate_data/    # Surrogate training context extraction
│   └── targets/           # Dual-stage projected gradient solver
│                          # for ground-truth generation
├── scripts/               # Data preparation and pipeline utilities
└── tests/                 # Unit test suite (139+ tests passing)
```

---

## Installation

Python 3.10 or higher is required.

```bash
git clone https://github.com/AngadSingh22/RobustSep.git
cd RobustSepTraining

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

---

## Usage

### 1. Data preparation and shard generation

Generate surrogate target shards. Source weighting uses A-Res reservoir sampling (Vitter, 1985).

```bash
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

### 2. Forward surrogate training

Train the Micro-CNN surrogate under symmetric ICC supervision across nominal and drifted conditions.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli train-surrogate \
  --manifest artifacts/full_shards/surrogate_train/surrogate_training_manifest.json \
  --out-dir artifacts/surrogate_training/full_v1 \
  --device cuda \
  --batch-size 128 \
  --epochs 1 \
  --learning-rate 0.001
```

### 3. Quality gate evaluation

Before deployment, the surrogate must pass a four-metric quality gate (mean ΔE₀₀, 90th-percentile ΔE₀₀, Spearman rank correlation, and top-1 agreement) against a fixed held-out drift bank, evaluated per PPP base family.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. .venv/bin/python -m robustsep_pkg.cli eval-surrogate-gate \
  --manifest artifacts/full_shards/surrogate_test/surrogate_training_manifest.json \
  --checkpoint artifacts/surrogate_training/full_v1/surrogate_checkpoint.pth \
  --out artifacts/surrogate_training/full_v1/gates/test_quality_gate.json \
  --device cuda \
  --batch-size 128
```

---

## Evaluation

Evaluation is planned across five PPP base families on 1,000 RGB images processed patch-wise. Metrics cover three axes:

**Color fidelity** — CIEDE2000 (ΔE₀₀), MSE, PSNR, SSIM.  
**Structural preservation** — edge error (%).  
**Print feasibility** — TAC compliance, per-channel ink consumption, neutral gray balance, K-channel distribution.

Robustness is assessed at the 50th, 75th, and 90th percentiles of the ΔE₀₀ distribution across drift conditions. Ablations cover: Refiner on/off, Surrogate on/off, PPP constraints on/off, λ sweep (0.1 → 0.9), K = 3 vs K = 5, and per-PPP-family comparisons.

Quantitative results will be reported once experimental runs are complete.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{srivastava2026robustsep,
  title   = {RobustSep: A Non-Deterministic Architecture for Robust,
             Drift-Aware Color Separation in Packaging},
  author  = {Srivastava, Ahuja, Wala and Lodha}
  journal = {TBD},
  year    = {2026}
}
```

---

## Authors

Shreeya Srivastava\* · Angad Singh Ahuja\* · Pranjal Wala\* — Constrained Image-Synthesis Lab  
Rushabh Lodha — IIT Mandi

\* Equal contribution

Full manuscript: [`paper/main/`](paper/main/)
