# RobustSep Dataset Implementation Plan

## Objective

Build a reproducible RobustSep training-data staging pipeline that turns raw RGB sources into
patch records with:

- 16x16 RGB patches
- alpha-aware crop/filter metadata
- structure labels: `flat`, `edge`, `textured`
- color labels: `saturated`, `neutral`, `dark`, `normal`
- Lab reference values
- deterministic CMYK baseline values
- PPP-projected CMYK values
- file-level and run-level manifests

This is not a replacement for the final ICC/spectral teacher. It is the strict first staging
layer needed before full synthetic CMYKOGV solving.

## Raw Sources

1. Existing vector artwork:
   - `data/png_full/png`
   - `data/png_full/svg`
   - paired PNG/SVG assets already verified by matching stems.

2. DocLayNet v1.2:
   - source: `docling-project/DocLayNet-v1.2` on Hugging Face
   - target directory: `data/external/raw/doclaynet`
   - expected payload: parquet shards for train, validation, and test splits.

3. SKU110K:
   - source: official project README and S3 payload
   - target directory: `data/external/raw/sku110k`
   - expected payload: `SKU110K_fixed.tar.gz`.

4. RGB/color coverage:
   - Kaggle requires authenticated API access in this environment.
   - deterministic local color supplements are generated under
     `data/external/processed/color_supplements`.

## Processing Stages

1. Quarantine corrupt PNGs.
   - Move unreadable PNGs to `data/quarantine/corrupt_png`.
   - Write a manifest entry for each moved file.

2. Alpha-aware patch sampling.
   - Load images as RGBA.
   - Compute visible alpha bounding box with `alpha >= 16`.
   - Crop to that box when it removes large transparent margins.
   - Sample 16x16 patches with deterministic stride/random seed.
   - Reject patches with visible alpha coverage below threshold.
   - Preserve partial-alpha edge patches deliberately.

3. Balance patch buckets.
   - Structure bucket from luminance/alpha edge proxy:
     - `flat`
     - `edge`
     - `textured`
   - Color bucket:
     - `dark`
     - `neutral`
     - `saturated`
     - `normal`
   - Cap each combined bucket to avoid over-representing transparent line-art or one source family.

4. Generate supplements.
   - RGB grid swatches for broad color gamut.
   - Smooth gradients for interpolation behavior.
   - Dark and near-neutral ramps.
   - High-saturation ramps for neon-like colors.

5. Generate baseline targets.
   - Convert RGB to linear RGB.
   - Convert linear RGB to XYZ with sRGB primaries.
   - Adapt D65 to D50 using Bradford adaptation.
   - Convert XYZ D50 to Lab.
   - Build deterministic CMYK baseline with gray-component replacement.
   - Apply conservative PPP projection:
     - channel caps: C/M/Y/K <= 1.0
     - TAC <= 3.0 by proportional scaling.

6. Manifests and hashes.
   - Patch metadata is emitted as JSONL.
   - Patch arrays are emitted as compressed NPZ shards.
   - A run manifest records parameters, source counts, output counts, and hashes.

## Current Limitations

- The original local CMYK baseline is deterministic and useful for staging; the current
  processed shards also include ICC/profile-derived CMYK baselines.
- The generated target is CMYK, not final CMYKOGV. OGV requires the later constrained solver.
- DocLayNet parquet processing has been run with `pyarrow` in the local virtual environment.
- SKU110K has been downloaded, extracted, patch-sampled, and ICC-processed.

## Architecture Math Resolution Pass

The dataset layer is now ahead of the model implementation. Before implementing
`robustsep_pkg`, the remaining architecture work is to consolidate the mathematically TBS
items in `RobustSep-v1.1/docs/final_arch.md` against the more detailed VAE/surrogate math in
`RobustSep-v1.1/docs/vae_fs_pargb2cmyk.md`.

Resolution workflow:

1. Treat `vae_fs_pargb2cmyk.md` as the controlling source for already-defined proposer,
   surrogate, target-generation, drift, feasibility, and risk semantics.
2. For each remaining TBS item, run a three-iteration Newton/Socrates design loop:
   - Newton proposes the strongest implementable mathematical definition consistent with the
     VAE architecture.
   - Socrates attacks ambiguity, reproducibility gaps, degenerate cases, and places where the
     definition could violate feasibility, robustness-first selection, or replayability.
   - The accepted result must resolve exact inputs/outputs, equations, deterministic
     pseudocode, correctness argument, and manifest requirements.
3. Collapse overlapping TBS items into shared primitives where appropriate:
   - PPP effective profile and feasibility set `K(PPP)`.
   - Projection operator `Pi_K`.
   - Deterministic patch geometry and seed policy.
   - Intent/structure statistics.
   - Drift bank construction.
   - Patch risk, escalation, blending, and global safety pass.
4. Update `final_arch.md` only after the loop outcomes are internally consistent.

Immediate implementation order after the math pass:

1. Scaffold `robustsep_pkg` with PPP parsing, config, manifests, and metrics.
2. Implement deterministic preprocessing: intent maps, patch aggregation, structure tokens.
3. Implement feasibility envelope, projection, drift sampler, DeltaE00/risk, and decision reports.
4. Implement CMYKOGV synthetic target solver from the staged ICC CMYK baselines.
5. Train the forward surrogate on GPU, then train the FiLM-conditioned VAE proposer.

Accepted resolution principles from the Newton/Socrates pass:

- Use CMYKOGV as the canonical internal candidate space. Export/downstream consumers may
  collapse to CMYK only through an explicitly documented export operator.
- Treat every patch as `(RGB, alpha)` where alpha is available. Transparent pixels are excluded
  from losses and risks; partial-alpha edge pixels are retained with a fixed alpha exponent.
- Define PPP feasibility as explicit per-pixel inequalities over seven channels: channel caps,
  TAC, OGV total cap, optional opponent-pair caps, and neutral/dark OGV caps.
- Use a deterministic projection closure: clip channels, enforce group caps by downward
  scaling, then enforce TAC by capped-simplex water filling. This preserves already-feasible
  pixels and closes all hard constraints.
- Define drift with seeded truncated-lognormal ink multipliers plus monotone TRC curves
  generated from fixed knots and deterministic isotonic projection.
- Define risk with non-interpolated finite order statistics: `sort(errors)[ceil(q*N)-1]`.
- Select candidates by a total deterministic ordering: feasibility, risk threshold status, risk,
  nominal error, TAC, lambda-aware OGV/chroma tie-break, then candidate index.
- Escalation appends candidates without resampling existing ones; candidate coverage increases
  before drift sample count.
- The manifest must record enough to replay patch choices: canonical PPP, image/data hashes,
  model/ICC hashes, seed derivation, patch grid, alpha policy, kernels, thresholds, projection,
  drift bank, quantile policy, and selection ordering.
