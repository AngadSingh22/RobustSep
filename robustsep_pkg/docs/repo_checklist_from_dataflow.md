# Repo Checklist from Dataflow Spec

Derived from `docs/VAE_FS_PARGB2CMYK.md`.

## 1. PPP Parsing and Embedding
- **Module**: `robustsep_pkg/models/conditioning/ppp.py`
- **Inputs**: 
  - `ppp_package` (base family, overrides, mask)
- **Outputs**: 
  - `ppp_embedding` (vector, dim=128)
  - `feasibility_envelope` (caps, limits)
  - `drift_distribution` (params)
- **Determinism Hooks**: None (deterministic transformation).
- **Tests**: `test_ppp_embedding_shape`, `test_override_masking`.
- **CLI Entrypoint**: Implicit via `generate-synth` or `infer`.

## 2. Structure and Intent Embedding
- **Module**: `robustsep_pkg/models/conditioning/embeddings.py` (New stub needed)
- **Inputs**: 
  - `structure_token` (enum)
  - `intent_weights` (vector)
  - `intent_raster` (low-res grid)
- **Outputs**: 
  - `structure_embedding` (vector, dim=32)
  - `intent_embedding` (vector, dim=128)
- **Determinism Hooks**: None.
- **Tests**: `test_structure_mapping`, `test_intent_raster_flattening`.
- **CLI Entrypoint**: Implicit.

## 3. FiLM Block Generators
- **Module**: `robustsep_pkg/models/conditioning/film.py` (New stub needed)
- **Inputs**: 
  - `conditioning_vector` (concat of PPP, structure, intent, lambda)
- **Outputs**: 
  - `scale` and `shift` vectors per channel for each block.
- **Determinism Hooks**: Weights initialization (training only).
- **Tests**: `test_film_dimension_matching`.
- **CLI Entrypoint**: Implicit.

## 4. Proposer (Conditional VAE)
- **Module**: `robustsep_pkg/models/proposer/estimator.py`
- **Inputs**: 
  - `rgb_patch`
  - `conditioning_vector`
  - `seed_policy` keys: (run_seed, patch_id, lambda_id, candidate_idx)
- **Outputs**: 
  - `cmykogv_candidate` (0-1 tensor)
- **Determinism Hooks**: `SeedPolicy.get_proposer_seed(...)` for latent sampling.
- **Tests**: `test_proposer_determinism`, `test_lambda_monotonicity`.
- **CLI Entrypoint**: `train-proposer`, `infer`.

## 5. Drift Sampler
- **Module**: `robustsep_pkg/models/conditioning/drift.py`
- **Inputs**: 
  - `ppp_drift_distribution`
  - `seed_policy` keys: (run_seed, patch_id)
- **Outputs**: 
  - `drift_instances` (list of perturbations: multipliers + curves)
- **Determinism Hooks**: `SeedPolicy.get_drift_seed(...)`.
- **Tests**: `test_drift_determinism`, `test_drift_clipping`.
- **CLI Entrypoint**: `generate-synth`, `eval-gate`.

## 6. Forward Surrogate
- **Module**: `robustsep_pkg/models/surrogate/model.py`
- **Inputs**: 
  - `cmykogv_context`
  - `drift_instance`
  - `conditioning_vector`
- **Outputs**: 
  - `predicted_lab` (center region)
- **Determinism Hooks**: None (deterministic inference).
- **Tests**: `test_surrogate_shape`, `test_surrogate_film_injection`.
- **CLI Entrypoint**: `train-surrogate`, `eval-gate`.

## 7. Refiner (Solver)
- **Module**: `robustsep_pkg/models/refiner/solver.py`
- **Inputs**: 
  - `cmykogv_candidate`
  - `feasibility_envelope` (PPP)
- **Outputs**: 
  - `feasible_candidate`
- **Determinism Hooks**: None (deterministic projection).
- **Tests**: `test_tac_cap_enforcement`, `test_channel_limit_clipping`.
- **CLI Entrypoint**: `infer`.

## 8. Metrics
- **Module**: `robustsep_pkg/eval/metrics.py`
- **Inputs**: 
  - `predicted_lab`
  - `reference_lab`
  - `intent_weights`
- **Outputs**: 
  - `delta_e_00`
  - `weighted_risk` (0.90 quantile)
- **Determinism Hooks**: None.
- **Tests**: `test_delta_e_implementation`, `test_risk_aggregation`.
- **CLI Entrypoint**: `eval-gate`.

## 9. Quality Gates
- **Module**: `robustsep_pkg/gates/quality_gate.py`
- **Inputs**: 
  - `metrics_dict`
  - `ppp_thresholds`
- **Outputs**: 
  - `pass_fail` boolean
- **Determinism Hooks**: None.
- **Tests**: `test_gate_thresholds`.
- **CLI Entrypoint**: `eval-gate`.

## 10. Logging and Manifests
- **Module**: `robustsep_pkg/manifests/run_manifest.py` & `robustsep_pkg/core/artifact_io.py`
- **Inputs**: 
  - `run_config`
  - `run_seed`
  - `git_info`
- **Outputs**: 
  - `run_manifest.json` (with sidecar metadata)
- **Determinism Hooks**: Records `run_seed` and policy version.
- **Tests**: `test_manifest_serializability`.
- **CLI Entrypoint**: All.

## 11. Config Knobs
- **Module**: `robustsep_pkg/core/config.py`
- **Inputs**: 
  - `config.yaml`
  - `cli_overrides`
- **Outputs**: 
  - `resolved_config` (frozen dict)
- **Determinism Hooks**: Hashes config for reproducibility.
- **Tests**: `test_config_resolution`.
- **CLI Entrypoint**: `print-config`.
