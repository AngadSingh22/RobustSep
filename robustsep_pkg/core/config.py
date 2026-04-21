from __future__ import annotations

from dataclasses import dataclass, field

from robustsep_pkg.core.channels import CHANNELS_CMYKOGV


@dataclass(frozen=True)
class SeedConfig:
    root_seed: int = 20260422
    rng_algorithm: str = "numpy.PCG64"
    hash_algorithm: str = "blake2b-uint64"


@dataclass(frozen=True)
class PreprocessConfig:
    patch_size: int = 16
    stride: int = 8
    alpha_gamma: float = 1.0
    alpha_min_patch: float = 1e-6
    alpha_visible_threshold: float = 16.0 / 255.0
    local_radius: int = 3
    eps: float = 1e-8
    theta_flat_var: float = 18.0
    theta_flat_edge: float = 2.5
    theta_grad_min: float = 1.5
    theta_grad_max: float = 35.0
    theta_grad_coh: float = 0.35
    theta_hue_smooth: float = 0.35
    theta_brand_chroma: float = 28.0
    theta_brand_edge: float = 4.0
    theta_brand_coh: float = 0.25
    theta_edge: float = 4.0
    theta_edge_density: float = 0.10
    theta_edge_coh: float = 0.25
    theta_flat_var_patch: float = 12.0
    theta_flat_density: float = 0.05


@dataclass(frozen=True)
class RiskConfig:
    q: float = 0.90
    tail_q: float = 0.95
    rho_tail: float = 0.0
    beta_brand: float = 2.0
    beta_gradient: float = 1.5
    beta_flat: float = 0.5
    eps: float = 1e-8


@dataclass(frozen=True)
class DriftConfig:
    sample_count: int = 32
    fallback_sample_count: int = 64
    trc_interior_knots: int = 7
    multiplier_sigma: dict[str, float] = field(default_factory=lambda: {c: 0.03 for c in CHANNELS_CMYKOGV})
    multiplier_clip: dict[str, float] = field(default_factory=lambda: {c: 0.12 for c in CHANNELS_CMYKOGV})
    trc_sigma: dict[str, float] = field(default_factory=lambda: {c: 0.015 for c in CHANNELS_CMYKOGV})
    trc_clip: dict[str, float] = field(default_factory=lambda: {c: 0.05 for c in CHANNELS_CMYKOGV})
