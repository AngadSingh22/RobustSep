from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from robustsep_pkg.core.artifact_io import canonical_json_hash, write_json
from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.eval.metrics import delta_e_00, finite_quantile
from robustsep_pkg.models.conditioning.drift import apply_drift, sample_drift_bank
from robustsep_pkg.models.conditioning.ppp import PPP, feasibility_violations, neutral_or_dark_mask
from robustsep_pkg.preprocess.color import cmyk_to_cmykogv
from robustsep_pkg.targets.generation_pipeline import TargetGenerationPipelineConfig, iter_generated_target_records
from robustsep_pkg.targets.teacher import calibrated_cmykogv_lab


PAPER_EVAL_REQUIREMENTS = {
    "color_fidelity": ["delta_e00", "mse_lab", "psnr_lab", "ssim_l"],
    "structure_preservation": ["edge_error_pct"],
    "print_feasibility": [
        "tac_mean",
        "tac_p90",
        "tac_exceedance_pct",
        "ink_channel_means",
        "gray_balance_delta_chroma",
        "k_mean",
        "k_std",
        "feasibility_violations",
    ],
    "robustness_percentiles": ["delta_e00_p50", "delta_e00_p75", "delta_e00_p90", "edge_error_p90", "tac_p90"],
    "ablation": ["full_model", "without_refiner", "without_surrogate", "without_ppp_constraints"],
}


@dataclass(frozen=True)
class PaperEvalConfig:
    max_records: int | None = 128
    drift_samples: int = 32
    root_seed: int = 20260422
    visual_examples: int = 8
    edge_threshold: float = 2.0
    psnr_peak: float = 100.0
    drift_config: DriftConfig = field(default_factory=DriftConfig)
    target_config: TargetGenerationPipelineConfig = field(default_factory=TargetGenerationPipelineConfig)


@dataclass(frozen=True)
class PaperEvalSummary:
    output_json: str
    records_evaluated: int
    ppp_hash: str
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_paper_evaluation_suite(
    split_manifest: str | Path,
    ppp: PPP,
    output_json: str | Path,
    *,
    root: str | Path | None = None,
    config: PaperEvalConfig = PaperEvalConfig(),
    visual_npz: str | Path | None = None,
) -> PaperEvalSummary:
    """Run the paper Evaluation-section protocol on generated target records."""
    rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, np.ndarray]] = []

    records = _limit(
        iter_generated_target_records(split_manifest, ppp, root=root, config=config.target_config),
        config.max_records,
    )
    for idx, record in enumerate(records):
        sample = record.enriched_sample.sample
        lab_ref = np.asarray(record.enriched_sample.lab, dtype=np.float32)
        full = np.asarray(record.target_result.target_cmykogv, dtype=np.float32)
        projected_icc = np.asarray(record.target_result.initial_cmykogv, dtype=np.float32)
        raw_icc = cmyk_to_cmykogv(sample.icc_cmyk)
        source_id = record.target_result.manifest_record.source_id or f"record-{idx}"
        variants = {
            "full_model": full,
            "without_refiner": projected_icc,
            "without_surrogate": full,
            "without_ppp_constraints": raw_icc,
        }
        anchor = projected_icc
        drift_bank = sample_drift_bank(
            config.drift_config,
            config.root_seed,
            source_id,
            ppp.hash,
            (record.sample_record.x, record.sample_record.y),
            sample_count=config.drift_samples,
        )
        variant_metrics = {
            name: _evaluate_variant(values, lab_ref, anchor, ppp, drift_bank, config)
            for name, values in variants.items()
        }
        variant_metrics["without_surrogate"]["note"] = (
            "Current target generation is ICC/calibrated-teacher only; no gated surrogate blend is enabled, "
            "so this variant is equivalent to full_model until hybrid target generation is unlocked."
        )
        rows.append(
            {
                "index": idx,
                "family": record.family_name,
                "split": record.split,
                "source_path": record.sample_record.source_path,
                "x": record.sample_record.x,
                "y": record.sample_record.y,
                "target_hash": record.target_result.manifest_record.target_hash,
                "variants": variant_metrics,
            }
        )
        if len(visual_rows) < config.visual_examples:
            full_lab = _nominal_lab(full, lab_ref, anchor)
            visual_rows.append(
                {
                    "lab_ref": lab_ref,
                    "full_lab": full_lab,
                    "delta_e00": delta_e_00(full_lab, lab_ref),
                    "full_cmykogv": full,
                    "projected_icc_cmykogv": projected_icc,
                }
            )

    report = _build_report(rows, ppp, config)
    out_path = Path(output_json)
    write_json(out_path, report)
    if visual_npz is not None:
        _write_visual_npz(visual_npz, visual_rows)
    return PaperEvalSummary(
        output_json=str(out_path),
        records_evaluated=len(rows),
        ppp_hash=ppp.hash,
        report_hash=canonical_json_hash(report),
    )


def _evaluate_variant(
    values: np.ndarray,
    lab_ref: np.ndarray,
    anchor: np.ndarray,
    ppp: PPP,
    drift_bank: Iterable[Any],
    config: PaperEvalConfig,
) -> dict[str, Any]:
    nominal_lab = _nominal_lab(values, lab_ref, anchor)
    de = delta_e_00(nominal_lab, lab_ref)
    tac = values.sum(axis=-1)
    k = values[..., 3]
    neutral = neutral_or_dark_mask(lab_ref, ppp)
    gray_balance = _gray_balance_delta_chroma(nominal_lab, lab_ref, neutral)
    drift_mean_de: list[float] = []
    drift_edge_error: list[float] = []
    drift_tac: list[float] = []
    for drift in drift_bank:
        drifted = apply_drift(values, drift)
        drift_lab = calibrated_cmykogv_lab(drifted, anchor_cmykogv=values, anchor_lab=nominal_lab)
        drift_de = delta_e_00(drift_lab, lab_ref)
        drift_mean_de.append(float(np.mean(drift_de)))
        drift_edge_error.append(edge_error_pct(drift_lab[..., 0], lab_ref[..., 0], threshold=config.edge_threshold))
        drift_tac.append(float(np.mean(drifted.sum(axis=-1))))
    return {
        "color_fidelity": {
            "delta_e00_mean": float(np.mean(de)),
            "delta_e00_p50": finite_quantile(de, 0.50),
            "delta_e00_p75": finite_quantile(de, 0.75),
            "delta_e00_p90": finite_quantile(de, 0.90),
            "mse_lab": mse(nominal_lab, lab_ref),
            "psnr_lab": psnr(nominal_lab, lab_ref, peak=config.psnr_peak),
            "ssim_l": ssim(nominal_lab[..., 0], lab_ref[..., 0], data_range=config.psnr_peak),
        },
        "structure_preservation": {
            "edge_error_pct": edge_error_pct(nominal_lab[..., 0], lab_ref[..., 0], threshold=config.edge_threshold),
        },
        "print_feasibility": {
            "tac_mean": float(np.mean(tac)),
            "tac_p50": finite_quantile(tac, 0.50),
            "tac_p75": finite_quantile(tac, 0.75),
            "tac_p90": finite_quantile(tac, 0.90),
            "tac_exceedance_pct": float(100.0 * np.mean(tac > ppp.tac_max)),
            "ink_channel_means": _channel_means(values),
            "gray_balance_delta_chroma": gray_balance,
            "k_mean": float(np.mean(k)),
            "k_std": float(np.std(k)),
            "feasibility_violations": feasibility_violations(values, ppp, lab_ref=lab_ref),
        },
        "robustness_percentiles": {
            "delta_e00_p50": finite_quantile(np.asarray(drift_mean_de, dtype=np.float32), 0.50),
            "delta_e00_p75": finite_quantile(np.asarray(drift_mean_de, dtype=np.float32), 0.75),
            "delta_e00_p90": finite_quantile(np.asarray(drift_mean_de, dtype=np.float32), 0.90),
            "edge_error_p50": finite_quantile(np.asarray(drift_edge_error, dtype=np.float32), 0.50),
            "edge_error_p75": finite_quantile(np.asarray(drift_edge_error, dtype=np.float32), 0.75),
            "edge_error_p90": finite_quantile(np.asarray(drift_edge_error, dtype=np.float32), 0.90),
            "tac_p50": finite_quantile(np.asarray(drift_tac, dtype=np.float32), 0.50),
            "tac_p75": finite_quantile(np.asarray(drift_tac, dtype=np.float32), 0.75),
            "tac_p90": finite_quantile(np.asarray(drift_tac, dtype=np.float32), 0.90),
        },
        "distributions": {
            "delta_e00": distribution_summary(de),
            "tac": distribution_summary(tac),
            "k": distribution_summary(k),
        },
    }


def _build_report(rows: list[dict[str, Any]], ppp: PPP, config: PaperEvalConfig) -> dict[str, Any]:
    variants = ("full_model", "without_refiner", "without_surrogate", "without_ppp_constraints")
    aggregate = {variant: _aggregate_variant(rows, variant) for variant in variants}
    return {
        "suite": "paper_evaluation_v1",
        "paper_requirements": PAPER_EVAL_REQUIREMENTS,
        "config": {
            "max_records": config.max_records,
            "drift_samples": config.drift_samples,
            "root_seed": config.root_seed,
            "visual_examples": config.visual_examples,
            "edge_threshold": config.edge_threshold,
        },
        "ppp": ppp.to_dict(),
        "ppp_hash": ppp.hash,
        "records_evaluated": len(rows),
        "quantitative_results": aggregate["full_model"]["quantitative_results"],
        "percentile_robustness": aggregate["full_model"]["percentile_robustness"],
        "distribution_analysis": aggregate["full_model"]["distribution_analysis"],
        "ablation": {variant: aggregate[variant]["ablation_row"] for variant in variants},
        "rows": rows,
    }


def _aggregate_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        empty = {
            "delta_e00": 0.0,
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "edge_error": 0.0,
            "tac": 0.0,
            "ink_usage": {},
            "gray_balance": 0.0,
            "k_mean": 0.0,
            "k_std": 0.0,
        }
        return {"quantitative_results": empty, "percentile_robustness": {}, "distribution_analysis": {}, "ablation_row": empty}
    metrics = [row["variants"][variant] for row in rows]
    color = [m["color_fidelity"] for m in metrics]
    structure = [m["structure_preservation"] for m in metrics]
    feasibility = [m["print_feasibility"] for m in metrics]
    robustness = [m["robustness_percentiles"] for m in metrics]
    quantitative = {
        "delta_e00": _mean_key(color, "delta_e00_mean"),
        "mse": _mean_key(color, "mse_lab"),
        "psnr": _mean_key(color, "psnr_lab"),
        "ssim": _mean_key(color, "ssim_l"),
        "edge_error": _mean_key(structure, "edge_error_pct"),
        "tac": _mean_key(feasibility, "tac_mean"),
        "ink_usage": _aggregate_channel_means([m["ink_channel_means"] for m in feasibility]),
        "gray_balance": _mean_key(feasibility, "gray_balance_delta_chroma"),
        "k_mean": _mean_key(feasibility, "k_mean"),
        "k_std": _mean_key(feasibility, "k_std"),
        "tac_exceedance_pct": _mean_key(feasibility, "tac_exceedance_pct"),
    }
    percentile = {
        "delta_e00": {
            "p50": _mean_key(robustness, "delta_e00_p50"),
            "p75": _mean_key(robustness, "delta_e00_p75"),
            "p90": _mean_key(robustness, "delta_e00_p90"),
        },
        "edge_error": {
            "p50": _mean_key(robustness, "edge_error_p50"),
            "p75": _mean_key(robustness, "edge_error_p75"),
            "p90": _mean_key(robustness, "edge_error_p90"),
        },
        "tac": {
            "p50": _mean_key(robustness, "tac_p50"),
            "p75": _mean_key(robustness, "tac_p75"),
            "p90": _mean_key(robustness, "tac_p90"),
        },
    }
    distributions = {
        "delta_e00": distribution_summary([m["color_fidelity"]["delta_e00_mean"] for m in metrics]),
        "tac": distribution_summary([m["print_feasibility"]["tac_mean"] for m in metrics]),
        "k": distribution_summary([m["print_feasibility"]["k_mean"] for m in metrics]),
    }
    return {
        "quantitative_results": quantitative,
        "percentile_robustness": percentile,
        "distribution_analysis": distributions,
        "ablation_row": {
            "delta_e00": quantitative["delta_e00"],
            "tac": quantitative["tac"],
            "edge_error": quantitative["edge_error"],
            "notes": _variant_note(rows, variant),
        },
    }


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, *, peak: float = 100.0) -> float:
    err = mse(a, b)
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(err))


def ssim(a: np.ndarray, b: np.ndarray, *, data_range: float = 100.0) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    vx = float(np.var(x))
    vy = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    denom = (mux * mux + muy * muy + c1) * (vx + vy + c2)
    if denom <= 1e-12:
        return 1.0
    return float(((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / denom)


def edge_error_pct(pred_l: np.ndarray, ref_l: np.ndarray, *, threshold: float = 2.0) -> float:
    pred_edges = _edge_magnitude(pred_l) > threshold
    ref_edges = _edge_magnitude(ref_l) > threshold
    return float(100.0 * np.mean(np.logical_xor(pred_edges, ref_edges)))


def distribution_summary(values: Any) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": finite_quantile(arr, 0.50),
        "p75": finite_quantile(arr, 0.75),
        "p90": finite_quantile(arr, 0.90),
        "max": float(np.max(arr)),
    }


def _nominal_lab(values: np.ndarray, lab_ref: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return calibrated_cmykogv_lab(values, anchor_cmykogv=anchor, anchor_lab=lab_ref)


def _edge_magnitude(luma: np.ndarray) -> np.ndarray:
    y = np.asarray(luma, dtype=np.float32)
    gy = np.zeros_like(y)
    gx = np.zeros_like(y)
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def _gray_balance_delta_chroma(pred_lab: np.ndarray, ref_lab: np.ndarray, neutral: np.ndarray) -> float:
    if not np.any(neutral):
        return 0.0
    pred_chroma = np.sqrt(pred_lab[..., 1] ** 2 + pred_lab[..., 2] ** 2)
    ref_chroma = np.sqrt(ref_lab[..., 1] ** 2 + ref_lab[..., 2] ** 2)
    return float(np.mean(np.abs(pred_chroma[neutral] - ref_chroma[neutral])))


def _channel_means(values: np.ndarray) -> dict[str, float]:
    names = ("C", "M", "Y", "K", "O", "G", "V")
    return {name: float(np.mean(values[..., idx])) for idx, name in enumerate(names)}


def _aggregate_channel_means(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _variant_note(rows: list[dict[str, Any]], variant: str) -> str | None:
    for row in rows:
        note = row["variants"][variant].get("note")
        if note:
            return str(note)
    return None


def _limit(records: Iterable[Any], max_records: int | None) -> Iterable[Any]:
    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be non-negative or None")
    for idx, record in enumerate(records):
        if max_records is not None and idx >= max_records:
            break
        yield record


def _write_visual_npz(path: str | Path, rows: list[dict[str, np.ndarray]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(out)
        return
    keys = rows[0].keys()
    np.savez_compressed(out, **{key: np.stack([row[key] for row in rows], axis=0) for key in keys})
