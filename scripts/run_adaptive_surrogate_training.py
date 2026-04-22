from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsep_pkg.core.artifact_io import write_json
from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig
from robustsep_pkg.models.surrogate.training import (
    SurrogateLossConfig,
    SurrogateQualityGateThresholds,
    SurrogateTrainingConfig,
    diagnose_surrogate_quality,
    train_surrogate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive RobustSep forward-surrogate training controller")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--epochs-per-round", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.65)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--probe-drift-samples", type=int, default=32)
    parser.add_argument("--probe-max-patches", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--threshold-mean", type=float, default=3.0)
    parser.add_argument("--threshold-q90", type=float, default=5.0)
    parser.add_argument("--threshold-spearman", type=float, default=0.80)
    parser.add_argument("--threshold-top1", type=float, default=0.80)
    parser.add_argument("--initial-hard-pixel-weight", type=float, default=0.0)
    parser.add_argument("--max-hard-pixel-weight", type=float, default=3.0)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    thresholds = SurrogateQualityGateThresholds(
        threshold_mean=args.threshold_mean,
        threshold_q90=args.threshold_q90,
        threshold_spearman=args.threshold_spearman,
        threshold_top1=args.threshold_top1,
    )
    probe = CandidateProbeConfig(
        drift_sample_count=args.probe_drift_samples,
        max_patches=args.probe_max_patches,
        batch_size=args.probe_batch_size,
        root_seed=args.seed,
    )

    checkpoint: str | None = None
    learning_rate = float(args.learning_rate)
    hard_pixel_weight = float(args.initial_hard_pixel_weight)
    rounds: list[dict[str, object]] = []

    for round_idx in range(int(args.rounds)):
        round_dir = out_root / f"round_{round_idx + 1:02d}"
        result = train_surrogate(
            args.manifest,
            round_dir,
            training_config=SurrogateTrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs_per_round,
                learning_rate=learning_rate,
                weight_decay=args.weight_decay,
                num_workers=0,
                seed=args.seed + round_idx,
                device=args.device,
                initial_checkpoint=checkpoint,
            ),
            gate_thresholds=thresholds,
            candidate_probe_config=probe,
            loss_config=SurrogateLossConfig(
                target_mode="teacher_proxy",
                hard_pixel_weight=hard_pixel_weight,
                hard_pixel_quantile=0.90,
            ),
        )
        diagnosis = diagnose_surrogate_quality(result.quality, thresholds)
        row = {
            "round": round_idx + 1,
            "checkpoint": result.checkpoint_path,
            "learning_rate": learning_rate,
            "hard_pixel_weight": hard_pixel_weight,
            "train_loss": result.train_loss,
            "quality": result.quality.to_dict(),
            "diagnosis": diagnosis,
        }
        rounds.append(row)
        write_json(out_root / "adaptive_training_report.json", {"rounds": rounds, "latest": row})
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)

        checkpoint = result.checkpoint_path
        if result.quality.passed:
            break
        failures = diagnosis["failures"]
        if failures.get("q90_delta_e00"):
            hard_pixel_weight = min(args.max_hard_pixel_weight, max(1.0, hard_pixel_weight * 1.5 if hard_pixel_weight > 0 else 1.0))
        if failures.get("spearman") or failures.get("top1_agreement"):
            learning_rate *= args.lr_decay

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
