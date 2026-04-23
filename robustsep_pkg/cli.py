from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from robustsep_pkg.core.artifact_io import canonical_json, write_json
from robustsep_pkg.core.config import DriftConfig
from robustsep_pkg.data import FamilyDataset, RobustSepDataset, SourceWeightPolicy, TrainingAdapter
from robustsep_pkg.models.conditioning.ppp import PPP
from robustsep_pkg.surrogate_data import SurrogateShardWriterConfig, write_surrogate_training_shards
from robustsep_pkg.targets import TargetGenerationPipelineConfig, TargetSolverConfig, iter_generated_target_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robustsep", description="RobustSep training-data pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("export-split-manifest", help="Export a TrainingAdapter v1.1 split manifest")
    split.add_argument("--family", action="append", required=True, help="Family mapping name=run_manifest.json")
    split.add_argument("--out", required=True, help="Output split manifest JSON")
    split.add_argument("--root", default=".", help="Root prepended to relative shard paths")
    split.add_argument("--split", default="train", choices=("train", "val", "test"))
    split.add_argument("--val-frac", type=float, default=0.10)
    split.add_argument("--test-frac", type=float, default=0.05)
    split.add_argument("--root-seed", type=int, default=20260422)
    split.add_argument("--alpha-policy", default="ones", choices=("ones", "visible_threshold", "passthrough"))
    split.add_argument("--weight", action="append", default=[], help="Optional source weight name=value")
    split.set_defaults(func=_cmd_export_split_manifest)

    targets = sub.add_parser("generate-targets", help="Generate CMYKOGV target JSONL from a v1.1 split manifest")
    _add_target_args(targets)
    targets.add_argument("--out", required=True, help="Output target records JSONL")
    targets.add_argument("--summary-out", help="Optional JSON summary path")
    targets.set_defaults(func=_cmd_generate_targets)

    shards = sub.add_parser("write-surrogate-shards", help="Generate targets and write surrogate training shards")
    _add_target_args(shards)
    shards.add_argument("--out-dir", required=True, help="Output directory for surrogate shard .npz/.jsonl files")
    shards.add_argument("--shard-size", type=int, default=4096)
    shards.add_argument("--run-id", default="surrogate_training_shards")
    shards.add_argument("--summary-out", help="Optional JSON summary path")
    shards.set_defaults(func=_cmd_write_surrogate_shards)

    train = sub.add_parser("train-surrogate", help="Train the forward surrogate and run the quality gate")
    train.add_argument("--manifest", required=True, help="Surrogate training manifest JSON")
    train.add_argument("--out-dir", required=True)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=20260422)
    train.add_argument("--device", default="auto")
    train.add_argument("--initial-checkpoint", default=None)
    train.add_argument("--loss-target-mode", default="teacher_delta", choices=("teacher_delta", "teacher_proxy", "lab_anchor"))
    train.add_argument("--hard-pixel-weight", type=float, default=0.0)
    train.add_argument("--hard-pixel-quantile", type=float, default=0.90)
    _add_gate_args(train)
    train.set_defaults(func=_cmd_train_surrogate)

    gate = sub.add_parser("eval-surrogate-gate", help="Evaluate a trained surrogate checkpoint against the candidate probe")
    gate.add_argument("--manifest", required=True, help="Surrogate training/eval manifest JSON")
    gate.add_argument("--checkpoint", required=True)
    gate.add_argument("--out", required=True, help="Output quality report JSON")
    gate.add_argument("--device", default="auto")
    gate.add_argument("--batch-size", type=int, default=32)
    _add_gate_args(gate)
    gate.set_defaults(func=_cmd_eval_surrogate_gate)

    paper_eval = sub.add_parser("run-paper-eval", help="Run the paper Evaluation-section metric suite")
    _add_target_args(paper_eval)
    paper_eval.add_argument("--out", required=True, help="Output evaluation report JSON")
    paper_eval.add_argument("--visual-npz", help="Optional NPZ with Lab/error-map visual example arrays")
    paper_eval.add_argument("--eval-drift-samples", type=int, default=32)
    paper_eval.add_argument("--visual-examples", type=int, default=8)
    paper_eval.add_argument("--edge-threshold", type=float, default=2.0)
    paper_eval.set_defaults(func=_cmd_run_paper_eval)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--ppp-base", default="film_generic_conservative")
    parser.add_argument("--ppp-overrides-json", default=None, help="JSON object with PPP overrides")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--drift-samples-per-patch", type=int, default=1)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--stage1-steps", type=int, default=8)
    parser.add_argument("--stage2-steps", type=int, default=4)
    parser.add_argument("--stage1-step-size", type=float, default=0.002)
    parser.add_argument("--stage2-step-size", type=float, default=0.001)
    parser.add_argument("--intent-raster-size", type=int, default=4)
    parser.add_argument("--alpha-l-min", type=float, default=10.0)
    parser.add_argument("--recompute-structure", action="store_true")


def _add_gate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold-mean", type=float, default=3.0)
    parser.add_argument("--threshold-q90", type=float, default=5.0)
    parser.add_argument("--threshold-spearman", type=float, default=0.80)
    parser.add_argument("--threshold-top1", type=float, default=0.80)
    parser.add_argument("--threshold-mean-regret", type=float, default=0.25)
    parser.add_argument("--threshold-q90-regret", type=float, default=1.0)
    parser.add_argument("--probe-drift-samples", type=int, default=32)
    parser.add_argument("--probe-max-patches", type=int, default=None)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-root-seed", type=int, default=20260422)


def _cmd_export_split_manifest(args: argparse.Namespace) -> int:
    families = []
    for item in args.family:
        name, manifest_path = _parse_mapping(item, "--family")
        dataset = RobustSepDataset(
            [manifest_path],
            root=args.root,
            split=args.split,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            root_seed=args.root_seed,
        )
        families.append(FamilyDataset(name, dataset))
    policy = SourceWeightPolicy(weights=dict(_parse_float_mapping(v, "--weight") for v in args.weight))
    adapter = TrainingAdapter(families, weight_policy=policy, root_seed=args.root_seed)
    adapter.export_split_manifest(args.out, alpha_policy=args.alpha_policy)
    print(json.dumps({"output": args.out, **adapter.summary()}, indent=2, sort_keys=True))
    return 0


def _cmd_generate_targets(args: argparse.Namespace) -> int:
    ppp = _ppp_from_args(args)
    config = _target_pipeline_config(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    first_hash = None
    last_hash = None
    with out_path.open("w", encoding="utf-8") as f:
        for record in _limit(
            iter_generated_target_records(args.split_manifest, ppp, root=args.root, config=config),
            args.max_records,
        ):
            row = record.to_manifest_dict()
            target_hash = row["target"]["target_hash"]
            first_hash = first_hash or target_hash
            last_hash = target_hash
            f.write(canonical_json(row) + "\n")
            count += 1
    summary = {
        "output_jsonl": str(out_path),
        "records_written": count,
        "ppp_hash": ppp.hash,
        "first_target_hash": first_hash,
        "last_target_hash": last_hash,
        "max_records": args.max_records,
    }
    _write_optional_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_write_surrogate_shards(args: argparse.Namespace) -> int:
    ppp = _ppp_from_args(args)
    config = _target_pipeline_config(args)
    records = list(
        _limit(
            iter_generated_target_records(args.split_manifest, ppp, root=args.root, config=config),
            args.max_records,
        )
    )
    summary = write_surrogate_training_shards(
        records,
        args.out_dir,
        ppp,
        config=SurrogateShardWriterConfig(shard_size=args.shard_size, run_id=args.run_id),
    ).to_dict()
    summary["records_input"] = len(records)
    _write_optional_summary(args.summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_train_surrogate(args: argparse.Namespace) -> int:
    from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig
    from robustsep_pkg.models.surrogate.training import (
        SurrogateLossConfig,
        SurrogateQualityGateThresholds,
        SurrogateTrainingConfig,
        train_surrogate,
    )

    result = train_surrogate(
        args.manifest,
        args.out_dir,
        training_config=SurrogateTrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            initial_checkpoint=args.initial_checkpoint,
        ),
        gate_thresholds=_thresholds_from_args(args),
        candidate_probe_config=CandidateProbeConfig(
            drift_sample_count=args.probe_drift_samples,
            root_seed=args.probe_root_seed,
            max_patches=args.probe_max_patches,
            batch_size=args.probe_batch_size,
        ),
        loss_config=SurrogateLossConfig(
            target_mode=args.loss_target_mode,
            hard_pixel_weight=args.hard_pixel_weight,
            hard_pixel_quantile=args.hard_pixel_quantile,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_eval_surrogate_gate(args: argparse.Namespace) -> int:
    import torch

    from robustsep_pkg.models.surrogate.data import SurrogateTrainingDataset
    from robustsep_pkg.models.surrogate.model import ForwardSurrogateCNN, SurrogateModelConfig
    from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig
    from robustsep_pkg.models.surrogate.training import evaluate_surrogate_quality

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = SurrogateModelConfig(**checkpoint.get("model_config", {}))
    model = ForwardSurrogateCNN(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = SurrogateTrainingDataset(args.manifest, model_config=model_config)
    quality = evaluate_surrogate_quality(
        model,
        dataset,
        device=device,
        thresholds=_thresholds_from_args(args),
        batch_size=args.batch_size,
        candidate_probe_config=CandidateProbeConfig(
            drift_sample_count=args.probe_drift_samples,
            root_seed=args.probe_root_seed,
            max_patches=args.probe_max_patches,
            batch_size=args.probe_batch_size,
        ),
    )
    payload = {"checkpoint": args.checkpoint, "manifest": args.manifest, "device": str(device), "quality": quality.to_dict()}
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_run_paper_eval(args: argparse.Namespace) -> int:
    from robustsep_pkg.eval.suite import PaperEvalConfig, run_paper_evaluation_suite

    ppp = _ppp_from_args(args)
    target_config = _target_pipeline_config(args)
    summary = run_paper_evaluation_suite(
        args.split_manifest,
        ppp,
        args.out,
        root=args.root,
        config=PaperEvalConfig(
            max_records=args.max_records,
            drift_samples=args.eval_drift_samples,
            root_seed=20260422,
            visual_examples=args.visual_examples,
            edge_threshold=args.edge_threshold,
            target_config=target_config,
        ),
        visual_npz=args.visual_npz,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def _parse_mapping(value: str, flag: str) -> tuple[str, str]:
    if "=" not in value:
        raise SystemExit(f"{flag} must be name=value, got {value!r}")
    key, val = value.split("=", 1)
    if not key or not val:
        raise SystemExit(f"{flag} must be name=value, got {value!r}")
    return key, val


def _parse_float_mapping(value: str, flag: str) -> tuple[str, float]:
    key, val = _parse_mapping(value, flag)
    return key, float(val)


def _ppp_from_args(args: argparse.Namespace) -> PPP:
    overrides = json.loads(args.ppp_overrides_json) if args.ppp_overrides_json else None
    if overrides is not None and not isinstance(overrides, dict):
        raise SystemExit("--ppp-overrides-json must decode to a JSON object")
    return PPP.from_base(args.ppp_base, overrides)


def _target_pipeline_config(args: argparse.Namespace) -> TargetGenerationPipelineConfig:
    return TargetGenerationPipelineConfig(
        target_solver_config=TargetSolverConfig(
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            stage1_step_size=args.stage1_step_size,
            stage2_step_size=args.stage2_step_size,
        ),
        drift_config=DriftConfig(),
        drift_samples_per_patch=args.drift_samples_per_patch,
        lambda_value=args.lambda_value,
        recompute_structure=args.recompute_structure,
        alpha_l_min=args.alpha_l_min,
        intent_raster_size=args.intent_raster_size,
        include_surrogate_examples=True,
    )


def _thresholds_from_args(args: argparse.Namespace) -> Any:
    from robustsep_pkg.models.surrogate.training import SurrogateQualityGateThresholds

    return SurrogateQualityGateThresholds(
        threshold_mean=args.threshold_mean,
        threshold_q90=args.threshold_q90,
        threshold_spearman=args.threshold_spearman,
        threshold_top1=args.threshold_top1,
        threshold_mean_regret=args.threshold_mean_regret,
        threshold_q90_regret=args.threshold_q90_regret,
    )


def _limit(records: Iterable[Any], max_records: int | None) -> Iterable[Any]:
    if max_records is not None and max_records < 0:
        raise SystemExit("--max-records must be non-negative or omitted")
    for idx, record in enumerate(records):
        if max_records is not None and idx >= max_records:
            break
        yield record


def _write_optional_summary(path: str | None, payload: dict[str, Any]) -> None:
    if path:
        write_json(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
