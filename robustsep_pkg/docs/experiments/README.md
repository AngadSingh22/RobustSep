# Experiments

This directory contains experimental runs. Each experiment should be contained in its own folder named `exp_YYYYMMDD_description`.

## Structure

Each experiment folder must contain:
1.  `spec.yaml`: The full configuration specification for the experiment.
2.  `README.md`: A description of the experiment goals and hypothesis.
3.  `logs/`: Directory containing execution logs.
4.  `artifacts/`: Directory for output artifacts.

## Execution

Experiments are executed via the CLI using the `robustsep exec-spec` command (future implementation) or by running the runners in `models/`.

Example:
```bash
python -m robustsep_pkg.cli.robustsep train-surrogate --config experiments/exp_20230101_init/spec.yaml --run-dir experiments/exp_20230101_init/artifacts
```

## Registry
The `registry.csv` file tracks all experiments with columns: `id`, `date`, `description`, `status`.
