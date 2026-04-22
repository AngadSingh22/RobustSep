"""Forward surrogate model modules."""

try:
    from robustsep_pkg.models.surrogate.probe import CandidateProbeConfig, CandidateProbeMetrics, evaluate_candidate_probe
except ImportError:  # pragma: no cover
    __all__: list[str] = []
else:
    __all__ = [
        "CandidateProbeConfig",
        "CandidateProbeMetrics",
        "evaluate_candidate_probe",
    ]
