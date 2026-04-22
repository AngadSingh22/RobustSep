from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("Proposer losses require PyTorch. Install torch to use this module.") from exc


@dataclass(frozen=True)
class ProposerLossWeights:
    ink: float = 1.0
    kl: float = 1e-3
    appearance: float = 0.0
    lambda_monotonicity: float = 0.1


def proposer_vae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_logvar: torch.Tensor,
    *,
    weights: ProposerLossWeights = ProposerLossWeights(),
    appearance_loss: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    ink = F.smooth_l1_loss(prediction, target)
    kl = -0.5 * torch.mean(1.0 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
    app = appearance_loss if appearance_loss is not None else prediction.new_tensor(0.0)
    total = weights.ink * ink + weights.kl * kl + weights.appearance * app
    return {"total": total, "ink": ink, "kl": kl, "appearance": app}


def lambda_monotonicity_hinge(outputs_by_lambda: Sequence[torch.Tensor], margin: float = 0.0) -> torch.Tensor:
    """Penalize decreases in patch-mean OGV across increasing lambda outputs."""
    if len(outputs_by_lambda) < 2:
        raise ValueError("lambda_monotonicity_hinge requires at least two outputs")
    means = [out[:, 4:7].mean(dim=(1, 2, 3)) for out in outputs_by_lambda]
    penalties = [F.relu(prev + margin - nxt) for prev, nxt in zip(means, means[1:])]
    return torch.stack(penalties, dim=0).mean()
