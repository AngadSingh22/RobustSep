from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch installed
    raise ImportError("ForwardSurrogateCNN requires PyTorch. Install torch to use robustsep_pkg.models.surrogate.model.") from exc


@dataclass(frozen=True)
class SurrogateModelConfig:
    num_base_families: int = 8
    ppp_numeric_dim: int = 17
    ppp_override_mask_dim: int = 17
    base_family_embedding_dim: int = 32
    structure_embedding_dim: int = 8
    ppp_embedding_dim: int = 128
    structure_condition_dim: int = 32
    intent_embedding_dim: int = 128
    intent_raster_size: int = 4
    drift_dim: int = 56
    film_hidden_dim: int = 128
    group_count: int = 8

    @property
    def intent_input_dim(self) -> int:
        return 3 + self.intent_raster_size * self.intent_raster_size * 3

    @property
    def base_condition_dim(self) -> int:
        return self.ppp_embedding_dim + self.structure_condition_dim + self.intent_embedding_dim + 1

    @property
    def surrogate_condition_dim(self) -> int:
        return self.base_condition_dim + self.drift_dim


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMHead(nn.Module):
    def __init__(self, cond_dim: int, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * channels),
        )

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.net(cond).chunk(2, dim=-1)
        return scale[:, :, None, None], shift[:, :, None, None]


class ConvFiLMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int, cond_dim: int, config: SurrogateModelConfig) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=min(config.group_count, out_channels), num_channels=out_channels)
        self.film = FiLMHead(cond_dim, out_channels, config.film_hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        scale, shift = self.film(cond)
        return self.activation(x * (1.0 + scale) + shift)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dilation: int, config: SurrogateModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(num_groups=min(config.group_count, out_channels), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurrogateConditioner(nn.Module):
    """Build the 345-D surrogate conditioning vector specified in the docs."""

    def __init__(self, config: SurrogateModelConfig = SurrogateModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.base_family_embedding = nn.Embedding(config.num_base_families, config.base_family_embedding_dim)
        self.structure_embedding = nn.Embedding(3, config.structure_embedding_dim)
        self.ppp_mlp = MLP(
            config.base_family_embedding_dim + config.ppp_numeric_dim + config.ppp_override_mask_dim,
            config.film_hidden_dim,
            config.ppp_embedding_dim,
        )
        self.structure_mlp = MLP(config.structure_embedding_dim, config.film_hidden_dim, config.structure_condition_dim)
        self.intent_mlp = MLP(config.intent_input_dim, config.film_hidden_dim, config.intent_embedding_dim)

    def forward(
        self,
        *,
        base_family_index: torch.Tensor,
        ppp_numeric: torch.Tensor,
        ppp_override_mask: torch.Tensor,
        structure_index: torch.Tensor,
        intent_weights: torch.Tensor,
        intent_raster: torch.Tensor,
        lambda_value: torch.Tensor,
        drift_vector: torch.Tensor,
    ) -> torch.Tensor:
        if intent_raster.ndim == 4:
            intent_flat = intent_raster.reshape(intent_raster.shape[0], -1)
        else:
            raise ValueError(f"intent_raster must have shape (N,H,W,3), got {tuple(intent_raster.shape)}")
        ppp_input = torch.cat(
            [
                self.base_family_embedding(base_family_index.long()),
                ppp_numeric.float(),
                ppp_override_mask.float(),
            ],
            dim=-1,
        )
        lambda_col = lambda_value.float().reshape(-1, 1)
        return torch.cat(
            [
                self.ppp_mlp(ppp_input),
                self.structure_mlp(self.structure_embedding(structure_index.long())),
                self.intent_mlp(torch.cat([intent_weights.float(), intent_flat.float()], dim=-1)),
                lambda_col,
                drift_vector.float(),
            ],
            dim=-1,
        )


class ForwardSurrogateCNN(nn.Module):
    """FiLM-conditioned Micro-CNN from CMYKOGV context to center Lab prediction."""

    def __init__(self, config: SurrogateModelConfig = SurrogateModelConfig()) -> None:
        super().__init__()
        self.config = config
        cond_dim = config.surrogate_condition_dim
        self.conditioner = SurrogateConditioner(config)
        self.block0 = ConvBlock(7, 32, dilation=1, config=config)
        self.block1 = ConvFiLMBlock(32, 32, dilation=1, cond_dim=cond_dim, config=config)
        self.block2 = ConvFiLMBlock(32, 64, dilation=2, cond_dim=cond_dim, config=config)
        self.block3 = ConvFiLMBlock(64, 64, dilation=2, cond_dim=cond_dim, config=config)
        self.block4 = ConvFiLMBlock(64, 96, dilation=4, cond_dim=cond_dim, config=config)
        self.block5 = ConvFiLMBlock(96, 96, dilation=4, cond_dim=cond_dim, config=config)
        self.block6 = ConvFiLMBlock(96, 64, dilation=1, cond_dim=cond_dim, config=config)
        self.block7 = ConvFiLMBlock(64, 32, dilation=1, cond_dim=cond_dim, config=config)
        self.head = nn.Conv2d(32, 3, kernel_size=1)

    def forward(
        self,
        cmykogv_context: torch.Tensor,
        *,
        base_family_index: torch.Tensor,
        ppp_numeric: torch.Tensor,
        ppp_override_mask: torch.Tensor,
        structure_index: torch.Tensor,
        intent_weights: torch.Tensor,
        intent_raster: torch.Tensor,
        lambda_value: torch.Tensor,
        drift_vector: torch.Tensor,
    ) -> torch.Tensor:
        x = _to_nchw(cmykogv_context)
        cond = self.conditioner(
            base_family_index=base_family_index,
            ppp_numeric=ppp_numeric,
            ppp_override_mask=ppp_override_mask,
            structure_index=structure_index,
            intent_weights=intent_weights,
            intent_raster=intent_raster,
            lambda_value=lambda_value,
            drift_vector=drift_vector,
        )
        x = self.block0(x)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.block3(x, cond)
        x = self.block4(x, cond)
        x = self.block5(x, cond)
        x = self.block6(x, cond)
        x = self.block7(x, cond)
        lab_full = self.head(x)
        start = (lab_full.shape[-1] - 16) // 2
        return lab_full[:, :, start : start + 16, start : start + 16]


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"cmykogv_context must be rank 4, got {tuple(x.shape)}")
    if x.shape[1] == 7:
        return x.float()
    if x.shape[-1] == 7:
        return x.permute(0, 3, 1, 2).contiguous().float()
    raise ValueError(f"expected CMYKOGV channel axis of length 7, got {tuple(x.shape)}")
