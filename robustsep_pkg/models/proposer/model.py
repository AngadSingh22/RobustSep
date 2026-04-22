from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("ConditionalVAEProposer requires PyTorch. Install torch to use this module.") from exc

from robustsep_pkg.models.surrogate.model import FiLMHead, MLP


@dataclass(frozen=True)
class ProposerModelConfig:
    num_base_families: int = 8
    ppp_numeric_dim: int = 17
    ppp_override_mask_dim: int = 17
    base_family_embedding_dim: int = 32
    structure_embedding_dim: int = 8
    ppp_embedding_dim: int = 128
    structure_condition_dim: int = 32
    intent_embedding_dim: int = 128
    intent_raster_size: int = 4
    film_hidden_dim: int = 128
    group_count: int = 8
    latent_dim: int = 64

    @property
    def intent_input_dim(self) -> int:
        return 3 + self.intent_raster_size * self.intent_raster_size * 3

    @property
    def condition_dim(self) -> int:
        return self.ppp_embedding_dim + self.structure_condition_dim + self.intent_embedding_dim + 1

    @property
    def input_channels(self) -> int:
        return 10


@dataclass(frozen=True)
class ProposerOutput:
    cmykogv: torch.Tensor
    latent_mean: torch.Tensor
    latent_logvar: torch.Tensor


class ProposerConditioner(nn.Module):
    def __init__(self, config: ProposerModelConfig = ProposerModelConfig()) -> None:
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
    ) -> torch.Tensor:
        intent_flat = intent_raster.reshape(intent_raster.shape[0], -1)
        ppp_input = torch.cat(
            [self.base_family_embedding(base_family_index.long()), ppp_numeric.float(), ppp_override_mask.float()],
            dim=-1,
        )
        return torch.cat(
            [
                self.ppp_mlp(ppp_input),
                self.structure_mlp(self.structure_embedding(structure_index.long())),
                self.intent_mlp(torch.cat([intent_weights.float(), intent_flat.float()], dim=-1)),
                lambda_value.float().reshape(-1, 1),
            ],
            dim=-1,
        )


class ProposerFiLMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, cond_dim: int, config: ProposerModelConfig) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(config.group_count, out_channels), num_channels=out_channels)
        self.film = FiLMHead(cond_dim, out_channels, config.film_hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        scale, shift = self.film(cond)
        return self.activation(x * (1.0 + scale) + shift)


class ProposerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, config: ProposerModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=min(config.group_count, out_channels), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalVAEProposer(nn.Module):
    """FiLM-conditioned VAE proposer producing CMYKOGV candidate patches."""

    def __init__(self, config: ProposerModelConfig = ProposerModelConfig()) -> None:
        super().__init__()
        self.config = config
        cond_dim = config.condition_dim
        self.conditioner = ProposerConditioner(config)

        self.e0 = ProposerBlock(config.input_channels, 32, stride=1, config=config)
        self.e1 = ProposerFiLMBlock(32, 32, stride=2, cond_dim=cond_dim, config=config)
        self.e2 = ProposerFiLMBlock(32, 64, stride=1, cond_dim=cond_dim, config=config)
        self.e3 = ProposerFiLMBlock(64, 64, stride=2, cond_dim=cond_dim, config=config)
        self.e4 = ProposerFiLMBlock(64, 96, stride=1, cond_dim=cond_dim, config=config)
        self.latent_mean = nn.Linear(96, config.latent_dim)
        self.latent_logvar = nn.Linear(96, config.latent_dim)

        self.d0 = nn.Linear(config.latent_dim, 96 * 4 * 4)
        self.d1 = ProposerFiLMBlock(96, 96, stride=1, cond_dim=cond_dim, config=config)
        self.d2 = ProposerFiLMBlock(96, 64, stride=1, cond_dim=cond_dim, config=config)
        self.d3 = ProposerFiLMBlock(64, 64, stride=1, cond_dim=cond_dim, config=config)
        self.d4 = ProposerFiLMBlock(64, 32, stride=1, cond_dim=cond_dim, config=config)
        self.d5 = ProposerFiLMBlock(32, 32, stride=1, cond_dim=cond_dim, config=config)
        self.head = nn.Conv2d(32, 7, kernel_size=1)

    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.e0(x)
        h = self.e1(h, cond)
        h = self.e2(h, cond)
        h = self.e3(h, cond)
        h = self.e4(h, cond)
        pooled = h.mean(dim=(2, 3))
        return self.latent_mean(pooled), self.latent_logvar(pooled)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.d0(z).reshape(z.shape[0], 96, 4, 4)
        h = self.d1(h, cond)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.d2(h, cond)
        h = self.d3(h, cond)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.d4(h, cond)
        h = self.d5(h, cond)
        return torch.sigmoid(self.head(h))

    def forward(
        self,
        rgb_patch: torch.Tensor,
        alpha: torch.Tensor,
        *,
        base_family_index: torch.Tensor,
        ppp_numeric: torch.Tensor,
        ppp_override_mask: torch.Tensor,
        structure_index: torch.Tensor,
        intent_weights: torch.Tensor,
        intent_raster: torch.Tensor,
        lambda_value: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> ProposerOutput:
        cond = self.conditioner(
            base_family_index=base_family_index,
            ppp_numeric=ppp_numeric,
            ppp_override_mask=ppp_override_mask,
            structure_index=structure_index,
            intent_weights=intent_weights,
            intent_raster=intent_raster,
            lambda_value=lambda_value,
        )
        x = build_proposer_input(rgb_patch, alpha, intent_weights, intent_raster)
        mean, logvar = self.encode(x, cond)
        latent = z if z is not None else self.reparameterize(mean, logvar)
        return ProposerOutput(cmykogv=self.decode(latent, cond), latent_mean=mean, latent_logvar=logvar)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std


def build_proposer_input(rgb_patch: torch.Tensor, alpha: torch.Tensor, intent_weights: torch.Tensor, intent_raster: torch.Tensor) -> torch.Tensor:
    rgb = _to_nchw(rgb_patch, channels=3)
    if alpha.ndim == 3:
        alpha_ch = alpha[:, None, :, :].float()
    elif alpha.ndim == 4 and alpha.shape[1] == 1:
        alpha_ch = alpha.float()
    elif alpha.ndim == 4 and alpha.shape[-1] == 1:
        alpha_ch = alpha.permute(0, 3, 1, 2).contiguous().float()
    else:
        raise ValueError(f"alpha must be (N,H,W), (N,1,H,W), or (N,H,W,1), got {tuple(alpha.shape)}")
    weights = intent_weights.float()[:, :, None, None].expand(-1, -1, 16, 16)
    raster = _to_nchw(intent_raster, channels=3)
    raster_up = F.interpolate(raster, size=(16, 16), mode="nearest")
    return torch.cat([rgb, alpha_ch, weights, raster_up], dim=1)


def _to_nchw(x: torch.Tensor, *, channels: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 tensor, got {tuple(x.shape)}")
    if x.shape[1] == channels:
        return x.float()
    if x.shape[-1] == channels:
        return x.permute(0, 3, 1, 2).contiguous().float()
    raise ValueError(f"expected channel axis length {channels}, got {tuple(x.shape)}")
