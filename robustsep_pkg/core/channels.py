from __future__ import annotations

from dataclasses import dataclass

CHANNELS_CMYKOGV: tuple[str, ...] = ("C", "M", "Y", "K", "O", "G", "V")
CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CHANNELS_CMYKOGV)}
CMYK_INDICES: tuple[int, ...] = (0, 1, 2, 3)
OGV_INDICES: tuple[int, ...] = (4, 5, 6)


@dataclass(frozen=True)
class ChannelLayout:
    names: tuple[str, ...] = CHANNELS_CMYKOGV

    @property
    def count(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return CHANNEL_INDEX[name]


def ensure_cmykogv_last_axis(values_shape: tuple[int, ...]) -> None:
    if not values_shape or values_shape[-1] != len(CHANNELS_CMYKOGV):
        raise ValueError(f"expected last axis of size 7 for CMYKOGV, got shape {values_shape}")
