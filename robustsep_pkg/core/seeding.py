from __future__ import annotations

import hashlib
from typing import Any


UINT64_MASK = (1 << 64) - 1


def uint64_hash(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        encoded = str(part).encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big", signed=False))
        h.update(encoded)
    return int.from_bytes(h.digest(), "big") & UINT64_MASK


def derive_seed(
    root_seed: int,
    input_hash: str,
    ppp_hash: str,
    scope: str,
    patch_coord: tuple[int, int] | str,
    candidate_index: int | None = None,
    drift_index: int | None = None,
) -> int:
    return uint64_hash(root_seed, input_hash, ppp_hash, scope, patch_coord, candidate_index, drift_index)
