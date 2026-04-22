from __future__ import annotations

from robustsep_pkg.core.seeding import uint64_hash


# ---------------------------------------------------------------------------
# Split names
# ---------------------------------------------------------------------------

SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST: str = "test"
SPLITS: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


def deterministic_split(
    n_shards: int,
    *,
    val_frac: float = 0.10,
    test_frac: float = 0.05,
    root_seed: int = 20260422,
    scope: str = "split",
) -> dict[str, list[int]]:
    """Assign *shard indices* (0 … n_shards-1) to train/val/test splits.

    The assignment is fully deterministic given ``(n_shards, val_frac,
    test_frac, root_seed, scope)``, so it can be replayed without storing
    a manifest field.  The split is performed by hashing each shard index
    to a stable pseudo-random key and ranking the keys, which avoids the
    dependence on list ordering that modulo-based splits have.

    Parameters
    ----------
    n_shards:
        Total number of shards to distribute.
    val_frac:
        Fraction of shards that go to validation (rounded to nearest int,
        minimum 0).
    test_frac:
        Fraction of shards that go to test (rounded to nearest int,
        minimum 0).
    root_seed:
        Master seed mixed into each per-shard hash.
    scope:
        Namespace string mixed into the hash to allow independent splits
        over different shard families from the same root seed.

    Returns
    -------
    dict mapping ``"train"``, ``"val"``, ``"test"`` to sorted lists of
    shard indices.
    """
    if n_shards <= 0:
        return {SPLIT_TRAIN: [], SPLIT_VAL: [], SPLIT_TEST: []}

    n_val = max(0, round(val_frac * n_shards))
    n_test = max(0, round(test_frac * n_shards))
    n_val = min(n_val, n_shards)
    n_test = min(n_test, n_shards - n_val)
    n_train = n_shards - n_val - n_test

    # Produce a stable permutation via per-shard hashes.
    keys = [uint64_hash(root_seed, scope, i) for i in range(n_shards)]
    order = sorted(range(n_shards), key=lambda i: keys[i])

    train_idx = sorted(order[:n_train])
    val_idx = sorted(order[n_train : n_train + n_val])
    test_idx = sorted(order[n_train + n_val :])

    return {SPLIT_TRAIN: train_idx, SPLIT_VAL: val_idx, SPLIT_TEST: test_idx}
