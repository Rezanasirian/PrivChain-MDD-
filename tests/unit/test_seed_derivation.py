"""Unit tests for collision-free seed derivation (Phase 1).

A repetition harness that reuses a seed silently reports too small a spread, so
the property under test is distinctness, not any particular value.
"""

from __future__ import annotations

import pytest

from privchain.seeding import derive_seed


def test_adjacent_seeds_do_not_share_fold_seeds() -> None:
    """The failure `seed + fold` had: seeds 42 and 45 shared three of five folds."""
    a = {derive_seed(42, fold) for fold in range(5)}
    b = {derive_seed(45, fold) for fold in range(5)}

    assert not (a & b)


def test_every_seed_fold_pair_is_distinct() -> None:
    pairs = [derive_seed(seed, fold) for seed in range(200) for fold in range(10)]

    assert len(set(pairs)) == len(pairs)


def test_derivation_is_stable_across_calls() -> None:
    """Reproducibility is the point; a per-process salt would defeat it."""
    assert derive_seed(42, 3) == derive_seed(42, 3)


def test_result_is_a_valid_numpy_seed() -> None:
    values = [derive_seed(seed, fold) for seed in (0, 7, 2024, -1) for fold in range(5)]

    assert all(0 <= value < 2**31 for value in values)


def test_argument_order_matters() -> None:
    assert derive_seed(1, 2) != derive_seed(2, 1)


def test_no_parts_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one part"):
        derive_seed()
