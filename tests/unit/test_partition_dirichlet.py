"""Unit tests for label-skewed client partitioning (Phase 2, ADR-0021).

IID sharding gives every client the corpus-wide class prevalence, which is the
easiest case for federated averaging and the least like real clinical
federation. These pin down that the Dirichlet split produces genuine skew while
still covering the data exactly once.
"""

from __future__ import annotations

import pytest

from privchain.config import FederationConfig, ModalityPattern, PartitionConfig
from privchain.federated.partition import (
    build_client_partitions,
    partition_indices,
    partition_indices_dirichlet,
)
from privchain.federated.simulation import _BestRoundTracker

_LABELS = [0] * 60 + [1] * 26  # roughly the real train split's balance


def _federation(mode: str, num_clients: int = 10) -> FederationConfig:
    return FederationConfig(
        num_clients=num_clients,
        num_rounds=5,
        clients_per_round=num_clients,
        local_epochs=1,
        modality_patterns=[
            ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.5),
            ModalityPattern(name="audio_only", capability=[1, 0, 0], fraction=0.5),
        ],
        partition=PartitionConfig(mode=mode, dirichlet_alpha=0.3),
    )


def _positive_rates(shards: list[list[int]], labels: list[int]) -> list[float]:
    return [sum(labels[i] for i in shard) / len(shard) for shard in shards]


def test_partition_covers_every_index_exactly_once() -> None:
    shards = partition_indices_dirichlet(_LABELS, 10, 0.3, seed=0)
    flat = [i for shard in shards for i in shard]
    assert sorted(flat) == list(range(len(_LABELS)))
    assert len(flat) == len(set(flat))  # no index handed to two clients


def test_no_client_is_left_empty() -> None:
    """A Dirichlet draw can starve a client; the population size is still honoured."""
    for seed in range(15):
        shards = partition_indices_dirichlet(_LABELS, 10, 0.05, seed=seed)
        assert len(shards) == 10
        assert all(shard for shard in shards)


def test_dirichlet_is_more_skewed_than_iid() -> None:
    iid = _positive_rates(partition_indices(len(_LABELS), 10, seed=0), _LABELS)
    skewed = _positive_rates(partition_indices_dirichlet(_LABELS, 10, 0.3, seed=0), _LABELS)
    # The point of the mode: client class mixes should spread out, not track the
    # corpus rate the way IID shards do.
    assert max(skewed) - min(skewed) > max(iid) - min(iid)


def test_larger_alpha_approaches_the_iid_split() -> None:
    tight = _positive_rates(partition_indices_dirichlet(_LABELS, 10, 0.1, seed=1), _LABELS)
    loose = _positive_rates(partition_indices_dirichlet(_LABELS, 10, 100.0, seed=1), _LABELS)
    assert max(loose) - min(loose) < max(tight) - min(tight)


def test_partition_is_reproducible() -> None:
    assert partition_indices_dirichlet(_LABELS, 8, 0.4, seed=7) == partition_indices_dirichlet(
        _LABELS, 8, 0.4, seed=7
    )


def test_build_partitions_uses_the_configured_mode() -> None:
    skewed = build_client_partitions(len(_LABELS), _federation("dirichlet"), 0, labels=_LABELS)
    iid = build_client_partitions(len(_LABELS), _federation("iid"), 0, labels=_LABELS)
    assert len(skewed) == len(iid) == 10

    skewed_rates = _positive_rates([p.indices for p in skewed], _LABELS)
    iid_rates = _positive_rates([p.indices for p in iid], _LABELS)
    assert max(skewed_rates) - min(skewed_rates) > max(iid_rates) - min(iid_rates)


def test_dirichlet_without_labels_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs per-item labels"):
        build_client_partitions(len(_LABELS), _federation("dirichlet"), 0)


@pytest.mark.parametrize(("alpha", "clients"), [(0.0, 5), (-1.0, 5)])
def test_invalid_alpha_is_rejected(alpha: float, clients: int) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        partition_indices_dirichlet(_LABELS, clients, alpha, seed=0)


def test_more_clients_than_items_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot partition"):
        partition_indices_dirichlet([0, 1, 0], 10, 0.5, seed=0)


# ── Early stopping on the selection split (ADR-0021) ─────────────────────────


def test_tracker_reports_improvements_and_stops_after_patience() -> None:
    tracker = _BestRoundTracker(patience=2)

    assert tracker.update(0.5, 1)  # first round is always an improvement
    assert tracker.best_round == 1
    assert not tracker.should_stop

    assert not tracker.update(0.4, 2)
    assert not tracker.should_stop  # one stale round is within patience
    assert not tracker.update(0.45, 3)
    assert tracker.should_stop  # two stale rounds exhausts it


def test_improvement_resets_the_patience_counter() -> None:
    tracker = _BestRoundTracker(patience=2)
    tracker.update(0.5, 1)
    tracker.update(0.4, 2)
    assert tracker.update(0.6, 3)
    assert tracker.best_round == 3
    assert not tracker.should_stop


def test_patience_none_never_stops() -> None:
    tracker = _BestRoundTracker(patience=None)
    tracker.update(0.9, 1)
    for round_num in range(2, 50):
        tracker.update(0.1, round_num)
    assert not tracker.should_stop
    assert tracker.best_round == 1
