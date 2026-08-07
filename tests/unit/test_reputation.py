"""Unit tests for reputation weighting (Phase 4, H2)."""

from __future__ import annotations

from collections import OrderedDict

import torch

from privchain.config import ReputationConfig
from privchain.federated.aggregation import ClientUpdate
from privchain.federated.reputation import ReputationTracker


def _update(client_id: int, delta: list[float], num_samples: int = 10) -> ClientUpdate:
    # Global state is zeros in these tests, so the client's state == its delta.
    return ClientUpdate(
        client_id, (1, 1, 1), num_samples, OrderedDict({"classifier.w": torch.tensor(delta)})
    )


def test_outlier_client_is_down_weighted() -> None:
    global_state = OrderedDict({"classifier.w": torch.zeros(4)})
    updates = [
        _update(0, [1.0, 1.0, 1.0, 1.0]),
        _update(1, [1.0, 1.0, 1.0, 0.9]),
        _update(2, [-1.0, -1.0, -1.0, -1.0]),  # points against the consensus
    ]
    tracker = ReputationTracker(ReputationConfig(volume_weight=0.5, ema_decay=0.0))
    weights = tracker.compute_weights(updates, global_state, use_reputation=True)

    agree = weights[0]["shared"]
    outlier = weights[2]["shared"]
    assert agree > outlier
    # The reputation snapshot is ledger-ready and records every client.
    snap = tracker.snapshot()
    assert snap[0]["shared"] > snap[2]["shared"]


def test_use_reputation_false_is_pure_volume() -> None:
    global_state = OrderedDict({"classifier.w": torch.zeros(4)})
    updates = [
        _update(0, [1.0, 0.0, 0.0, 0.0], num_samples=7),
        _update(1, [-5.0, 0.0, 0.0, 0.0], num_samples=3),
    ]
    tracker = ReputationTracker(ReputationConfig())
    weights = tracker.compute_weights(updates, global_state, use_reputation=False)
    assert weights[0]["shared"] == 7.0
    assert weights[1]["shared"] == 3.0


def test_min_reputation_floor_keeps_weights_positive() -> None:
    global_state = OrderedDict({"classifier.w": torch.zeros(4)})
    updates = [
        _update(0, [1.0, 1.0, 1.0, 1.0]),
        _update(1, [1.0, 1.0, 1.0, 1.0]),
        _update(2, [-1.0, -1.0, -1.0, -1.0]),
    ]
    tracker = ReputationTracker(ReputationConfig(volume_weight=0.0, min_reputation=0.05))
    weights = tracker.compute_weights(updates, global_state, use_reputation=True)
    # Even the fully-inconsistent client keeps a positive (floored) weight.
    assert weights[2]["shared"] > 0.0
