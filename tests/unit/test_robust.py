"""Unit tests for the Byzantine outlier filter (Phase 5, H3)."""

from __future__ import annotations

from collections import OrderedDict

import torch

from privchain.federated.aggregation import ClientUpdate
from privchain.federated.robust import flag_byzantine_updates


def _update(client_id: int, shared: float) -> ClientUpdate:
    state = OrderedDict({"classifier.w": torch.full((4,), shared)})
    return ClientUpdate(client_id, (1, 1, 1), 10, state)


def _global() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict({"classifier.w": torch.zeros(4)})


def test_flags_gross_outlier() -> None:
    updates = [_update(0, 0.1), _update(1, 0.11), _update(2, 0.09), _update(3, 50.0)]
    flagged = flag_byzantine_updates(updates, _global(), z_thresh=2.5)
    assert flagged == {3}


def test_no_outlier_flags_nothing() -> None:
    updates = [_update(i, 0.1) for i in range(5)]
    assert flag_byzantine_updates(updates, _global(), z_thresh=2.5) == set()


def test_small_cohort_is_never_filtered() -> None:
    updates = [_update(0, 0.1), _update(1, 99.0)]
    assert flag_byzantine_updates(updates, _global(), z_thresh=2.5) == set()


def test_never_drops_half_or_more() -> None:
    # Two "normal" and two "extreme" — flagging both extremes would be half.
    updates = [_update(0, 0.1), _update(1, 0.1), _update(2, 80.0), _update(3, 90.0)]
    assert flag_byzantine_updates(updates, _global(), z_thresh=2.5) == set()
