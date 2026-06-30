"""Unit tests for FedAvg aggregation (Phase 2)."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from privchain.federated.aggregation import fedavg


def _state(value: float) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "w": torch.full((2, 2), value),
            "b": torch.tensor([value, -value]),
        }
    )


def test_equal_weights_is_simple_mean() -> None:
    out = fedavg([_state(2.0), _state(4.0)], [1.0, 1.0])
    assert torch.allclose(out["w"], torch.full((2, 2), 3.0))
    assert torch.allclose(out["b"], torch.tensor([3.0, -3.0]))


def test_weighted_average() -> None:
    # 3:1 weighting of values 2 and 6 -> (3*2 + 1*6)/4 = 3.0
    out = fedavg([_state(2.0), _state(6.0)], [3.0, 1.0])
    assert torch.allclose(out["w"], torch.full((2, 2), 3.0))


def test_single_client_returns_its_state() -> None:
    out = fedavg([_state(5.0)], [10.0])
    assert torch.allclose(out["w"], torch.full((2, 2), 5.0))


def test_errors() -> None:
    with pytest.raises(ValueError):
        fedavg([], [])
    with pytest.raises(ValueError):
        fedavg([_state(1.0)], [1.0, 2.0])
    with pytest.raises(ValueError):
        fedavg([_state(1.0)], [0.0])
