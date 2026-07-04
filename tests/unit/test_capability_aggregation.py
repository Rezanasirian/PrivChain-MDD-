"""Unit tests for capability-aware aggregation (Phase 4, H2)."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from privchain.federated.aggregation import ClientUpdate, capability_aware_aggregate


def _state(
    audio: float, video: float, text: float, shared: float
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        {
            "encoders.audio.w": torch.full((2,), audio),
            "encoders.video.w": torch.full((2,), video),
            "encoders.text.w": torch.full((2,), text),
            "classifier.w": torch.full((2,), shared),
        }
    )


def _weights(**groups: float) -> dict[str, float]:
    return dict(groups)


def test_modality_encoder_only_averaged_over_its_subgraph() -> None:
    global_state = _state(0.0, 0.0, 0.0, 0.0)
    full = ClientUpdate(0, (1, 1, 1), 10, _state(4.0, 4.0, 4.0, 4.0))
    text_only = ClientUpdate(1, (0, 0, 1), 10, _state(99.0, 99.0, 8.0, 8.0))

    weights = [
        _weights(audio=1.0, video=1.0, text=1.0, shared=1.0),
        _weights(text=1.0, shared=1.0),
    ]
    out = capability_aware_aggregate([full, text_only], global_state, weights)

    # Audio/video: only the full client is in the subgraph -> its value, untainted
    # by the text-only client's zero-imputed (99.0) encoder params.
    assert torch.allclose(out["encoders.audio.w"], torch.full((2,), 4.0))
    assert torch.allclose(out["encoders.video.w"], torch.full((2,), 4.0))
    # Text: both clients contribute -> average of 4 and 8.
    assert torch.allclose(out["encoders.text.w"], torch.full((2,), 6.0))
    # Shared: both clients contribute -> average of 4 and 8.
    assert torch.allclose(out["classifier.w"], torch.full((2,), 6.0))


def test_empty_subgraph_keeps_global_value() -> None:
    global_state = _state(1.0, 7.0, 1.0, 1.0)
    # No client declares video -> the video encoder must stay at the global value.
    a = ClientUpdate(0, (1, 0, 1), 5, _state(2.0, 123.0, 2.0, 2.0))
    b = ClientUpdate(1, (1, 0, 0), 5, _state(2.0, 123.0, 2.0, 2.0))
    weights = [_weights(audio=1.0, text=1.0, shared=1.0), _weights(audio=1.0, shared=1.0)]

    out = capability_aware_aggregate([a, b], global_state, weights)
    assert torch.allclose(out["encoders.video.w"], torch.full((2,), 7.0))


def test_reputation_weighting_biases_the_average() -> None:
    global_state = _state(0.0, 0.0, 0.0, 0.0)
    a = ClientUpdate(0, (1, 1, 1), 10, _state(0.0, 0.0, 0.0, 0.0))
    b = ClientUpdate(1, (1, 1, 1), 10, _state(0.0, 0.0, 0.0, 10.0))
    # 3:1 shared weighting -> (3*0 + 1*10)/4 = 2.5
    weights = [
        _weights(audio=1.0, video=1.0, text=1.0, shared=3.0),
        _weights(audio=1.0, video=1.0, text=1.0, shared=1.0),
    ]
    out = capability_aware_aggregate([a, b], global_state, weights)
    assert torch.allclose(out["classifier.w"], torch.full((2,), 2.5))


def test_errors() -> None:
    with pytest.raises(ValueError):
        capability_aware_aggregate([], _state(0, 0, 0, 0), [])
    update = ClientUpdate(0, (1, 1, 1), 1, _state(1, 1, 1, 1))
    with pytest.raises(ValueError):
        capability_aware_aggregate([update], _state(0, 0, 0, 0), [])
