"""Unit tests for capability subgraph grouping (Phase 4, H2)."""

from __future__ import annotations

from privchain.federated.capability import (
    SHARED_GROUP,
    has_modality,
    is_missing_any,
    modality_subgraphs,
    param_group,
)


def test_param_group_routes_encoders_and_shared() -> None:
    assert param_group("encoders.audio.gru.weight_ih_l0") == "audio"
    assert param_group("encoders.video.proj.bias") == "video"
    assert param_group("encoders.text.gru.weight") == "text"
    assert param_group("fusion.net.0.weight") == SHARED_GROUP
    assert param_group("classifier.0.weight") == SHARED_GROUP
    assert param_group("regressor.weight") == SHARED_GROUP


def test_has_modality() -> None:
    cap = (1, 0, 1)  # audio + text
    assert has_modality(cap, "audio")
    assert not has_modality(cap, "video")
    assert has_modality(cap, "text")


def test_is_missing_any() -> None:
    assert not is_missing_any((1, 1, 1))
    assert is_missing_any((1, 0, 1))
    assert is_missing_any((0, 0, 1))


def test_modality_subgraphs_group_by_declared_capability() -> None:
    caps = [(1, 1, 1), (1, 0, 1), (1, 0, 0), (0, 0, 1)]
    subgraphs = modality_subgraphs(caps)
    assert subgraphs["audio"] == [0, 1, 2]
    assert subgraphs["video"] == [0]
    assert subgraphs["text"] == [0, 1, 3]
