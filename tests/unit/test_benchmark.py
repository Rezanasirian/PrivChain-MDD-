"""Unit tests for the evaluation benchmark core (Phase 7, H5)."""

from __future__ import annotations

import math

import pytest
import torch

from privchain.config import (
    AudioConfig,
    DataConfig,
    EncoderConfig,
    FusionConfig,
    HeadConfig,
    ModelConfig,
    TextConfig,
    VideoConfig,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset
from privchain.eval.benchmark import (
    aggregate_metrics,
    held_out_split,
    k_fold_indices,
    measure_inference_latency,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel


def test_k_fold_indices_partition_covers_and_is_disjoint() -> None:
    splits = k_fold_indices(20, 5, seed=0)
    assert len(splits) == 5
    all_test: list[int] = []
    for train, test in splits:
        assert set(train).isdisjoint(test)
        assert sorted(train + test) == list(range(20))  # train ∪ test = everything
        all_test.extend(test)
    assert sorted(all_test) == list(range(20))  # each item tested exactly once


def test_k_fold_indices_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        k_fold_indices(5, 1, seed=0)
    with pytest.raises(ValueError):
        k_fold_indices(5, 6, seed=0)


def test_held_out_split_is_disjoint_and_covers() -> None:
    dev, held = held_out_split(50, 0.2, seed=1)
    assert len(held) == 10
    assert set(dev).isdisjoint(held)
    assert sorted(dev + held) == list(range(50))
    with pytest.raises(ValueError):
        held_out_split(3, 0.01, seed=1)  # empty held-out


def test_aggregate_metrics_is_nan_aware() -> None:
    folds = [
        {"f1": 0.4, "roc_auc": 0.6},
        {"f1": 0.6, "roc_auc": float("nan")},  # undefined AUC ignored
        {"f1": 0.5, "roc_auc": 0.8},
    ]
    agg = aggregate_metrics(folds)
    assert agg["num_folds"] == 3.0
    assert agg["f1_mean"] == pytest.approx(0.5)
    assert agg["roc_auc_mean"] == pytest.approx(0.7)  # mean of 0.6 and 0.8 only
    with pytest.raises(ValueError):
        aggregate_metrics([])


def _tiny_model() -> tuple[MultimodalDepressionModel, MockDaicWozDataset]:
    data_cfg = DataConfig(
        num_sessions=8,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=8, min_frames=6, max_frames=10),
        video=VideoConfig(n_features=6, min_frames=4, max_frames=8),
        text=TextConfig(embed_dim=8, min_tokens=3, max_tokens=6),
    )
    model_cfg = ModelConfig(
        encoder=EncoderConfig(type="gru", hidden_dim=6, out_dim=6),
        fusion=FusionConfig(hidden_dim=8),
        head=HeadConfig(hidden_dim=6),
    )
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    return model, MockDaicWozDataset(data_cfg, seed=0)


def test_measure_inference_latency_shapes_and_positivity() -> None:
    model, dataset = _tiny_model()
    records = measure_inference_latency(
        model, dataset, batch_sizes=[1, 4], repeats=3, device=torch.device("cpu")
    )
    assert [r["batch_size"] for r in records] == [1, 4]
    for record in records:
        assert record["ms_per_batch"] > 0.0
        assert record["ms_per_sample"] == pytest.approx(
            record["ms_per_batch"] / record["batch_size"]
        )
    assert not math.isnan(records[0]["ms_per_sample"])


def test_measure_inference_latency_errors() -> None:
    model, dataset = _tiny_model()
    with pytest.raises(ValueError):
        measure_inference_latency(
            model, dataset, batch_sizes=[1], repeats=0, device=torch.device("cpu")
        )
    with pytest.raises(ValueError):
        measure_inference_latency(
            model, dataset, batch_sizes=[999], repeats=1, device=torch.device("cpu")
        )
