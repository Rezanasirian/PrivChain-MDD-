"""End-to-end Phase 1 training smoke test.

Definition of Done for Phase 1: the model trains/evaluates on mock data and
reports F1 and ROC-AUC. This runs a couple of epochs over the mock dataset and
checks that metrics are produced and run artifacts (metrics.jsonl, checkpoint)
are written.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

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
    TrainConfig,
    VideoConfig,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.loaders import build_train_val_loaders
from privchain.training.objective import DepressionObjective, positive_class_weight
from privchain.training.trainer import CentralizedTrainer


def _data_config() -> DataConfig:
    return DataConfig(
        num_sessions=24,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=12, min_frames=10, max_frames=16),
        video=VideoConfig(n_features=9, min_frames=6, max_frames=10),
        text=TextConfig(embed_dim=16, min_tokens=5, max_tokens=9),
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(type="gru", hidden_dim=8, out_dim=8, dropout=0.0),
        fusion=FusionConfig(hidden_dim=16),
        head=HeadConfig(hidden_dim=8),
        use_phq_regression=True,
        phq_loss_weight=0.1,
    )


def _train_config() -> TrainConfig:
    return TrainConfig(batch_size=4, epochs=2, learning_rate=0.01, val_fraction=0.25)


@pytest.mark.integration
def test_centralized_training_reports_f1_and_auc(tmp_path: Path) -> None:
    data_cfg, model_cfg, train_cfg = _data_config(), _model_config(), _train_config()
    seed_everything(42)

    train_loader, val_loader = build_train_val_loaders(data_cfg, train_cfg, seed=42)
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    trainer = CentralizedTrainer(
        model,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history = trainer.fit(train_loader, val_loader, epochs=train_cfg.epochs, run_dir=run_dir)

    # Metrics are produced for every epoch, including F1 and ROC-AUC.
    assert len(history) == train_cfg.epochs
    final = history[-1]
    assert "val_f1" in final
    assert "val_roc_auc" in final
    assert isinstance(final["val_f1"], float)
    assert 0.0 <= final["val_f1"] <= 1.0

    # Artifacts written.
    assert (run_dir / "best_model.pt").is_file()
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == train_cfg.epochs
    record = json.loads(lines[0])
    assert "val_roc_auc" in record


@pytest.mark.integration
def test_early_stopping_ends_before_the_epoch_cap(tmp_path: Path) -> None:
    """With patience=1, a run stops as soon as the metric fails to improve."""
    data_cfg, model_cfg = _data_config(), _model_config()
    train_cfg = TrainConfig(batch_size=4, epochs=50, learning_rate=0.01, early_stopping_patience=1)
    seed_everything(42)

    train_loader, val_loader = build_train_val_loaders(data_cfg, train_cfg, seed=42)
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    trainer = CentralizedTrainer(
        model,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=train_cfg.epochs,
        run_dir=run_dir,
        early_stopping_patience=train_cfg.early_stopping_patience,
    )

    assert 0 < len(history) < train_cfg.epochs
    # metrics.jsonl records exactly the epochs that actually ran.
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(history)


@pytest.mark.integration
def test_explicit_val_dataset_is_used_verbatim() -> None:
    """A supplied validation set is used whole, ignoring ``val_fraction``."""
    data_cfg, train_cfg = _data_config(), _train_config()
    seed_everything(3)

    train_set = MockDaicWozDataset(data_cfg, seed=3)
    val_set = MockDaicWozDataset(_data_config().model_copy(update={"num_sessions": 10}), seed=99)

    train_loader, val_loader = build_train_val_loaders(
        data_cfg, train_cfg, seed=3, dataset=train_set, val_dataset=val_set
    )

    # No 25% carve-out: train keeps all 24 sessions and val is exactly the 10 given.
    assert sum(b["label"].numel() for b in train_loader) == 24
    assert sum(b["label"].numel() for b in val_loader) == 10


def test_positive_class_weight_counts_neg_over_pos() -> None:
    """The BCE positive weight is measured from the data as neg/pos."""
    data_cfg, train_cfg = _data_config(), _train_config()
    seed_everything(11)
    train_loader, _ = build_train_val_loaders(data_cfg, train_cfg, seed=11)

    labels = torch.cat([b["label"] for b in train_loader])
    positives = int(labels.sum().item())
    negatives = int(labels.numel()) - positives

    weight = positive_class_weight(train_loader)
    if positives == 0 or negatives == 0:
        assert weight is None
    else:
        assert weight == pytest.approx(negatives / positives)


def test_positive_class_weight_is_none_for_a_single_class() -> None:
    """Weighting is undefined when a class is absent, so it is disabled."""

    class _OneClassLoader:
        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            yield {"label": torch.zeros(4, dtype=torch.long)}

    assert positive_class_weight(cast("Any", _OneClassLoader())) is None


def test_weighted_objective_penalizes_missed_positives_more() -> None:
    """pos_weight makes a false negative cost more than a false positive."""
    batch = cast(
        "Any",
        {
            "label": torch.tensor([1.0, 0.0]),
            "phq8_score": torch.tensor([12, 2]),
        },
    )
    confident_wrong_on_positive = {"logit": torch.tensor([-3.0, -3.0])}
    confident_wrong_on_negative = {"logit": torch.tensor([3.0, 3.0])}

    unweighted = DepressionObjective(24, 0.0)
    weighted = DepressionObjective(24, 0.0, pos_weight=5.0)

    # Symmetric without weighting...
    assert float(unweighted(confident_wrong_on_positive, batch)) == pytest.approx(
        float(unweighted(confident_wrong_on_negative, batch)), rel=1e-5
    )
    # ...and asymmetric with it: missing the positive now hurts more.
    assert float(weighted(confident_wrong_on_positive, batch)) > float(
        weighted(confident_wrong_on_negative, batch)
    )


@pytest.mark.integration
def test_evaluate_returns_metric_keys(tmp_path: Path) -> None:
    data_cfg, model_cfg, train_cfg = _data_config(), _model_config(), _train_config()
    seed_everything(7)
    train_loader, val_loader = build_train_val_loaders(data_cfg, train_cfg, seed=7)
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    trainer = CentralizedTrainer(
        model,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
    )
    metrics = trainer.evaluate(val_loader)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "loss"):
        assert key in metrics
