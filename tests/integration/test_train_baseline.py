"""End-to-end Phase 1 training smoke test.

Definition of Done for Phase 1: the model trains/evaluates on mock data and
reports F1 and ROC-AUC. This runs a couple of epochs over the mock dataset and
checks that metrics are produced and run artifacts (metrics.jsonl, checkpoint)
are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.loaders import build_train_val_loaders
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
