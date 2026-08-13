"""Unit tests for the multimodal baseline model and fusion (Phase 1)."""

from __future__ import annotations

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
from privchain.data.mock_daic_woz import MockDaicWozDataset, collate_fn
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.fusion.multimodal_fusion import ConcatFusion


def _data_config() -> DataConfig:
    return DataConfig(
        num_sessions=6,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=12, min_frames=10, max_frames=14),
        video=VideoConfig(n_features=9, min_frames=6, max_frames=8),
        text=TextConfig(embed_dim=16, min_tokens=5, max_tokens=7),
    )


def _model_config(use_reg: bool = True) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(type="gru", hidden_dim=8, out_dim=8),
        fusion=FusionConfig(hidden_dim=16),
        head=HeadConfig(hidden_dim=8),
        use_phq_regression=use_reg,
    )


def test_concat_fusion_shape_and_presence_masking() -> None:
    fusion = ConcatFusion({"audio": 4, "video": 4, "text": 4}, hidden_dim=10)
    emb = {m: torch.randn(3, 4) for m in ("audio", "video", "text")}
    out = fusion(emb)
    assert out.shape == (3, 10)

    presence = {
        "audio": torch.ones(3),
        "video": torch.zeros(3),  # video absent for all
        "text": torch.ones(3),
    }
    out_masked = fusion(emb, presence)
    assert out_masked.shape == (3, 10)


def test_model_forward_shapes_with_regression() -> None:
    data_cfg = _data_config()
    dataset = MockDaicWozDataset(data_cfg, seed=1)
    batch = collate_fn([dataset[i] for i in range(4)])

    model = MultimodalDepressionModel(modality_input_dims(data_cfg), _model_config(True))
    out = model(batch)

    assert out["logit"].shape == (4,)
    assert "phq_pred" in out
    assert out["phq_pred"].shape == (4,)


def test_model_without_regression_head() -> None:
    data_cfg = _data_config()
    dataset = MockDaicWozDataset(data_cfg, seed=1)
    batch = collate_fn([dataset[i] for i in range(3)])

    model = MultimodalDepressionModel(modality_input_dims(data_cfg), _model_config(False))
    out = model(batch)
    assert out["logit"].shape == (3,)
    assert "phq_pred" not in out


def test_model_is_differentiable() -> None:
    data_cfg = _data_config()
    dataset = MockDaicWozDataset(data_cfg, seed=2)
    batch = collate_fn([dataset[i] for i in range(4)])

    model = MultimodalDepressionModel(modality_input_dims(data_cfg), _model_config())
    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logit"], batch["label"].float()
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def _override_config(**kwargs: object) -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(type="stats", hidden_dim=8, out_dim=8, dropout=0.0),
        fusion=FusionConfig(hidden_dim=16),
        head=HeadConfig(hidden_dim=8),
        **kwargs,  # type: ignore[arg-type]
    )


def test_encoder_for_returns_the_shared_config_without_overrides() -> None:
    config = _override_config()
    assert config.encoder_for("text") is config.encoder


def test_encoder_for_applies_a_partial_override() -> None:
    """An override changes only the named fields; the rest are inherited."""
    config = _override_config(encoder_overrides={"text": {"type": "mean"}})

    text_cfg = config.encoder_for("text")
    assert text_cfg.type == "mean"
    assert text_cfg.hidden_dim == config.encoder.hidden_dim  # inherited
    assert config.encoder_for("audio").type == "stats"  # untouched


def test_encoder_for_validates_the_override() -> None:
    config = _override_config(encoder_overrides={"text": {"type": "not-an-encoder"}})
    with pytest.raises(ValueError, match="type"):
        config.encoder_for("text")


def test_model_builds_a_different_encoder_per_modality() -> None:
    """The text override reaches the constructed model, not just the config."""
    config = _override_config(encoder_overrides={"text": {"type": "mean"}})
    model = MultimodalDepressionModel({"audio": 5, "video": 4, "text": 9}, config)

    assert model.encoders["audio"].config.type == "stats"
    assert model.encoders["text"].config.type == "mean"
    # `stats` projects from 5 functionals per channel, `mean` straight from the dim.
    assert model.encoders["audio"].proj.in_features == 5 * 5
    assert model.encoders["text"].proj.in_features == 9
