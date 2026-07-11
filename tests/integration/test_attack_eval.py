"""End-to-end Phase 6 attacker-evaluation smoke test (H5).

Extracts per-modality subject embeddings through the encoders, then confirms the
re-identification attacker behaves as the Chapter-4 table needs: a real identity
signal at low noise, collapsing toward chance under heavy noise. On mock data the
depression label is noise, but subject identity is a genuine signal.
"""

from __future__ import annotations

import numpy as np
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
from privchain.eval.attackers import ReidentificationAttacker, add_gaussian_noise
from privchain.eval.embeddings import extract_subject_embeddings, split_enroll_probe
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything


def _data_config() -> DataConfig:
    return DataConfig(
        num_sessions=12,
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


@pytest.mark.integration
def test_reidentification_signal_decays_with_noise() -> None:
    data_cfg, model_cfg = _data_config(), _model_config()
    seed_everything(7)
    dataset = MockDaicWozDataset(data_cfg, seed=7)
    input_dims = modality_input_dims(data_cfg)
    model = MultimodalDepressionModel(input_dims, model_cfg)

    embeddings, subjects, views = extract_subject_embeddings(
        model, dataset, "audio", num_views=6, jitter=0.05, seed=7, device=torch.device("cpu")
    )
    # 12 subjects x 6 views.
    assert embeddings.shape[0] == 12 * 6
    assert set(np.unique(subjects)) == set(range(12))

    enroll_emb, enroll_ids, probe_emb, probe_ids = split_enroll_probe(
        embeddings, subjects, views, enroll_views=3
    )

    attacker = ReidentificationAttacker()
    attacker.enroll(enroll_emb, enroll_ids)
    clean = attacker.attack(probe_emb, probe_ids)

    rng = np.random.default_rng(0)
    noisy = attacker.attack(add_gaussian_noise(probe_emb, std=100.0, rng=rng), probe_ids)

    chance = ReidentificationAttacker.chance_accuracy(12)
    assert clean > 3 * chance          # identity is clearly recoverable when clean
    assert noisy <= clean              # heavy noise never helps the attacker
    assert 0.0 <= noisy <= 1.0
