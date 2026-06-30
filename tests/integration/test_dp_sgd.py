"""End-to-end Phase 3 per-modality DP-SGD test.

Checks that DP-SGD training runs over mock data, that parameters update, and that
the per-modality grouping/noise plumbing is correct. The accuracy-vs-ε reporting
itself is exercised by `scripts/run_dp_sweep.py`.
"""

from __future__ import annotations

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
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.dp_sgd import (
    SHARED_GROUP,
    dp_train_epoch,
    map_parameter_groups,
    resolve_group_sigmas,
)
from privchain.seeding import seed_everything
from privchain.training.objective import DepressionObjective


def _data_config() -> DataConfig:
    return DataConfig(
        num_sessions=12,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=8, min_frames=6, max_frames=10),
        video=VideoConfig(n_features=7, min_frames=5, max_frames=8),
        text=TextConfig(embed_dim=10, min_tokens=4, max_tokens=6),
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(type="mean", hidden_dim=8, out_dim=6, dropout=0.0),
        fusion=FusionConfig(hidden_dim=10),
        head=HeadConfig(hidden_dim=6),
        use_phq_regression=True,
        phq_loss_weight=0.1,
    )


def test_parameter_grouping_covers_all_params() -> None:
    model = MultimodalDepressionModel(modality_input_dims(_data_config()), _model_config())
    groups = map_parameter_groups(model)
    assert set(groups) == {"audio", "video", "text", SHARED_GROUP}
    grouped = sum(len(v) for v in groups.values())
    assert grouped == len(list(model.parameters()))
    assert len(groups["audio"]) > 0 and len(groups[SHARED_GROUP]) > 0


def test_resolve_group_sigmas_adds_shared_max() -> None:
    sigmas = resolve_group_sigmas({"audio": 3.0, "video": 2.0, "text": 1.0})
    assert sigmas[SHARED_GROUP] == 3.0


def test_dp_train_epoch_updates_parameters() -> None:
    seed_everything(0)
    data_cfg, model_cfg = _data_config(), _model_config()
    dataset = MockDaicWozDataset(data_cfg, seed=0)
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    objective = DepressionObjective(data_cfg.phq8_max, model_cfg.phq_loss_weight)

    groups = map_parameter_groups(model)
    group_sigmas = resolve_group_sigmas({"audio": 1.0, "video": 0.7, "text": 0.5})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(0)

    before = [p.detach().clone() for p in model.parameters()]
    batches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    loss = dp_train_epoch(
        model,
        dataset,
        batches,
        objective,
        groups=groups,
        group_sigmas=group_sigmas,
        max_grad_norm=1.0,
        optimizer=optimizer,
        device=torch.device("cpu"),
        generator=generator,
    )

    assert loss > 0.0
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_zero_noise_still_trains() -> None:
    # sigma=0 -> no noise added; should still run and update.
    seed_everything(1)
    data_cfg, model_cfg = _data_config(), _model_config()
    dataset = MockDaicWozDataset(data_cfg, seed=1)
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    objective = DepressionObjective(data_cfg.phq8_max, model_cfg.phq_loss_weight)
    groups = map_parameter_groups(model)
    sigmas = resolve_group_sigmas({"audio": 0.0, "video": 0.0, "text": 0.0})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss = dp_train_epoch(
        model,
        dataset,
        [[0, 1, 2], [3, 4, 5]],
        objective,
        groups=groups,
        group_sigmas=sigmas,
        max_grad_norm=1.0,
        optimizer=optimizer,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    assert loss > 0.0
