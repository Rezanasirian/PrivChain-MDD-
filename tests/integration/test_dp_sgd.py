"""End-to-end Phase 3 per-modality DP-SGD test.

Checks that DP-SGD training runs over mock data, that parameters update, that
Poisson subsampling behaves, and — most importantly — that the fast
``GradSampleModule`` path produces exactly the same clipped gradients as the
straightforward one-sample-at-a-time path. The accuracy-vs-ε reporting itself is
exercised by `scripts/run_dp_sweep.py`.
"""

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
from privchain.data.mock_daic_woz import MockDaicWozDataset
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.dp_sgd import (
    SHARED_GROUP,
    dp_train_steps,
    map_parameter_groups,
    poisson_batches,
    resolve_group_sigmas,
    steps_for_epochs,
    wrap_for_per_sample_grads,
)
from privchain.seeding import seed_everything
from privchain.training.objective import DepressionObjective

DEVICE = torch.device("cpu")


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


def _model_config(encoder_type: str = "mean") -> ModelConfig:
    return ModelConfig(
        encoder=EncoderConfig(type=encoder_type, hidden_dim=8, out_dim=6, dropout=0.0),
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


def test_parameter_grouping_survives_grad_sample_wrapper() -> None:
    """`GradSampleModule` renames parameters to `_module.*` — grouping must cope."""
    model = MultimodalDepressionModel(modality_input_dims(_data_config()), _model_config())
    wrapped = wrap_for_per_sample_grads(model)
    plain_groups = map_parameter_groups(model)
    wrapped_groups = map_parameter_groups(wrapped)
    assert {k: len(v) for k, v in plain_groups.items()} == {
        k: len(v) for k, v in wrapped_groups.items()
    }
    assert len(wrapped_groups[SHARED_GROUP]) > 0


def test_resolve_group_sigmas_adds_shared_max() -> None:
    sigmas = resolve_group_sigmas({"audio": 3.0, "video": 2.0, "text": 1.0})
    assert sigmas[SHARED_GROUP] == 3.0


# ── Poisson subsampling ─────────────────────────────────────────────────────


def test_poisson_batches_have_varying_size_around_expectation() -> None:
    generator = torch.Generator().manual_seed(0)
    batches = poisson_batches(200, 0.1, 50, generator)
    assert len(batches) == 50
    sizes = [len(b) for b in batches]
    assert len(set(sizes)) > 1, "fixed-size batches would violate the accountant"
    assert 15 < sum(sizes) / len(sizes) < 25  # E[|B|] = q*N = 20
    for batch in batches:
        assert len(batch) == len(set(batch)), "sampling is without replacement within a step"


def test_poisson_batches_are_reproducible() -> None:
    first = poisson_batches(50, 0.2, 5, torch.Generator().manual_seed(7))
    second = poisson_batches(50, 0.2, 5, torch.Generator().manual_seed(7))
    assert first == second


def test_poisson_batches_reject_invalid_rate() -> None:
    with pytest.raises(ValueError):
        poisson_batches(10, 0.0, 3, torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        poisson_batches(10, 1.5, 3, torch.Generator().manual_seed(0))


def test_steps_for_epochs() -> None:
    assert steps_for_epochs(100, 10, 3) == 30
    assert steps_for_epochs(95, 10, 2) == 20


# ── training ────────────────────────────────────────────────────────────────


def _train(
    backend: str,
    batches: list[list[int]],
    *,
    encoder_type: str = "mean",
    sigmas: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[list[torch.Tensor], float]:
    """Train one DP run and return the resulting parameters plus the loss."""
    seed_everything(seed)
    data_cfg, model_cfg = _data_config(), _model_config(encoder_type)
    dataset = MockDaicWozDataset(data_cfg, seed=0)
    base = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    base.eval()  # disable dropout so the two backends are comparable
    model: torch.nn.Module = wrap_for_per_sample_grads(base) if backend == "grad_sample" else base

    objective = DepressionObjective(data_cfg.phq8_max, model_cfg.phq_loss_weight)
    groups = map_parameter_groups(model)
    group_sigmas = resolve_group_sigmas(sigmas or {"audio": 0.0, "video": 0.0, "text": 0.0})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    loss = dp_train_steps(
        model,
        dataset,
        batches,
        objective,
        groups=groups,
        group_sigmas=group_sigmas,
        max_grad_norm=1.0,
        expected_batch_size=4.0,
        optimizer=optimizer,
        device=DEVICE,
        generator=torch.Generator(device=DEVICE).manual_seed(seed),
        backend=backend,  # type: ignore[arg-type]
    )
    return [p.detach().clone() for p in base.parameters()], loss


@pytest.mark.parametrize("encoder_type", ["mean", "gru"])
def test_backends_agree(encoder_type: str) -> None:
    """The fast path must never be trusted blindly: it must match microbatching."""
    batches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fast, fast_loss = _train("grad_sample", batches, encoder_type=encoder_type)
    slow, slow_loss = _train("microbatch", batches, encoder_type=encoder_type)

    assert fast_loss == pytest.approx(slow_loss, rel=1e-4)
    for a, b in zip(fast, slow, strict=True):
        assert torch.allclose(a, b, atol=1e-5), "per-sample gradients diverge between backends"


def test_dp_train_updates_parameters_with_noise() -> None:
    seed_everything(0)
    batches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    noisy, loss = _train("grad_sample", batches, sigmas={"audio": 1.0, "video": 0.7, "text": 0.5})
    clean, _ = _train("grad_sample", batches)
    assert loss > 0.0
    assert any(not torch.allclose(a, b) for a, b in zip(noisy, clean, strict=True))


def test_empty_poisson_batch_still_steps() -> None:
    """An empty draw must still consume its mechanism application (noise + step)."""
    before, _ = _train("grad_sample", [], sigmas={"audio": 1.0, "video": 1.0, "text": 1.0})
    after, _ = _train("grad_sample", [[]], sigmas={"audio": 1.0, "video": 1.0, "text": 1.0})
    assert any(not torch.allclose(a, b) for a, b in zip(before, after, strict=True))


def test_rejects_generator_on_wrong_device_type() -> None:
    class _FakeGenerator:
        device = torch.device("cuda")

    data_cfg, model_cfg = _data_config(), _model_config()
    model = wrap_for_per_sample_grads(
        MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    )
    with pytest.raises(ValueError, match="generator is on"):
        dp_train_steps(
            model,
            MockDaicWozDataset(data_cfg, seed=0),
            [[0, 1]],
            DepressionObjective(data_cfg.phq8_max, model_cfg.phq_loss_weight),
            groups=map_parameter_groups(model),
            group_sigmas=resolve_group_sigmas({"audio": 1.0, "video": 1.0, "text": 1.0}),
            max_grad_norm=1.0,
            expected_batch_size=2.0,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            device=DEVICE,
            generator=_FakeGenerator(),  # type: ignore[arg-type]
        )


def test_backend_model_mismatch_is_rejected() -> None:
    data_cfg, model_cfg = _data_config(), _model_config()
    model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)
    with pytest.raises(ValueError, match="GradSampleModule"):
        dp_train_steps(
            model,
            MockDaicWozDataset(data_cfg, seed=0),
            [[0, 1]],
            DepressionObjective(data_cfg.phq8_max, model_cfg.phq_loss_weight),
            groups=map_parameter_groups(model),
            group_sigmas=resolve_group_sigmas({"audio": 1.0, "video": 1.0, "text": 1.0}),
            max_grad_norm=1.0,
            expected_batch_size=2.0,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            device=DEVICE,
            generator=torch.Generator(device=DEVICE).manual_seed(0),
            backend="grad_sample",
        )
