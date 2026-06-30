"""End-to-end Phase 0 smoke test.

Definition of Done for Phase 0: a smoke-test ``pytest`` run on the mock-data
pipeline passes and produces correctly shaped tensors. This drives the pipeline
the way real code will: load + validate the real ``configs/baseline.yaml``,
seed everything, build a DataLoader, and assert the batched tensor shapes match
the configured modality dimensions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from privchain.config import load_baseline_config
from privchain.data.mock_daic_woz import build_dataloader
from privchain.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"


@pytest.mark.integration
def test_baseline_config_loads_and_validates() -> None:
    config = load_baseline_config(BASELINE_CONFIG)
    assert config.seed == 42
    assert config.data.num_sessions > 0


@pytest.mark.integration
def test_mock_pipeline_produces_correctly_shaped_tensors() -> None:
    config = load_baseline_config(BASELINE_CONFIG)
    seed_everything(config.seed)

    batch_size = config.train.batch_size
    loader = build_dataloader(
        config.data, batch_size=batch_size, seed=config.seed, shuffle=True
    )

    batch = next(iter(loader))

    # Feature dims must match the config exactly; time dims are the padded max.
    assert batch["audio"].ndim == 3
    assert batch["audio"].shape[0] == batch_size
    assert batch["audio"].shape[2] == config.data.audio.n_mels
    assert batch["video"].shape[2] == config.data.video.n_features
    assert batch["text"].shape[2] == config.data.text.embed_dim

    # Labels and lengths are well-formed.
    assert batch["label"].shape == (batch_size,)
    assert torch.all((batch["label"] == 0) | (batch["label"] == 1))
    assert torch.all(batch["audio_lengths"] <= batch["audio"].shape[1])

    # Float features, integer labels — exactly what the encoders will expect.
    assert batch["audio"].dtype == torch.float32
    assert batch["label"].dtype == torch.long


@pytest.mark.integration
def test_full_pass_over_loader_is_stable() -> None:
    config = load_baseline_config(BASELINE_CONFIG)
    seed_everything(config.seed)
    loader = build_dataloader(config.data, batch_size=8, seed=config.seed)

    total = 0
    for batch in loader:
        total += int(batch["label"].shape[0])
    assert total == config.data.num_sessions
