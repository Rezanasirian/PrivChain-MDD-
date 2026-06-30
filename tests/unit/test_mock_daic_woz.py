"""Unit tests for the synthetic DAIC-WOZ mock dataset (Phase 0)."""

from __future__ import annotations

import pytest
import torch

from privchain.config import AudioConfig, DataConfig, TextConfig, VideoConfig
from privchain.data.mock_daic_woz import (
    MODALITIES,
    MockDaicWozDataset,
    build_dataloader,
    collate_fn,
)


@pytest.fixture
def data_config() -> DataConfig:
    return DataConfig(
        num_sessions=12,
        root="data/mock",
        phq8_max=24,
        depression_cutoff=10,
        audio=AudioConfig(n_mels=40, min_frames=80, max_frames=120),
        video=VideoConfig(n_features=49, min_frames=40, max_frames=60),
        text=TextConfig(embed_dim=64, min_tokens=20, max_tokens=30),
    )


def test_modalities_constant() -> None:
    assert MODALITIES == ("audio", "video", "text")


def test_dataset_length(data_config: DataConfig) -> None:
    dataset = MockDaicWozDataset(data_config, seed=7)
    assert len(dataset) == data_config.num_sessions


def test_sample_shapes_and_dtypes(data_config: DataConfig) -> None:
    dataset = MockDaicWozDataset(data_config, seed=7)
    sample = dataset[0]

    assert sample["audio"].ndim == 2
    assert sample["audio"].shape[1] == data_config.audio.n_mels
    assert sample["video"].shape[1] == data_config.video.n_features
    assert sample["text"].shape[1] == data_config.text.embed_dim

    assert sample["audio"].dtype == torch.float32
    assert sample["video"].dtype == torch.float32
    assert sample["text"].dtype == torch.float32

    # Sequence lengths fall within the configured bounds.
    assert data_config.audio.min_frames <= sample["audio"].shape[0] <= data_config.audio.max_frames
    assert data_config.video.min_frames <= sample["video"].shape[0] <= data_config.video.max_frames
    assert data_config.text.min_tokens <= sample["text"].shape[0] <= data_config.text.max_tokens


def test_labels_in_range(data_config: DataConfig) -> None:
    dataset = MockDaicWozDataset(data_config, seed=7)
    for index in range(len(dataset)):
        sample = dataset[index]
        score = int(sample["phq8_score"].item())
        label = int(sample["label"].item())
        assert 0 <= score <= data_config.phq8_max
        assert label in (0, 1)
        assert label == int(score >= data_config.depression_cutoff)


def test_generation_is_deterministic(data_config: DataConfig) -> None:
    a = MockDaicWozDataset(data_config, seed=123)[3]
    b = MockDaicWozDataset(data_config, seed=123)[3]
    assert torch.equal(a["audio"], b["audio"])
    assert torch.equal(a["text"], b["text"])
    assert torch.equal(a["phq8_score"], b["phq8_score"])


def test_different_seeds_differ(data_config: DataConfig) -> None:
    a = MockDaicWozDataset(data_config, seed=1)[0]
    b = MockDaicWozDataset(data_config, seed=2)[0]
    # Shapes may differ; compare a fixed-size scalar to avoid shape mismatch.
    assert not torch.equal(a["phq8_score"], b["phq8_score"]) or not torch.equal(
        a["audio"][:1], b["audio"][:1]
    )


def test_index_out_of_range_raises(data_config: DataConfig) -> None:
    dataset = MockDaicWozDataset(data_config, seed=7)
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_collate_pads_to_batch_max(data_config: DataConfig) -> None:
    dataset = MockDaicWozDataset(data_config, seed=7)
    samples = [dataset[i] for i in range(4)]
    batch = collate_fn(samples)

    expected_audio_max = max(s["audio"].shape[0] for s in samples)
    assert batch["audio"].shape == (4, expected_audio_max, data_config.audio.n_mels)
    assert batch["audio_lengths"].tolist() == [s["audio"].shape[0] for s in samples]
    # Padded region beyond the true length must be zero.
    shortest = int(batch["audio_lengths"].argmin().item())
    true_len = int(batch["audio_lengths"][shortest].item())
    assert torch.all(batch["audio"][shortest, true_len:] == 0)


def test_collate_empty_raises() -> None:
    with pytest.raises(ValueError):
        collate_fn([])


def test_build_dataloader_batches(data_config: DataConfig) -> None:
    loader = build_dataloader(data_config, batch_size=4, seed=7)
    batch = next(iter(loader))
    assert batch["audio"].shape[0] == 4
    assert batch["label"].shape == (4,)
    assert batch["phq8_score"].shape == (4,)
