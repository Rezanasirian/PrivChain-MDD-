"""Unit tests for federated partitioning and modality masking (Phase 2)."""

from __future__ import annotations

import pytest
import torch

from privchain.config import (
    AudioConfig,
    DataConfig,
    FederationConfig,
    ModalityPattern,
    TextConfig,
    VideoConfig,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset
from privchain.federated.partition import (
    ModalityMaskedDataset,
    assign_capabilities,
    build_client_partitions,
    partition_indices,
)


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


def _federation(num_clients: int = 8) -> FederationConfig:
    return FederationConfig(
        num_clients=num_clients,
        num_rounds=3,
        clients_per_round=num_clients,
        local_epochs=1,
        modality_patterns=[
            ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.4),
            ModalityPattern(name="audio_text", capability=[1, 0, 1], fraction=0.3),
            ModalityPattern(name="audio_only", capability=[1, 0, 0], fraction=0.2),
            ModalityPattern(name="text_only", capability=[0, 0, 1], fraction=0.1),
        ],
    )


def test_assign_capabilities_counts_sum_to_num_clients() -> None:
    fed = _federation(8)
    assignments = assign_capabilities(fed, seed=1)
    assert len(assignments) == 8
    names = [name for name, _ in assignments]
    # All four patterns should appear for this mix at N=8.
    assert set(names) == {"full", "audio_text", "audio_only", "text_only"}


def test_partition_indices_disjoint_and_complete() -> None:
    shards = partition_indices(num_items=23, num_clients=5, seed=3)
    assert len(shards) == 5
    flat = [i for shard in shards for i in shard]
    assert sorted(flat) == list(range(23))  # complete + disjoint
    assert all(len(shard) > 0 for shard in shards)


def test_partition_indices_rejects_too_many_clients() -> None:
    with pytest.raises(ValueError):
        partition_indices(num_items=2, num_clients=5, seed=0)


def test_build_client_partitions() -> None:
    fed = _federation(8)
    partitions = build_client_partitions(num_items=40, federation=fed, seed=7)
    assert len(partitions) == 8
    assert {p.client_id for p in partitions} == set(range(8))
    all_indices = [i for p in partitions for i in p.indices]
    assert sorted(all_indices) == list(range(40))


def test_modality_masking_zeros_absent_modalities() -> None:
    data_cfg = _data_config()
    base = MockDaicWozDataset(data_cfg, seed=2)
    # audio_text capability: video absent.
    masked = ModalityMaskedDataset(base, indices=[0, 1, 2], capability=(1, 0, 1))
    sample = masked[0]
    original = base[0]

    assert sample["video"].shape == (1, data_cfg.video.n_features)
    assert torch.all(sample["video"] == 0)
    # Present modalities are untouched.
    assert torch.equal(sample["audio"], original["audio"])
    assert torch.equal(sample["text"], original["text"])
    assert len(masked) == 3
