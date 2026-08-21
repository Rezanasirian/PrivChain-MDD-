"""End-to-end Phase 5 test: federated round writes an auditable trail (H3).

Runs the capability-aware simulation with a MockLedger + DP budget allocator and
asserts the ledger holds the expected reads/writes: a subgraph per round,
registered clients, append-only per-modality budget entries, and reputation.
This is the offline stand-in for the Phase 5 Definition of Done.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from privchain.chain_client import MockLedger
from privchain.config import (
    AggregationConfig,
    AllocationConfig,
    AudioConfig,
    DataConfig,
    DistillationConfig,
    EncoderConfig,
    FederationConfig,
    FusionConfig,
    HeadConfig,
    ModalityPattern,
    ModalityPrivacy,
    ModelConfig,
    ReputationConfig,
    TextConfig,
    VideoConfig,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, collate_fn
from privchain.federated.client import ClientDPConfig
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import build_federated_clients, run_capability_aware_simulation
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.budget_allocator import allocate_target_epsilons
from privchain.seeding import seed_everything
from privchain.training.loaders import split_dataset


def _data_config() -> DataConfig:
    return DataConfig(
        num_sessions=40,
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


def _federation() -> FederationConfig:
    return FederationConfig(
        num_clients=8,
        num_rounds=2,
        clients_per_round=8,
        local_epochs=1,
        modality_patterns=[
            ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.4),
            ModalityPattern(name="audio_text", capability=[1, 0, 1], fraction=0.3),
            ModalityPattern(name="audio_only", capability=[1, 0, 0], fraction=0.2),
            ModalityPattern(name="text_only", capability=[0, 0, 1], fraction=0.1),
        ],
    )


def _aggregation() -> AggregationConfig:
    return AggregationConfig(
        strategy="capability_aware",
        reputation_weighting=True,
        federated_distillation=False,
        byzantine_filter=True,
        reputation=ReputationConfig(),
        distillation=DistillationConfig(),
    )


def _dp_config() -> ClientDPConfig:
    per_modality = {
        "audio": ModalityPrivacy(epsilon=2.0, reidentification_risk=0.9),
        "video": ModalityPrivacy(epsilon=4.0, reidentification_risk=0.6),
        "text": ModalityPrivacy(epsilon=8.0, reidentification_risk=0.3),
    }
    return ClientDPConfig(
        target_epsilons=allocate_target_epsilons(AllocationConfig(mode="explicit"), per_modality),
        delta=1e-5,
        max_grad_norm=1.0,
        batch_size=4,
        num_rounds=2,
        seed=42,
    )


@pytest.mark.integration
def test_federated_round_is_recorded_to_ledger(tmp_path: Path) -> None:
    data_cfg, model_cfg, fed_cfg = _data_config(), _model_config(), _federation()
    seed_everything(42)

    full = MockDaicWozDataset(data_cfg, seed=42)
    train_subset, val_subset = split_dataset(full, 0.25, 42)
    val_loader: DataLoader = DataLoader(
        val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    partitions = build_client_partitions(len(train_subset), fed_cfg, seed=42)
    input_dims = modality_input_dims(data_cfg)
    clients = build_federated_clients(
        train_subset,
        partitions,
        input_dims=input_dims,
        model_config=model_cfg,
        batch_size=4,
        local_epochs=fed_cfg.local_epochs,
        learning_rate=0.01,
        weight_decay=0.0,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
        seed=42,
        client_dp=_dp_config(),
    )
    global_model = MultimodalDepressionModel(input_dims, model_cfg)
    ledger = MockLedger()

    run_dir = tmp_path / "ledger"
    run_dir.mkdir()
    history = run_capability_aware_simulation(
        global_model,
        clients,
        val_loader,
        aggregation=_aggregation(),
        num_rounds=fed_cfg.num_rounds,
        clients_per_round=fed_cfg.clients_per_round,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
        run_dir=run_dir,
        seed=42,
        ledger=ledger,
    )
    assert len(history) == fed_cfg.num_rounds
    assert "num_byzantine_flagged" in history[-1]

    # A subgraph was published for every round.
    subgraphs = [ledger.get_subgraph(r) for r in range(1, fed_cfg.num_rounds + 1)]
    assert all(sg is not None for sg in subgraphs)
    first = subgraphs[0]
    assert first is not None

    # The first client of round 1 is registered, with budget + reputation logged.
    client_id = first.client_ids[0]
    client = ledger.get_client(client_id)
    assert client is not None

    modalities = [
        m for m, flag in zip(("audio", "video", "text"), client.capability, strict=True) if flag
    ]
    assert modalities  # a registered client has at least one modality
    for modality in modalities:
        history_entries = ledger.budget_history(client_id, modality)
        # Append-only: one entry per round the client participated in.
        assert [e.round for e in history_entries] == [1, 2]
        assert all(e.epsilon_spent > 0.0 for e in history_entries)
        assert all(e.epsilon_incremental > 0.0 for e in history_entries)
        assert [e.epsilon_cumulative for e in history_entries] == sorted(
            e.epsilon_cumulative for e in history_entries
        )
        assert ledger.get_reputation(client_id, modality) is not None

    shared = ledger.budget_history(client_id, "shared")
    composed = ledger.budget_history(client_id, "composed")
    assert len(shared) == fed_cfg.num_rounds
    assert len(composed) == fed_cfg.num_rounds
