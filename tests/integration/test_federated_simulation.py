"""End-to-end Phase 2 federated simulation smoke test.

Definition of Done for Phase 2: simulated federated training runs across >= 3
heterogeneous clients with metrics logged. This builds a heterogeneous client
population over mock data and runs a couple of FedAvg rounds via the in-house
simulator (the Flower backend mirrors this but needs the optional ``flwr`` dep).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from privchain.config import (
    AudioConfig,
    DataConfig,
    EncoderConfig,
    FederationConfig,
    FusionConfig,
    HeadConfig,
    ModalityPattern,
    ModelConfig,
    TextConfig,
    VideoConfig,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, collate_fn
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import build_federated_clients, run_simulation
from privchain.fusion.baseline_model import MultimodalDepressionModel
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


@pytest.mark.integration
def test_federated_simulation_runs_and_logs(tmp_path: Path) -> None:
    data_cfg, model_cfg, fed_cfg = _data_config(), _model_config(), _federation()
    seed_everything(42)

    full = MockDaicWozDataset(data_cfg, seed=42)
    train_subset, val_subset = split_dataset(full, 0.25, 42)
    val_loader: DataLoader = DataLoader(
        val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    partitions = build_client_partitions(len(train_subset), fed_cfg, seed=42)
    # Heterogeneity: at least 3 distinct modality-access patterns present.
    assert len({p.pattern_name for p in partitions}) >= 3

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
    )
    assert len(clients) >= 3

    global_model = MultimodalDepressionModel(input_dims, model_cfg)
    run_dir = tmp_path / "fed"
    run_dir.mkdir()
    history = run_simulation(
        global_model,
        clients,
        val_loader,
        num_rounds=fed_cfg.num_rounds,
        clients_per_round=fed_cfg.clients_per_round,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
        run_dir=run_dir,
        seed=42,
    )

    assert len(history) == fed_cfg.num_rounds
    final = history[-1]
    assert "val_f1" in final and "val_roc_auc" in final
    assert (run_dir / "best_global_model.pt").is_file()
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == fed_cfg.num_rounds
    assert "round" in json.loads(lines[0])


def test_class_weighting_reaches_clients_from_their_own_shard() -> None:
    """Clients weight their BCE like the centralized arm, from local labels only.

    The centralized baseline weighted its loss whenever ``train.class_weighting``
    was set while every federated client trained unweighted, so the Chapter-4
    comparison charged federation for a difference in objective rather than in
    method (ADR-0026). Pins both halves: the flag has to reach the clients, and
    the weight has to be measured per shard rather than shared.
    """
    seed_everything(42)
    data_cfg, model_cfg, fed_cfg = _data_config(), _model_config(), _federation()
    full = MockDaicWozDataset(data_cfg, seed=42)
    train_subset, _ = split_dataset(full, 0.25, 42)
    partitions = build_client_partitions(len(train_subset), fed_cfg, seed=42)

    common = {
        "input_dims": modality_input_dims(data_cfg),
        "model_config": model_cfg,
        "batch_size": 4,
        "local_epochs": fed_cfg.local_epochs,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "phq8_max": data_cfg.phq8_max,
        "phq_loss_weight": model_cfg.phq_loss_weight,
        "seed": 42,
    }

    off = build_federated_clients(train_subset, partitions, **common)  # type: ignore[arg-type]
    assert all(c.pos_weight is None for c in off)
    assert all(c.objective.pos_weight is None for c in off)

    per_shard = build_federated_clients(  # type: ignore[arg-type]
        train_subset, partitions, class_weight_mode="per_shard", **common
    )
    weights = [c.pos_weight for c in per_shard]
    assert any(w is not None for w in weights), "the mode never reached the clients"
    assert all(w is None or w > 0.0 for w in weights)
    # A shard holding one class only is undefined, and stays unweighted.
    for client, weight in zip(per_shard, weights, strict=True):
        assert client.objective.pos_weight == weight

    pooled = build_federated_clients(  # type: ignore[arg-type]
        train_subset, partitions, class_weight_mode="pooled_oracle", **common
    )
    pooled_weights = {c.pos_weight for c in pooled}
    assert len(pooled_weights) == 1, "the oracle weight must be identical across clients"
    # The control only means anything if it differs from what shards see locally.
    assert len(set(weights)) > 1, "fixture no longer produces uneven shards"


def test_round_log_records_local_work() -> None:
    """Each round logs the optimizer steps and examples it actually spent.

    A shard smaller than ``batch_size`` collapses to one batch, so a 120-round
    budget can carry 120 optimizer steps per client rather than the many implied
    by the round count. The counts have to be logged for a sweep to hold total
    steps fixed while varying how they are grouped, and for the claim to be
    checkable at all.
    """
    seed_everything(42)
    data_cfg, model_cfg, fed_cfg = _data_config(), _model_config(), _federation()
    full = MockDaicWozDataset(data_cfg, seed=42)
    train_subset, val_subset = split_dataset(full, 0.25, 42)
    val_loader: DataLoader = DataLoader(
        val_subset, batch_size=4, shuffle=False, collate_fn=collate_fn
    )
    partitions = build_client_partitions(len(train_subset), fed_cfg, seed=42)
    clients = build_federated_clients(
        train_subset,
        partitions,
        input_dims=modality_input_dims(data_cfg),
        model_config=model_cfg,
        batch_size=4,
        local_epochs=fed_cfg.local_epochs,
        learning_rate=0.01,
        weight_decay=0.0,
        phq8_max=data_cfg.phq8_max,
        phq_loss_weight=model_cfg.phq_loss_weight,
        seed=42,
    )
    global_model = MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        history = run_simulation(
            global_model,
            clients,
            val_loader,
            num_rounds=fed_cfg.num_rounds,
            clients_per_round=fed_cfg.clients_per_round,
            phq8_max=data_cfg.phq8_max,
            phq_loss_weight=model_cfg.phq_loss_weight,
            run_dir=run_dir,
            seed=42,
        )

    first = history[0]
    for key in (
        "optimizer_steps",
        "num_local_batches",
        "examples_seen",
        "cumulative_optimizer_steps",
        "cumulative_examples_seen",
        "elapsed_seconds",
    ):
        assert key in first, f"{key} missing from the round record"

    # One step per batch, and no client can spend more examples than it holds.
    assert first["optimizer_steps"] == first["num_local_batches"]
    held = sum(len(p.indices) for p in partitions)
    assert 0 < first["examples_seen"] <= held * fed_cfg.local_epochs
    # Cumulative counters only ever grow.
    assert history[-1]["cumulative_optimizer_steps"] >= first["cumulative_optimizer_steps"]
