"""End-to-end Phase 7 evaluation-harness smoke test (H5).

Exercises the shared CV protocol on real components: a held-out split + 2-fold CV
where each fold trains the capability-aware federated variant on the training
indices and scores the test indices, then aggregates. Confirms the Chapter-4
harness wiring (split -> per-fold train/eval -> aggregate -> latency) works. On
mock data the accuracy numbers are placeholders (labels are random).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from privchain.config import (
    AggregationConfig,
    AudioConfig,
    DataConfig,
    DistillationConfig,
    EncoderConfig,
    FederationConfig,
    FusionConfig,
    HeadConfig,
    ModalityPattern,
    ModelConfig,
    ReputationConfig,
    TextConfig,
    VideoConfig,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, collate_fn
from privchain.eval.benchmark import (
    aggregate_metrics,
    held_out_split,
    k_fold_indices,
    measure_inference_latency,
)
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import build_federated_clients, run_capability_aware_simulation
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.objective import DepressionObjective, evaluate_model


def _configs() -> tuple[DataConfig, ModelConfig, FederationConfig]:
    data = DataConfig(
        num_sessions=20, root="data/mock", phq8_max=24, depression_cutoff=10,
        audio=AudioConfig(n_mels=10, min_frames=8, max_frames=12),
        video=VideoConfig(n_features=8, min_frames=5, max_frames=9),
        text=TextConfig(embed_dim=12, min_tokens=4, max_tokens=8),
    )
    model = ModelConfig(
        encoder=EncoderConfig(type="gru", hidden_dim=6, out_dim=6),
        fusion=FusionConfig(hidden_dim=8), head=HeadConfig(hidden_dim=6),
    )
    federation = FederationConfig(
        num_clients=4, num_rounds=1, clients_per_round=4, local_epochs=1,
        modality_patterns=[
            ModalityPattern(name="full", capability=[1, 1, 1], fraction=0.5),
            ModalityPattern(name="audio_text", capability=[1, 0, 1], fraction=0.25),
            ModalityPattern(name="text_only", capability=[0, 0, 1], fraction=0.25),
        ],
    )
    return data, model, federation


def _run_fold(full: MockDaicWozDataset, train_idx: list[int], test_idx: list[int]) -> dict:
    data_cfg, model_cfg, fed_cfg = _configs()
    seed_everything(0)
    input_dims = modality_input_dims(data_cfg)
    train_subset = Subset(full, train_idx)
    partitions = build_client_partitions(len(train_subset), fed_cfg, seed=0)
    clients = build_federated_clients(
        train_subset, partitions, input_dims=input_dims, model_config=model_cfg,
        batch_size=4, local_epochs=1, learning_rate=0.01, weight_decay=0.0,
        phq8_max=data_cfg.phq8_max, phq_loss_weight=model_cfg.phq_loss_weight, seed=0,
    )
    global_model = MultimodalDepressionModel(input_dims, model_cfg)
    test_loader: DataLoader = DataLoader(
        Subset(full, test_idx), batch_size=4, collate_fn=collate_fn
    )
    aggregation = AggregationConfig(
        strategy="capability_aware", reputation_weighting=True, federated_distillation=True,
        reputation=ReputationConfig(), distillation=DistillationConfig(),
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_capability_aware_simulation(
            global_model, clients, test_loader, aggregation=aggregation,
            num_rounds=1, clients_per_round=4, phq8_max=data_cfg.phq8_max,
            phq_loss_weight=model_cfg.phq_loss_weight, run_dir=Path(tmp), seed=0,
        )
    objective = DepressionObjective(data_cfg.phq8_max, model_cfg.phq_loss_weight)
    return evaluate_model(global_model, test_loader, objective, torch.device("cpu"))


@pytest.mark.integration
def test_cross_validation_harness_runs(tmp_path: Path) -> None:
    data_cfg, model_cfg, _ = _configs()
    full = MockDaicWozDataset(data_cfg, seed=0)
    n = len(full)

    dev_idx, held_out_idx = held_out_split(n, 0.2, seed=0)
    assert set(dev_idx).isdisjoint(held_out_idx)
    splits = k_fold_indices(len(dev_idx), 2, seed=0)

    per_fold = [
        _run_fold(full, [dev_idx[i] for i in tr], [dev_idx[i] for i in te])
        for tr, te in splits
    ]
    agg = aggregate_metrics(per_fold)
    assert agg["num_folds"] == 2.0
    assert "f1_mean" in agg and "f1_std" in agg
    assert "roc_auc_mean" in agg

    # Held-out evaluation runs too.
    held = _run_fold(full, dev_idx, held_out_idx)
    assert "f1" in held and "roc_auc" in held

    # Latency benchmark produces one record per batch size.
    latency = measure_inference_latency(
        MultimodalDepressionModel(modality_input_dims(data_cfg), model_cfg),
        full, batch_sizes=[1, 4], repeats=2, device=torch.device("cpu"),
    )
    assert len(latency) == 2
    assert all(r["ms_per_batch"] > 0 for r in latency)
