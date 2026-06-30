"""In-house federated FedAvg simulator (Phase 2, objective H2).

A framework-agnostic server loop: build heterogeneous clients from partitions,
then each round broadcast the global parameters, run local training on a sampled
subset of clients, FedAvg-aggregate, and evaluate the global model on a held-out
full-modality validation set, logging per-round metrics.

This mirrors what the Flower backend (:mod:`privchain.federated.flower_app`)
does, but runs without the ``flwr`` dependency so it is testable offline. Both
backends reuse the same :class:`~privchain.federated.client.FederatedClient` and
:func:`~privchain.federated.aggregation.fedavg`.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from privchain.config import ModelConfig
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.federated.aggregation import fedavg
from privchain.federated.client import FederatedClient
from privchain.federated.partition import ClientPartition, ModalityMaskedDataset
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.training.experiment import JsonlMetricLogger
from privchain.training.objective import DepressionObjective, evaluate_model


def build_federated_clients(
    base_dataset: Any,
    partitions: list[ClientPartition],
    *,
    input_dims: dict[str, int],
    model_config: ModelConfig,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    phq8_max: int,
    phq_loss_weight: float,
    seed: int,
    device: str = "cpu",
) -> list[FederatedClient]:
    """Construct one :class:`FederatedClient` per partition.

    Args:
        base_dataset: The underlying dataset shared by all clients.
        partitions: Per-client partitions (indices + capability).
        input_dims: Per-modality input feature dims for the model.
        model_config: Model configuration (each client gets its own instance).
        batch_size: Local batch size.
        local_epochs: Local epochs per round.
        learning_rate: Local optimizer learning rate.
        weight_decay: Local optimizer weight decay.
        phq8_max: Max PHQ-8 score (objective).
        phq_loss_weight: PHQ-8 regression weight.
        seed: Base seed for per-client loader shuffling.
        device: Torch device string.

    Returns:
        A list of constructed clients (skipping any empty partition).
    """
    clients: list[FederatedClient] = []
    for partition in partitions:
        if not partition.indices:
            continue
        dataset = ModalityMaskedDataset(base_dataset, partition.indices, partition.capability)
        loader: DataLoader[Sample] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed + partition.client_id),
        )
        model = MultimodalDepressionModel(input_dims, model_config)
        clients.append(
            FederatedClient(
                partition.client_id,
                partition.capability,
                model,
                loader,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                phq8_max=phq8_max,
                phq_loss_weight=phq_loss_weight,
                device=device,
            )
        )
    return clients


def run_simulation(
    global_model: MultimodalDepressionModel,
    clients: list[FederatedClient],
    val_loader: DataLoader[Sample],
    *,
    num_rounds: int,
    clients_per_round: int,
    phq8_max: int,
    phq_loss_weight: float,
    run_dir: Path,
    seed: int,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run FedAvg for ``num_rounds`` rounds, logging per-round global metrics.

    Args:
        global_model: The server's global model (updated in place each round).
        clients: The federated clients.
        val_loader: Held-out full-modality validation loader.
        num_rounds: Number of federated rounds.
        clients_per_round: Clients sampled (without replacement) each round.
        phq8_max: Max PHQ-8 score (objective).
        phq_loss_weight: PHQ-8 regression weight.
        run_dir: Experiment run directory for ``metrics.jsonl`` / checkpoints.
        seed: Base seed for per-round client sampling.
        device: Torch device string.

    Returns:
        Per-round history records.

    Raises:
        ValueError: If there are no clients to train.
    """
    if not clients:
        raise ValueError("no clients to run federated simulation")

    torch_device = torch.device(device)
    global_model = global_model.to(torch_device)
    objective = DepressionObjective(phq8_max, phq_loss_weight)
    logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    history: list[dict[str, Any]] = []
    best_score = -float("inf")

    global_state: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items()
    )
    k_per_round = min(clients_per_round, len(clients))

    for round_num in range(1, num_rounds + 1):
        rng = np.random.default_rng(seed + round_num)
        selected_idx = rng.choice(len(clients), size=k_per_round, replace=False)
        selected = [clients[int(i)] for i in selected_idx]

        states: list[OrderedDict[str, torch.Tensor]] = []
        weights: list[float] = []
        for client in selected:
            updated, num_samples = client.fit(global_state)
            states.append(updated)
            weights.append(float(num_samples))

        global_state = fedavg(states, weights)
        global_model.load_state_dict(global_state)

        metrics = evaluate_model(global_model, val_loader, objective, torch_device)
        record: dict[str, Any] = {"round": round_num, "num_clients": len(selected)}
        record.update({f"val_{k}": v for k, v in metrics.items()})
        logger.log(record)
        history.append(record)

        selector = metrics["roc_auc"]
        if np.isnan(selector):
            selector = metrics["f1"]
        if selector > best_score:
            best_score = selector
            torch.save(global_state, run_dir / "best_global_model.pt")

    return history
