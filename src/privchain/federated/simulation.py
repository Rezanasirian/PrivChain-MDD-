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

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from privchain.chain_client import LedgerClient, record_round
from privchain.config import AggregationConfig, ModelConfig
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.federated.aggregation import ClientUpdate, capability_aware_aggregate, fedavg
from privchain.federated.capability import is_missing_any
from privchain.federated.client import FederatedClient
from privchain.federated.partition import ClientPartition, ModalityMaskedDataset
from privchain.federated.reputation import ReputationTracker
from privchain.federated.robust import flag_byzantine_updates
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.budget_allocator import PerModalityBudgetAllocator
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


def _evaluate_capabilities(
    global_model: MultimodalDepressionModel,
    capability_val_loaders: dict[str, DataLoader[Sample]] | None,
    objective: DepressionObjective,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the global model under each modality-access pattern.

    Args:
        global_model: The current global model.
        capability_val_loaders: Optional ``{pattern_name: masked val loader}``
            simulating inference when only that pattern's modalities are present.
        objective: The evaluation objective.
        device: Torch device.

    Returns:
        Flat metrics keyed ``val_<pattern>_<metric>`` (empty if no loaders).
    """
    if not capability_val_loaders:
        return {}
    record: dict[str, float] = {}
    for pattern_name, loader in capability_val_loaders.items():
        pattern_metrics = evaluate_model(global_model, loader, objective, device)
        record.update({f"val_{pattern_name}_{k}": v for k, v in pattern_metrics.items()})
    return record


def _record_round_to_ledger(
    ledger: LedgerClient,
    round_num: int,
    updates: list[ClientUpdate],
    tracker: ReputationTracker,
    budget_allocator: PerModalityBudgetAllocator | None,
    registered: set[str],
) -> None:
    """Write a round's subgraph, consumed budget, and reputation to the ledger.

    Args:
        ledger: The audit ledger client.
        round_num: The current round.
        updates: The client updates that were aggregated this round.
        tracker: The reputation tracker (its snapshot is logged).
        budget_allocator: Optional DP allocator for per-modality consumed ε.
        registered: Set of already-registered client IDs (updated in place).
    """
    participants = [(str(u.client_id), u.capability) for u in updates]
    reputation = {str(cid): groups for cid, groups in tracker.snapshot().items()}
    consumed = budget_allocator.consumed_epsilon(round_num) if budget_allocator else None
    record_round(
        ledger,
        round_num=round_num,
        participants=participants,
        reputation=reputation,
        consumed_epsilon=consumed,
        registered=registered,
    )


def run_capability_aware_simulation(
    global_model: MultimodalDepressionModel,
    clients: list[FederatedClient],
    val_loader: DataLoader[Sample],
    *,
    aggregation: AggregationConfig,
    num_rounds: int,
    clients_per_round: int,
    phq8_max: int,
    phq_loss_weight: float,
    run_dir: Path,
    seed: int,
    capability_val_loaders: dict[str, DataLoader[Sample]] | None = None,
    ledger: LedgerClient | None = None,
    budget_allocator: PerModalityBudgetAllocator | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run the Phase 4 capability-aware protocol (objectives H2, and H3 wiring).

    Replaces plain FedAvg with: (1) per-modality subgraph aggregation, (2)
    reputation-weighted averaging, (3) federated distillation for
    missing-modality clients, and — when enabled — (4) a Byzantine outlier
    filter (Phase 5). Per-round global metrics — overall and, when
    ``capability_val_loaders`` is given, per modality-access pattern — are logged
    to ``metrics.jsonl`` and the per-client reputation snapshot to
    ``reputation.jsonl``.

    When a ``ledger`` is supplied, each round writes its audit trail (subgraph,
    per-modality consumed ε from ``budget_allocator``, and reputation) to the
    blockchain layer and is thereby auditable (objective H3, Phase 5).

    Args:
        global_model: The server's global model (updated in place each round).
        clients: The federated clients.
        val_loader: Held-out full-modality validation loader.
        aggregation: Aggregation configuration (reputation + distillation flags).
        num_rounds: Number of federated rounds.
        clients_per_round: Clients sampled (without replacement) each round.
        phq8_max: Max PHQ-8 score (objective).
        phq_loss_weight: PHQ-8 regression weight.
        run_dir: Experiment run directory for logs / checkpoints.
        seed: Base seed for per-round client sampling.
        capability_val_loaders: Optional masked validation loaders per pattern.
        ledger: Optional audit ledger; when set, each round is recorded to it.
        budget_allocator: Optional DP allocator; when set (with a ledger), the
            per-modality consumed ε is logged each round.
        device: Torch device string.

    Returns:
        Per-round history records.

    Raises:
        ValueError: If there are no clients to train.
    """
    if not clients:
        raise ValueError("no clients to run capability-aware simulation")

    torch_device = torch.device(device)
    global_model = global_model.to(torch_device)
    objective = DepressionObjective(phq8_max, phq_loss_weight)
    logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
    reputation_logger = JsonlMetricLogger(run_dir / "reputation.jsonl")
    tracker = ReputationTracker(aggregation.reputation)
    distill = aggregation.distillation

    # A single reusable frozen teacher; its parameters are refreshed each round.
    teacher: MultimodalDepressionModel | None = None
    if aggregation.federated_distillation and distill.weight > 0.0:
        teacher = copy.deepcopy(global_model).to(torch_device)
        for param in teacher.parameters():
            param.requires_grad_(False)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")

    global_state: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items()
    )
    k_per_round = min(clients_per_round, len(clients))
    registered: set[str] = set()

    for round_num in range(1, num_rounds + 1):
        rng = np.random.default_rng(seed + round_num)
        selected_idx = rng.choice(len(clients), size=k_per_round, replace=False)
        selected = [clients[int(i)] for i in selected_idx]

        if teacher is not None:
            teacher.load_state_dict(global_state)
            teacher.eval()

        updates: list[ClientUpdate] = []
        for client in selected:
            use_distill = teacher is not None and (
                distill.apply_to == "all" or is_missing_any(client.capability)
            )
            updated, num_samples = client.fit(
                global_state,
                teacher=teacher if use_distill else None,
                distill_weight=distill.weight if use_distill else 0.0,
                distill_temperature=distill.temperature,
            )
            updates.append(ClientUpdate(client.client_id, client.capability, num_samples, updated))

        # Byzantine robustness: drop shared-group outlier updates before aggregation.
        num_flagged = 0
        if aggregation.byzantine_filter:
            flagged = flag_byzantine_updates(
                updates, global_state, z_thresh=aggregation.byzantine_z
            )
            num_flagged = len(flagged)
            if flagged:
                updates = [u for i, u in enumerate(updates) if i not in flagged]

        group_weights = tracker.compute_weights(
            updates, global_state, use_reputation=aggregation.reputation_weighting
        )
        global_state = capability_aware_aggregate(updates, global_state, group_weights)
        global_model.load_state_dict(global_state)

        if ledger is not None:
            _record_round_to_ledger(
                ledger, round_num, updates, tracker, budget_allocator, registered
            )

        metrics = evaluate_model(global_model, val_loader, objective, torch_device)
        record: dict[str, Any] = {
            "round": round_num,
            "num_clients": len(selected),
            "num_aggregated": len(updates),
            "num_byzantine_flagged": num_flagged,
        }
        record.update({f"val_{k}": v for k, v in metrics.items()})
        record.update(
            _evaluate_capabilities(global_model, capability_val_loaders, objective, torch_device)
        )
        logger.log(record)
        reputation_logger.log(
            {
                "round": round_num,
                "reputation": tracker.snapshot(),
                # Logged separately so the H1/H2 tension (noisier DP clients look
                # less consistent, and are down-weighted for it) is measurable.
                "consistency": tracker.consistency_snapshot(),
            }
        )
        history.append(record)

        selector = metrics["roc_auc"]
        if np.isnan(selector):
            selector = metrics["f1"]
        if selector > best_score:
            best_score = selector
            torch.save(global_state, run_dir / "best_global_model.pt")

    return history
