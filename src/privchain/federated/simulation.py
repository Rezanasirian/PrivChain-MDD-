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
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from privchain.chain_client import LedgerClient, record_round
from privchain.config import AggregationConfig, ModelConfig
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.federated.aggregation import ClientUpdate, capability_aware_aggregate, fedavg
from privchain.federated.capability import is_missing_any
from privchain.federated.client import ClientDPConfig, FederatedClient, PrivacySpend
from privchain.federated.distillation import synthesize_anchors
from privchain.federated.partition import ClientPartition, ModalityMaskedDataset
from privchain.federated.reputation import ReputationTracker
from privchain.federated.robust import flag_byzantine_updates
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.training.experiment import JsonlMetricLogger
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    positive_class_weight,
)

#: How a client weights its BCE term. ``pooled_oracle`` is a control that
#: deliberately crosses the federated boundary; see ADR-0026.
ClassWeightMode = Literal["off", "per_shard", "pooled_oracle"]


@dataclass
class _BestRoundTracker:
    """Track the best round on the selection split, and when to stop.

    Federated arms were previously trained for a fixed round budget while the
    centralized baseline stopped at its best epoch. Comparing them then charged
    federation for a difference in schedule rather than in method (the same trap
    ADR-0013 found in the DP arm), so every arm now stops on the same rule.

    Args:
        patience: Rounds without improvement before stopping; ``None`` disables.
    """

    patience: int | None
    best_score: float = -float("inf")
    best_round: int = 0
    stale: int = 0

    def update(self, score: float, round_num: int) -> bool:
        """Record a round's selection score.

        Args:
            score: The selection-split metric for this round.
            round_num: 1-based round index.

        Returns:
            ``True`` when this round is the new best (the caller should
            checkpoint), ``False`` otherwise.
        """
        if score > self.best_score:
            self.best_score, self.best_round, self.stale = score, round_num, 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether patience has been exhausted."""
        return self.patience is not None and self.stale >= self.patience


def _round_work(selected: Sequence[FederatedClient]) -> dict[str, int]:
    """Sum the local work the selected clients performed this round.

    A shard smaller than ``batch_size`` collapses to one batch, so a round can
    carry far fewer optimizer steps than its budget suggests. Logging the counts
    turns that from something inferred off shard sizes into something recorded,
    and lets a sweep hold total steps fixed while varying how they are grouped.

    Args:
        selected: The clients whose ``fit`` was just called.

    Returns:
        Per-round ``optimizer_steps``, ``num_local_batches`` and
        ``examples_seen`` totals.
    """
    return {
        "optimizer_steps": sum(c.last_work.optimizer_steps for c in selected),
        "num_local_batches": sum(c.last_work.num_batches for c in selected),
        "examples_seen": sum(c.last_work.examples_seen for c in selected),
    }


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
    client_dp: ClientDPConfig | None = None,
    class_weight_mode: ClassWeightMode = "off",
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
        client_dp: Optional client-side DP-SGD configuration. Each constructed
            client receives an independently seeded accountant/mechanism.
        class_weight_mode: How each client weights its BCE term.
            ``"off"`` leaves the loss unweighted. ``"per_shard"`` uses
            ``n_neg / n_pos`` measured on that client's own partition, which is
            all a real client can observe, and mirrors ``train.class_weighting``
            in the centralized arm — without it the arms train under different
            losses and the comparison charges federation for the difference
            (ADR-0026). ``"pooled_oracle"`` gives every client one weight
            measured across all shards; that crosses the federated boundary and
            ADR-0026 rejected it as an architecture, so it exists only as an
            experimental control for how much per-shard weight noise costs. A
            shard holding a single class is left unweighted either way.

    Returns:
        A list of constructed clients (skipping any empty partition).
    """
    clients: list[FederatedClient] = []
    # The oracle weight is measured once over every shard's labels, which is the
    # federated-boundary violation that makes it a control rather than a design.
    pooled_weight = (
        positive_class_weight(
            DataLoader(
                Subset(base_dataset, [i for p in partitions for i in p.indices]),
                batch_size=batch_size,
                collate_fn=collate_fn,
            )
        )
        if class_weight_mode == "pooled_oracle"
        else None
    )
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
        # Counted on an unshuffled pass over the same shard, so the weight a
        # client trains under is derived only from data it actually holds.
        if class_weight_mode == "per_shard":
            pos_weight = positive_class_weight(
                DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
            )
        else:
            pos_weight = pooled_weight
        clients.append(
            FederatedClient(
                partition.client_id,
                partition.capability,
                model,
                loader,
                pos_weight=pos_weight,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                phq8_max=phq8_max,
                phq_loss_weight=phq_loss_weight,
                device=device,
                dp=(
                    client_dp
                    if client_dp is None
                    else ClientDPConfig(
                        target_epsilons=client_dp.target_epsilons,
                        delta=client_dp.delta,
                        max_grad_norm=client_dp.max_grad_norm,
                        batch_size=batch_size,
                        num_rounds=client_dp.num_rounds,
                        seed=client_dp.seed + partition.client_id,
                        backend=client_dp.backend,
                    )
                ),
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
    early_stopping_patience: int | None = None,
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
        early_stopping_patience: Rounds without a selection-split improvement
            before stopping; ``None`` runs the full budget.

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
    tracker = _BestRoundTracker(early_stopping_patience)

    global_state: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items()
    )
    k_per_round = min(clients_per_round, len(clients))
    started = time.perf_counter()
    cumulative_steps = 0
    cumulative_examples = 0

    for round_num in range(1, num_rounds + 1):
        rng = np.random.default_rng(seed + round_num)
        selected_idx = rng.choice(len(clients), size=k_per_round, replace=False)
        selected = [clients[int(i)] for i in selected_idx]

        states: list[OrderedDict[str, torch.Tensor]] = []
        weights: list[float] = []
        for client in selected:
            updated, num_samples, _spend = client.fit(global_state)
            states.append(updated)
            weights.append(float(num_samples))

        global_state = fedavg(states, weights)
        global_model.load_state_dict(global_state)

        work = _round_work(selected)
        cumulative_steps += work["optimizer_steps"]
        cumulative_examples += work["examples_seen"]

        metrics = evaluate_model(global_model, val_loader, objective, torch_device)
        record: dict[str, Any] = {"round": round_num, "num_clients": len(selected)}
        record.update(work)
        record["cumulative_optimizer_steps"] = cumulative_steps
        record["cumulative_examples_seen"] = cumulative_examples
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        record.update({f"val_{k}": v for k, v in metrics.items()})
        logger.log(record)
        history.append(record)

        selector = metrics["roc_auc"]
        if np.isnan(selector):
            selector = metrics["f1"]
        if tracker.update(selector, round_num):
            torch.save(global_state, run_dir / "best_global_model.pt")
        elif tracker.should_stop:
            break

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
    privacy_spend: dict[str, PrivacySpend],
    registered: set[str],
    aggregated_client_ids: list[str] | None = None,
) -> None:
    """Write a round's subgraph, consumed budget, and reputation to the ledger.

    Args:
        ledger: The audit ledger client.
        round_num: The current round.
        updates: The client updates that were aggregated this round.
        tracker: The reputation tracker (its snapshot is logged).
        privacy_spend: Per-client spend returned by mechanisms executed this round.
        registered: Set of already-registered client IDs (updated in place).
        aggregated_client_ids: IDs retained by the Byzantine filter. Privacy
            spend is still recorded for every participant because their local
            mechanism already executed.
    """
    participants = [(str(u.client_id), u.capability) for u in updates]
    reputation = {str(cid): groups for cid, groups in tracker.snapshot().items()}
    consumed = {
        client_id: {
            "incremental": spend.incremental,
            "cumulative": spend.cumulative,
        }
        for client_id, spend in privacy_spend.items()
    }
    record_round(
        ledger,
        round_num=round_num,
        participants=participants,
        reputation=reputation,
        consumed_epsilon=consumed or None,
        subgraph_client_ids=aggregated_client_ids,
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
    device: str = "cpu",
    early_stopping_patience: int | None = None,
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
        device: Torch device string.
        early_stopping_patience: Rounds without a selection-split improvement
            before stopping; ``None`` runs the full budget.

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
    tracker_best = _BestRoundTracker(early_stopping_patience)

    global_state: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items()
    )
    k_per_round = min(clients_per_round, len(clients))
    registered: set[str] = set()
    started = time.perf_counter()
    cumulative_steps = 0
    cumulative_examples = 0

    for round_num in range(1, num_rounds + 1):
        rng = np.random.default_rng(seed + round_num)
        selected_idx = rng.choice(len(clients), size=k_per_round, replace=False)
        selected = [clients[int(i)] for i in selected_idx]

        if teacher is not None:
            teacher.load_state_dict(global_state)
            teacher.eval()
        anchor = None
        if teacher is not None and distill.mode in {"anchor", "random"}:
            anchor = synthesize_anchors(
                teacher,
                distill,
                device=torch_device,
                generator=torch.Generator(device=torch_device).manual_seed(seed + round_num),
            )

        updates: list[ClientUpdate] = []
        round_spend: dict[str, PrivacySpend] = {}
        for client in selected:
            use_distill = teacher is not None and (
                distill.apply_to == "all" or is_missing_any(client.capability)
            )
            updated, num_samples, spend = client.fit(
                global_state,
                teacher=teacher if use_distill else None,
                distill_weight=distill.weight if use_distill else 0.0,
                distill_temperature=distill.temperature,
                anchor=anchor if use_distill else None,
                distill_steps=distill.student_steps,
            )
            if spend is not None:
                round_spend[str(client.client_id)] = spend
            updates.append(ClientUpdate(client.client_id, client.capability, num_samples, updated))

        # Preserve all participants for privacy accounting: a rejected update has
        # still executed its local DP mechanism and consumed budget.
        audited_updates = list(updates)

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
            if not round_spend:
                raise ValueError("a ledger audit cannot record epsilon without client DP")
            _record_round_to_ledger(
                ledger,
                round_num,
                audited_updates,
                tracker,
                round_spend,
                registered,
                aggregated_client_ids=[str(update.client_id) for update in updates],
            )

        metrics = evaluate_model(global_model, val_loader, objective, torch_device)
        work = _round_work(selected)
        cumulative_steps += work["optimizer_steps"]
        cumulative_examples += work["examples_seen"]
        record: dict[str, Any] = {
            "round": round_num,
            "num_clients": len(selected),
            "num_aggregated": len(updates),
            "num_byzantine_flagged": num_flagged,
            **work,
            "cumulative_optimizer_steps": cumulative_steps,
            "cumulative_examples_seen": cumulative_examples,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
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
        if tracker_best.update(selector, round_num):
            torch.save(global_state, run_dir / "best_global_model.pt")
        elif tracker_best.should_stop:
            break

    return history
