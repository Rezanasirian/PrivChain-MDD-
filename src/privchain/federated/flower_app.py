"""Flower (``flwr``) backend for the heterogeneous federated baseline (Phase 2).

The thesis names **Flower** as the federated-orchestration framework. This module
adapts the framework-agnostic :class:`~privchain.federated.client.FederatedClient`
to Flower's ``NumPyClient`` and runs a FedAvg simulation, so the exact same local
training/aggregation logic used by the in-house simulator runs under Flower.

``flwr`` is imported lazily (it is an optional dependency); importing this module
does not require it. Install with ``pip install flwr`` and run via
``scripts/run_federated.py --backend flower``.

NOTE: Targets the Flower simulation API (``flwr.simulation.start_simulation`` with
``NumPyClient``). It has NOT been executed in this offline environment — validate
against your installed ``flwr`` version. See ADR-0003.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from privchain.config import ModelConfig
from privchain.data.mock_daic_woz import Sample
from privchain.federated.client import FederatedClient
from privchain.federated.partition import ClientPartition
from privchain.federated.simulation import build_federated_clients

if TYPE_CHECKING:  # pragma: no cover - typing only
    from privchain.fusion.baseline_model import MultimodalDepressionModel


def state_to_ndarrays(model: torch.nn.Module) -> list[np.ndarray]:
    """Extract a model's parameters as a list of NumPy arrays (Flower format)."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def ndarrays_to_state(
    model: torch.nn.Module, arrays: list[np.ndarray]
) -> OrderedDict[str, torch.Tensor]:
    """Rebuild a ``state_dict`` from Flower NumPy arrays using the model's keys."""
    keys = list(model.state_dict().keys())
    return OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, arrays))


def run_flower_simulation(
    base_dataset: Any,
    partitions: list[ClientPartition],
    val_loader: DataLoader[Sample],
    *,
    input_dims: dict[str, int],
    model_config: ModelConfig,
    global_model: "MultimodalDepressionModel",
    num_rounds: int,
    clients_per_round: int,
    batch_size: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    phq8_max: int,
    phq_loss_weight: float,
    seed: int,
    device: str = "cpu",
) -> Any:
    """Run a Flower FedAvg simulation over the heterogeneous clients.

    Args:
        base_dataset: Underlying dataset shared by clients.
        partitions: Per-client partitions (indices + capability).
        val_loader: Held-out validation loader for centralized evaluation.
        input_dims: Per-modality input feature dims.
        model_config: Model configuration.
        global_model: A model instance used to seed/initialize global params.
        num_rounds: Number of federated rounds.
        clients_per_round: Clients sampled per round.
        batch_size: Local batch size.
        local_epochs: Local epochs per round.
        learning_rate: Local optimizer learning rate.
        weight_decay: Local optimizer weight decay.
        phq8_max: Max PHQ-8 score.
        phq_loss_weight: PHQ-8 regression weight.
        seed: Base seed.
        device: Torch device string.

    Returns:
        The Flower ``History`` object returned by ``start_simulation``.

    Raises:
        ImportError: If ``flwr`` is not installed.
    """
    try:
        import flwr as fl
    except ImportError as exc:  # pragma: no cover - exercised only without flwr
        raise ImportError(
            "The Flower backend requires 'flwr'. Install it with `pip install flwr` "
            "or use the in-house simulator backend (`--backend sim`)."
        ) from exc

    clients = build_federated_clients(
        base_dataset,
        partitions,
        input_dims=input_dims,
        model_config=model_config,
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        phq8_max=phq8_max,
        phq_loss_weight=phq_loss_weight,
        seed=seed,
        device=device,
    )
    clients_by_cid: dict[str, FederatedClient] = {str(c.client_id): c for c in clients}

    class _FlowerClient(fl.client.NumPyClient):
        def __init__(self, fed_client: FederatedClient) -> None:
            self._fc = fed_client

        def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
            return state_to_ndarrays(self._fc.model)

        def fit(
            self, parameters: list[np.ndarray], config: dict[str, Any]
        ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
            state = ndarrays_to_state(self._fc.model, parameters)
            updated, num_samples = self._fc.fit(state)
            self._fc.set_parameters(updated)
            return state_to_ndarrays(self._fc.model), num_samples, {}

        def evaluate(
            self, parameters: list[np.ndarray], config: dict[str, Any]
        ) -> tuple[float, int, dict[str, Any]]:
            state = ndarrays_to_state(self._fc.model, parameters)
            metrics = self._fc.evaluate(state, val_loader)
            return float(metrics["loss"]), self._fc.num_samples, {
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }

    def client_fn(cid: str) -> fl.client.Client:
        return _FlowerClient(clients_by_cid[cid]).to_client()

    initial_parameters = fl.common.ndarrays_to_parameters(state_to_ndarrays(global_model))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=clients_per_round / max(len(clients), 1),
        min_fit_clients=clients_per_round,
        min_available_clients=len(clients),
        initial_parameters=initial_parameters,
    )
    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
