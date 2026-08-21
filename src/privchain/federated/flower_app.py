"""Flower (``flwr``) backend for the heterogeneous federated baseline (Phase 2).

The thesis names **Flower** as the federated-orchestration framework. This module
adapts the framework-agnostic :class:`~privchain.federated.client.FederatedClient`
to Flower's ``NumPyClient`` and runs a FedAvg simulation, so the exact same local
training/aggregation logic used by the in-house simulator runs under Flower.

``flwr`` is imported lazily (it is an optional dependency); importing this module
does not require it. Install with ``pip install flwr`` and run via
``scripts/run_federated.py --backend flower``.

Executed against flwr 1.33 on real DAIC-WOZ (ADR-0021). Running it exposed two
defects that could only show up at runtime: clients were built in the parent
process, so Ray had to pickle CUDA tensors into every worker, and workers were
granted no GPU. Both are fixed below, and both failed *silently* — Flower logs
the failures and proceeds with the untrained initial parameters.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from privchain.config import ModelConfig
from privchain.data.mock_daic_woz import Sample
from privchain.federated.client import FederatedClient
from privchain.federated.partition import ClientPartition
from privchain.federated.simulation import build_federated_clients

if TYPE_CHECKING:  # pragma: no cover - typing only
    from privchain.fusion.baseline_model import MultimodalDepressionModel


def state_to_ndarrays(model: torch.nn.Module) -> list[NDArray[Any]]:
    """Extract a model's parameters as a list of NumPy arrays (Flower format)."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def ndarrays_to_state(
    model: torch.nn.Module, arrays: list[NDArray[Any]]
) -> OrderedDict[str, torch.Tensor]:
    """Rebuild a ``state_dict`` from Flower NumPy arrays using the model's keys."""
    keys = list(model.state_dict().keys())
    return OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, arrays, strict=True))


def run_flower_simulation(
    base_dataset: Any,
    partitions: list[ClientPartition],
    val_loader: DataLoader[Sample],
    *,
    input_dims: dict[str, int],
    model_config: ModelConfig,
    global_model: MultimodalDepressionModel,
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
        ``(history, final_state)`` — Flower's ``History`` and the aggregated
        global ``state_dict``. The state is returned explicitly because
        ``History`` does not carry it, and a caller that assumes otherwise
        silently evaluates the untrained initial weights.

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

    # Clients are built *inside* the Ray worker, not here. Constructing them in
    # the parent and closing over the list makes Ray pickle each client — model
    # weights included — into every worker; when those tensors are on CUDA the
    # worker cannot deserialize them and Flower reports "received 0 results and N
    # failures" every round, then carries on with the untrained initial
    # parameters. That looks like a converged run rather than a total failure,
    # which is how the defect survived being written but never executed (ADR-0003).
    partitions_by_cid: dict[str, ClientPartition] = {
        str(p.client_id): p for p in partitions if p.indices
    }
    num_clients = len(partitions_by_cid)

    class _FlowerClient(fl.client.NumPyClient):  # type: ignore[misc]
        def __init__(self, fed_client: FederatedClient) -> None:
            self._fc = fed_client

        def get_parameters(self, config: dict[str, Any]) -> list[NDArray[Any]]:
            return state_to_ndarrays(self._fc.model)

        def fit(
            self, parameters: list[NDArray[Any]], config: dict[str, Any]
        ) -> tuple[list[NDArray[Any]], int, dict[str, Any]]:
            state = ndarrays_to_state(self._fc.model, parameters)
            updated, num_samples, spend = self._fc.fit(state)
            self._fc.set_parameters(updated)
            metrics: dict[str, Any] = {}
            if spend is not None:
                metrics["epsilon_composed"] = spend.cumulative["composed"]
            return state_to_ndarrays(self._fc.model), num_samples, metrics

        def evaluate(
            self, parameters: list[NDArray[Any]], config: dict[str, Any]
        ) -> tuple[float, int, dict[str, Any]]:
            state = ndarrays_to_state(self._fc.model, parameters)
            metrics = self._fc.evaluate(state, val_loader)
            return (
                float(metrics["loss"]),
                self._fc.num_samples,
                {
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                },
            )

    def client_fn(cid: str) -> fl.client.Client:
        """Construct this client's model and loader inside the Ray worker."""
        # A worker only sees a GPU if one was granted below; fall back rather
        # than fail, so a resource-starved worker still contributes.
        worker_device = device if torch.cuda.is_available() else "cpu"
        built = build_federated_clients(
            base_dataset,
            [partitions_by_cid[cid]],
            input_dims=input_dims,
            model_config=model_config,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            phq8_max=phq8_max,
            phq_loss_weight=phq_loss_weight,
            seed=seed,
            device=worker_device,
        )
        return _FlowerClient(built[0]).to_client()

    class _RecordingFedAvg(fl.server.strategy.FedAvg):  # type: ignore[misc]
        """FedAvg that keeps the last aggregate.

        Flower's ``History`` carries per-round metrics but **not** the final
        global parameters, so without this the caller has no way to get the
        trained model back and silently evaluates the initial weights instead.
        """

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.final_parameters: Any = None

        def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> Any:
            parameters, metrics = super().aggregate_fit(server_round, results, failures)
            if parameters is not None:
                self.final_parameters = parameters
            return parameters, metrics

    initial_parameters = fl.common.ndarrays_to_parameters(state_to_ndarrays(global_model))
    strategy = _RecordingFedAvg(
        fraction_fit=clients_per_round / max(num_clients, 1),
        min_fit_clients=clients_per_round,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        # Client-side evaluation is switched off: the parity check evaluates the
        # final global model centrally, and enabling it would drag the validation
        # loader through Ray's serializer for no benefit.
        fraction_evaluate=0.0,
    )
    # Flower runs each client inside a Ray worker, and a Ray worker is granted no
    # GPU unless one is requested here. Without this, a CUDA-built client cannot
    # deserialize its tensors in the worker and *every* client fails every round
    # — Flower reports "received 0 results and N failures" and carries on with the
    # untrained initial parameters, which looks like a converged run rather than a
    # crash. Requesting a fraction lets all clients share the one physical GPU.
    client_resources = {"num_cpus": 1.0, "num_gpus": 0.0}
    if torch.cuda.is_available() and device.startswith("cuda"):
        client_resources["num_gpus"] = 1.0 / max(clients_per_round, 1)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    final_state: OrderedDict[str, torch.Tensor] | None = None
    if strategy.final_parameters is not None:
        final_state = ndarrays_to_state(
            global_model, fl.common.parameters_to_ndarrays(strategy.final_parameters)
        )
    return history, final_state
