"""Framework-agnostic federated client (Phase 2, objective H2).

Holds a local model copy and dataset shard; ``fit`` loads the server's global
parameters, runs a few local epochs, and returns the updated parameters with the
local sample count (for FedAvg weighting). This same client backs both the
in-house simulator (:mod:`privchain.federated.simulation`) and the Flower adapter
(:mod:`privchain.federated.flower_app`).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from privchain.data.mock_daic_woz import Sample
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.training.objective import DepressionObjective, evaluate_model, move_batch_to_device


class FederatedClient:
    """A single simulated federated client.

    Args:
        client_id: Identifier for logging.
        capability: ``[audio, video, text]`` availability flags (metadata).
        model: The client's local model (architecture matches the server).
        train_loader: DataLoader over the client's shard.
        local_epochs: Local SGD epochs per round.
        learning_rate: Local optimizer learning rate.
        weight_decay: Local optimizer weight decay.
        phq8_max: Max PHQ-8 score (for the objective).
        phq_loss_weight: Weight on the PHQ-8 regression term.
        device: Torch device string.
    """

    def __init__(
        self,
        client_id: int,
        capability: tuple[int, int, int],
        model: MultimodalDepressionModel,
        train_loader: DataLoader[Sample],
        *,
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        phq8_max: int,
        phq_loss_weight: float,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.capability = capability
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.objective = DepressionObjective(phq8_max, phq_loss_weight)

    @property
    def num_samples(self) -> int:
        """Number of local training samples (FedAvg weight)."""
        return len(self.train_loader.dataset)  # type: ignore[arg-type]

    def set_parameters(self, state: OrderedDict[str, torch.Tensor]) -> None:
        """Load a (server) ``state_dict`` into the local model.

        Args:
            state: The global model parameters to adopt.
        """
        self.model.load_state_dict(state)

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return a CPU copy of the local model ``state_dict``.

        Returns:
            The local parameters.
        """
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items())

    def fit(
        self, global_state: OrderedDict[str, torch.Tensor]
    ) -> tuple[OrderedDict[str, torch.Tensor], int]:
        """Adopt global params, train locally, and return updated params.

        Args:
            global_state: The server's current global parameters.

        Returns:
            ``(updated_state, num_samples)``.
        """
        self.set_parameters(global_state)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.model.train()
        for _ in range(self.local_epochs):
            for raw_batch in self.train_loader:
                batch = move_batch_to_device(raw_batch, self.device)
                optimizer.zero_grad()
                loss = self.objective(self.model(batch), batch)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), self.num_samples

    def evaluate(
        self, global_state: OrderedDict[str, torch.Tensor], loader: DataLoader[Sample]
    ) -> dict[str, float]:
        """Evaluate the given global params on a loader.

        Args:
            global_state: Parameters to evaluate.
            loader: Evaluation DataLoader.

        Returns:
            Classification metrics including ``f1``, ``roc_auc``, ``loss``.
        """
        self.set_parameters(global_state)
        return evaluate_model(self.model, loader, self.objective, self.device)
