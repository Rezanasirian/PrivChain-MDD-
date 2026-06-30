"""Shared training objective and evaluation (Phases 1–2).

Factored out of the centralized trainer so the federated clients reuse exactly
the same loss and evaluation logic: a binary cross-entropy classification loss
plus an optional (normalized) PHQ-8 regression term.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from privchain.data.mock_daic_woz import Batch, Sample
from privchain.eval.metrics import binary_classification_metrics


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor in a collated batch to ``device``.

    Args:
        batch: A collated batch.
        device: Target device.

    Returns:
        A batch with all tensors on ``device``.
    """
    return {key: value.to(device) for key, value in batch.items()}  # type: ignore[return-value]


class DepressionObjective:
    """Multi-task loss: BCE on the binary head + optional PHQ-8 MSE.

    Args:
        phq8_max: Maximum PHQ-8 score, used to normalize the regression target.
        phq_loss_weight: Weight on the PHQ-8 regression MSE term (0 disables it).
    """

    def __init__(self, phq8_max: int, phq_loss_weight: float) -> None:
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.phq8_max = float(phq8_max)
        self.phq_loss_weight = phq_loss_weight

    def __call__(self, outputs: dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        """Compute the combined loss for one batch.

        Args:
            outputs: Model outputs (``logit`` and optionally ``phq_pred``).
            batch: The corresponding collated batch (labels/scores).

        Returns:
            Scalar loss tensor.
        """
        loss = self.bce(outputs["logit"], batch["label"].float())
        if "phq_pred" in outputs and self.phq_loss_weight > 0:
            target = batch["phq8_score"].float() / self.phq8_max
            loss = loss + self.phq_loss_weight * self.mse(outputs["phq_pred"], target)
        return loss


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Sample],
    objective: DepressionObjective,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model, returning classification metrics plus mean loss.

    Args:
        model: The model to evaluate.
        loader: Evaluation DataLoader.
        objective: The loss object (for the reported ``loss``).
        device: Device to run on.

    Returns:
        Metric mapping including ``f1``, ``roc_auc``, ``accuracy``, ``loss``.
    """
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    count = 0
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch)
        total_loss += float(objective(outputs, batch).item())
        count += 1
        all_scores.append(torch.sigmoid(outputs["logit"]).cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

    metrics = binary_classification_metrics(np.concatenate(all_scores), np.concatenate(all_labels))
    metrics["loss"] = total_loss / max(count, 1)
    return metrics
