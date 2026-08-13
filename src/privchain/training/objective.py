"""Shared training objective and evaluation (Phases 1–2).

Factored out of the centralized trainer so the federated clients reuse exactly
the same loss and evaluation logic: a binary cross-entropy classification loss
plus an optional (normalized) PHQ-8 regression term.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
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
    moved = {key: cast("torch.Tensor", value).to(device) for key, value in batch.items()}
    return cast("Batch", moved)


class DepressionObjective:
    """Multi-task loss: BCE on the binary head + optional PHQ-8 MSE.

    Args:
        phq8_max: Maximum PHQ-8 score, used to normalize the regression target.
        phq_loss_weight: Weight on the PHQ-8 regression MSE term (0 disables it).
        pos_weight: Optional weight on the positive class in the BCE term,
            counteracting DAIC-WOZ's ~28%-positive imbalance. Typically
            ``n_negative / n_positive`` over the training split; ``None`` leaves
            the loss unweighted.
    """

    def __init__(
        self, phq8_max: int, phq_loss_weight: float, pos_weight: float | None = None
    ) -> None:
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=None if pos_weight is None else torch.tensor(float(pos_weight))
        )
        self.mse = nn.MSELoss()
        self.phq8_max = float(phq8_max)
        self.phq_loss_weight = phq_loss_weight
        self.pos_weight = pos_weight

    def to(self, device: torch.device) -> DepressionObjective:
        """Move the loss's internal buffers (``pos_weight``) to ``device``.

        Args:
            device: Target device.

        Returns:
            The same instance, for chaining.
        """
        self.bce = self.bce.to(device)
        return self

    def __call__(self, outputs: dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        """Compute the combined loss for one batch.

        Args:
            outputs: Model outputs (``logit`` and optionally ``phq_pred``).
            batch: The corresponding collated batch (labels/scores).

        Returns:
            Scalar loss tensor.
        """
        loss: torch.Tensor = self.bce(outputs["logit"], batch["label"].float())
        if "phq_pred" in outputs and self.phq_loss_weight > 0:
            target = batch["phq8_score"].float() / self.phq8_max
            mse = cast("torch.Tensor", self.mse(outputs["phq_pred"], target))
            loss = loss + self.phq_loss_weight * mse
        return loss


def evaluate_with_selected_threshold(
    model: nn.Module,
    selection_loader: DataLoader[Sample],
    report_loader: DataLoader[Sample],
    objective: DepressionObjective,
    device: torch.device,
) -> dict[str, float]:
    """Pick the decision threshold on one split, then report on another.

    Choosing the F1-maximizing threshold on the same data the F1 is reported on
    is circular and inflates the result. This selects on ``selection_loader``
    and applies that fixed threshold to ``report_loader``.

    Both the private and non-private arms must call this, or their F1 values are
    not comparable: a tuned threshold against a fixed 0.5 favours whichever arm
    got the tuning (ADR-0015).

    Args:
        model: The trained model.
        selection_loader: Split the threshold is chosen on (never reported).
        report_loader: Split the returned metrics describe.
        objective: The loss object (for the reported ``loss``).
        device: Device to run on.

    Returns:
        Metrics on ``report_loader`` at the selected threshold.
    """
    chosen = evaluate_model(model, selection_loader, objective, device, threshold=None)
    return evaluate_model(model, report_loader, objective, device, threshold=chosen["threshold"])


@torch.no_grad()
def collect_scores(
    model: nn.Module, loader: DataLoader[Sample], device: torch.device
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Return per-sample predicted scores and labels, in loader order.

    Aggregate metrics cannot support a confidence interval; the per-sample scores
    can. Keeping them lets a run report how far its number would move on a
    different sample of participants, and lets two arms be compared *paired* on
    the same sessions (ADR-0020).

    The scores are positionally aligned with the split, and carry no participant
    identifier — an artifact written from them cannot be linked back to a person.

    Args:
        model: The trained model.
        loader: Evaluation DataLoader; must not shuffle if the order matters.
        device: Device to run on.

    Returns:
        ``(scores, labels)``, each of shape ``(N,)``.
    """
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        scores.append(torch.sigmoid(model(batch)["logit"]).cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return (
        np.concatenate(scores).astype(np.float64),
        np.concatenate(labels).astype(np.int_),
    )


def positive_class_weight(loader: DataLoader[Sample]) -> float | None:
    """Compute ``n_negative / n_positive`` over a loader's labels.

    This is measured from the data rather than configured, so it stays correct
    across splits, folds, and federated client shards.

    Args:
        loader: The training DataLoader to count labels over.

    Returns:
        The positive-class weight, or ``None`` when either class is absent (in
        which case weighting is undefined and the loss is left unweighted).
    """
    positives = 0
    total = 0
    for batch in loader:
        labels = batch["label"]
        positives += int(labels.sum().item())
        total += int(labels.numel())
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return None
    return negatives / positives


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Sample],
    objective: DepressionObjective,
    device: torch.device,
    threshold: float | None = 0.5,
) -> dict[str, float]:
    """Evaluate a model, returning classification metrics plus mean loss.

    Args:
        model: The model to evaluate.
        loader: Evaluation DataLoader.
        objective: The loss object (for the reported ``loss``).
        device: Device to run on.
        threshold: Decision threshold, or ``None`` to pick the F1-maximizing one
            from the scores. See
            :func:`~privchain.eval.metrics.best_f1_threshold`.

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

    metrics = binary_classification_metrics(
        np.concatenate(all_scores), np.concatenate(all_labels), threshold=threshold
    )
    metrics["loss"] = total_loss / max(count, 1)
    return metrics
