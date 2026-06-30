"""Centralized training loop for the multimodal baseline (Phase 1, H4/H5).

Trains :class:`~privchain.fusion.baseline_model.MultimodalDepressionModel` with a
binary cross-entropy objective plus an optional (normalized) PHQ-8 regression
term, evaluating F1/ROC-AUC each epoch and logging to an experiment run dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from privchain.data.mock_daic_woz import Batch, Sample
from privchain.eval.metrics import binary_classification_metrics
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.training.experiment import JsonlMetricLogger


class CentralizedTrainer:
    """Train and evaluate the centralized multimodal baseline.

    Args:
        model: The multimodal model to optimize.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2).
        phq8_max: Maximum PHQ-8 score, used to normalize the regression target.
        phq_loss_weight: Weight on the PHQ-8 regression MSE term.
        device: Torch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        model: MultimodalDepressionModel,
        *,
        learning_rate: float,
        weight_decay: float,
        phq8_max: int,
        phq_loss_weight: float,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.phq8_max = float(phq8_max)
        self.phq_loss_weight = phq_loss_weight

    def _to_device(self, batch: Batch) -> Batch:
        """Move every tensor in a batch to the configured device."""
        return {key: value.to(self.device) for key, value in batch.items()}  # type: ignore[return-value]

    def _compute_loss(self, outputs: dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
        """Combine the classification BCE and optional PHQ-8 regression MSE."""
        loss = self.bce(outputs["logit"], batch["label"].float())
        if "phq_pred" in outputs and self.phq_loss_weight > 0:
            target = batch["phq8_score"].float() / self.phq8_max
            loss = loss + self.phq_loss_weight * self.mse(outputs["phq_pred"], target)
        return loss

    def train_epoch(self, loader: DataLoader[Sample]) -> float:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.

        Returns:
            Mean per-batch training loss.
        """
        self.model.train()
        total = 0.0
        count = 0
        for raw_batch in loader:
            batch = self._to_device(raw_batch)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self._compute_loss(outputs, batch)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[Sample]) -> dict[str, float]:
        """Evaluate the model, returning classification metrics + mean loss.

        Args:
            loader: Validation/test DataLoader.

        Returns:
            Metric mapping including ``f1``, ``roc_auc``, ``accuracy``, ``loss``.
        """
        self.model.eval()
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        total_loss = 0.0
        count = 0
        for raw_batch in loader:
            batch = self._to_device(raw_batch)
            outputs = self.model(batch)
            total_loss += float(self._compute_loss(outputs, batch).item())
            count += 1
            probs = torch.sigmoid(outputs["logit"]).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(batch["label"].cpu().numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        metrics = binary_classification_metrics(scores, labels)
        metrics["loss"] = total_loss / max(count, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader[Sample],
        val_loader: DataLoader[Sample],
        *,
        epochs: int,
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        """Train for ``epochs`` epochs, logging metrics and the best checkpoint.

        The best checkpoint (by validation ROC-AUC, falling back to F1 when AUC
        is undefined) is saved to ``<run_dir>/best_model.pt``.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Number of epochs.
            run_dir: Experiment run directory for logs/checkpoints.

        Returns:
            Per-epoch history records.
        """
        logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
        history: list[dict[str, Any]] = []
        best_score = -float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.log(record)
            history.append(record)

            selector = val_metrics["roc_auc"]
            if np.isnan(selector):
                selector = val_metrics["f1"]
            if selector > best_score:
                best_score = selector
                torch.save(self.model.state_dict(), run_dir / "best_model.pt")

        return history
