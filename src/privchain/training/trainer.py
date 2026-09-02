"""Centralized training loop for the multimodal baseline (Phase 1, H4/H5).

Trains any :class:`~privchain.fusion.base.DepressionModelBase` with a
binary cross-entropy objective plus an optional (normalized) PHQ-8 regression
term, evaluating F1/ROC-AUC each epoch and logging to an experiment run dir. The
loss/eval logic is shared with the federated clients via
:mod:`privchain.training.objective`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from privchain.data.mock_daic_woz import Sample
from privchain.fusion.base import DepressionModelBase
from privchain.training.experiment import JsonlMetricLogger
from privchain.training.modality_dropout import ModalityDropout
from privchain.training.objective import DepressionObjective, evaluate_model, move_batch_to_device


class CentralizedTrainer:
    """Train and evaluate the centralized multimodal baseline.

    Args:
        model: The multimodal model to optimize.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2).
        phq8_max: Maximum PHQ-8 score, used to normalize the regression target.
        phq_loss_weight: Weight on the PHQ-8 regression term.
        device: Torch device string (default ``"cpu"``).
        pos_weight: Optional positive-class weight for the BCE term; see
            :func:`~privchain.training.objective.positive_class_weight`.
        objective: Optional pre-built objective, normally from
            :func:`~privchain.training.objective.build_objective`. Config-driven
            callers pass it so the loss the config asks for (Huber vs MSE) is the
            loss that actually runs; when omitted, the MSE default is used.
        modality_dropout: Optional per-sample capability dropout, applied to
            training steps only (ADR-0027).
    """

    def __init__(
        self,
        model: DepressionModelBase,
        *,
        learning_rate: float,
        weight_decay: float,
        phq8_max: int,
        phq_loss_weight: float,
        device: str = "cpu",
        pos_weight: float | None = None,
        objective: DepressionObjective | None = None,
        modality_dropout: ModalityDropout | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.objective = (
            objective or DepressionObjective(phq8_max, phq_loss_weight, pos_weight)
        ).to(self.device)
        self.modality_dropout = modality_dropout

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
            batch = move_batch_to_device(raw_batch, self.device)
            self.optimizer.zero_grad()
            # Dropout is applied here and nowhere else: `evaluate` must see the
            # modalities the split actually holds.
            presence = self.modality_dropout(batch) if self.modality_dropout else None
            outputs = self.model(batch, presence)
            loss = self.objective(outputs, batch)
            loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    def evaluate(self, loader: DataLoader[Sample]) -> dict[str, float]:
        """Evaluate the model, returning classification metrics + mean loss.

        Args:
            loader: Validation/test DataLoader.

        Returns:
            Metric mapping including ``f1``, ``roc_auc``, ``accuracy``, ``loss``.
        """
        return evaluate_model(self.model, loader, self.objective, self.device)

    def fit(
        self,
        train_loader: DataLoader[Sample],
        val_loader: DataLoader[Sample],
        *,
        epochs: int,
        run_dir: Path,
        selection_metric: str = "roc_auc",
        early_stopping_patience: int | None = None,
        on_epoch_start: Callable[[int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Train for up to ``epochs`` epochs, logging metrics and the best checkpoint.

        The best checkpoint (by ``selection_metric``, falling back to F1 when
        that metric is undefined) is saved to ``<run_dir>/best_model.pt``.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum number of epochs.
            run_dir: Experiment run directory for logs/checkpoints.
            selection_metric: Validation metric used to select the best
                checkpoint and to drive early stopping (``"roc_auc"``/``"f1"``).
            early_stopping_patience: Stop after this many consecutive epochs
                without improvement; ``None`` trains the full ``epochs``.
            on_epoch_start: Called with the zero-based epoch before each training
                pass. The capability schedule (ADR-0028) uses it to change which
                modalities each participant is seen under; the epoch cannot be
                counted inside the dataset because DataLoader workers each hold
                their own copy.

        Returns:
            Per-epoch history records.
        """
        logger = JsonlMetricLogger(run_dir / "metrics.jsonl")
        history: list[dict[str, Any]] = []
        best_score = -float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            if on_epoch_start is not None:
                on_epoch_start(epoch - 1)
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.log(record)
            history.append(record)

            selector = val_metrics[selection_metric]
            if np.isnan(selector):
                selector = val_metrics["f1"]
            if selector > best_score:
                best_score = selector
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), run_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
                if (
                    early_stopping_patience is not None
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    break

        return history
