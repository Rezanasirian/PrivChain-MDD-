"""Dataset and linear model for the published 24-D CTD reference baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from privchain.data.mock_daic_woz import MODALITIES, Batch, Sample
from privchain.fusion.base import DepressionModelBase

CTD_REFERENCE_DIM = 24


class CtdReferenceDataset(Dataset[Sample]):
    """Read leakage-safe session representations emitted by the reference code.

    CTD is carried in the ``audio`` slot solely to reuse the project's batch,
    federated-client and DP contracts. Video and text are explicitly absent.
    The reference artifacts contain binary labels but not PHQ-8 scores, so this
    dataset is classification-only and must be used with ``phq_loss_weight=0``.
    """

    feature_dims = {"audio": CTD_REFERENCE_DIM, "video": 1, "text": 1}
    capability = (1, 0, 0)

    def __init__(
        self,
        artifact_dir: Path | str,
        split: str,
        *,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        path = Path(artifact_dir) / f"ctd_{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing CTD reference artifact: {path}")
        with np.load(path) as payload:
            self.session_ids = payload["session_id"].astype(np.int64)
            self.features = payload["emb"].astype(np.float32)
            self.labels = payload["label"].astype(np.int64)
        expected = (len(self.labels), CTD_REFERENCE_DIM)
        if self.features.shape != expected:
            raise ValueError(f"expected CTD features shaped {expected}, got {self.features.shape}")
        if len(self.session_ids) != len(self.labels):
            raise ValueError("CTD session IDs and labels have different lengths")
        if not np.isfinite(self.features).all():
            raise ValueError("CTD reference features contain non-finite values")
        if not set(self.labels.tolist()) <= {0, 1}:
            raise ValueError("CTD reference labels must be binary")
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("feature_mean and feature_std must be supplied together")
        if feature_mean is not None and feature_std is not None:
            mean = np.asarray(feature_mean, dtype=np.float32)
            std = np.asarray(feature_std, dtype=np.float32)
            if mean.shape != (CTD_REFERENCE_DIM,) or std.shape != (CTD_REFERENCE_DIM,):
                raise ValueError("CTD normalization vectors must have width 24")
            self.features = (self.features - mean) / np.maximum(std, 1e-6)

    def __len__(self) -> int:
        """Return the number of participant sessions."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Sample:
        """Return one session in the shared project sample format."""
        return Sample(
            audio=torch.from_numpy(self.features[index]).unsqueeze(0),
            video=torch.zeros((1, 1), dtype=torch.float32),
            text=torch.zeros((1, 1), dtype=torch.float32),
            presence={
                modality: torch.tensor(int(modality == "audio"), dtype=torch.long)
                for modality in MODALITIES
            },
            phq8_score=torch.tensor(0, dtype=torch.long),
            label=torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


class CtdLinearClassifier(DepressionModelBase):
    """A single logistic layer matching the published CTD classifier family."""

    def __init__(self) -> None:
        super().__init__()
        self.input_dims = dict(CtdReferenceDataset.feature_dims)
        classifier = nn.Linear(CTD_REFERENCE_DIM, 1)
        # Logistic regression is convex, so random initialization adds variance
        # without adding capacity. A zero start also keeps Central/FedAvg/DP
        # comparisons from charging an arm for a lucky random orientation.
        nn.init.zeros_(classifier.weight)
        nn.init.zeros_(classifier.bias)
        self.encoders = nn.ModuleDict({"audio": classifier})

    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return one depression logit per session."""
        effective_presence = batch["presence"] if presence is None else presence
        if not torch.all(effective_presence["audio"] == 1):
            raise ValueError("CTD classifier requires the CTD/audio capability")
        features = batch["audio"][:, 0, :]
        return {"logit": self.encoders["audio"](features).squeeze(-1)}
