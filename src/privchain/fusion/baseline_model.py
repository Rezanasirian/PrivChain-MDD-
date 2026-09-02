"""Centralized multimodal depression-detection model (Phase 1, objective H4).

Wires the three per-modality encoders into the fusion module and prediction
heads:

* a **binary classification** head (depressed vs. not — the head that F1 and
  ROC-AUC are reported on), and
* an optional **PHQ-8 score regression** head (auxiliary multi-task signal, on
  by default since DAIC-WOZ ships PHQ-8 scores).

No federation and no differential privacy here — this establishes the accuracy
"ceiling" before those constraints are added in later phases.
"""

from __future__ import annotations

import torch
from torch import nn

from privchain.config import ModelConfig
from privchain.data.mock_daic_woz import MODALITIES, Batch
from privchain.encoders.audio import AudioEncoder
from privchain.encoders.text import TextEncoder
from privchain.encoders.video import VideoEncoder
from privchain.fusion.base import DepressionModelBase
from privchain.fusion.multimodal_fusion import build_fusion


class MultimodalDepressionModel(DepressionModelBase):
    """Encoders + fusion + (classification, optional regression) heads.

    Args:
        input_dims: Per-modality input feature dimensions, e.g. from
            :func:`privchain.config.modality_input_dims`.
        config: Validated model configuration.
    """

    def __init__(self, input_dims: dict[str, int], config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dims = dict(input_dims)
        # Each modality may override the shared encoder config (ADR-0014).
        encoder_configs = {modality: config.encoder_for(modality) for modality in MODALITIES}
        self.encoders = nn.ModuleDict(
            {
                "audio": AudioEncoder(input_dims["audio"], encoder_configs["audio"]),
                "video": VideoEncoder(input_dims["video"], encoder_configs["video"]),
                "text": TextEncoder(input_dims["text"], encoder_configs["text"]),
            }
        )
        modality_dims = {modality: encoder_configs[modality].out_dim for modality in MODALITIES}
        self.fusion = build_fusion(modality_dims, config.fusion)

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.out_dim, config.head.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.head.dropout),
            nn.Linear(config.head.hidden_dim, 1),
        )
        self.regressor: nn.Linear | None = (
            nn.Linear(self.fusion.out_dim, 1) if config.use_phq_regression else None
        )

    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Run a forward pass over a collated batch.

        Args:
            batch: A collated :class:`~privchain.data.mock_daic_woz.Batch`.
            presence: Optional per-modality 0/1 presence mask (Phase 2+).

        Returns:
            Dict with ``logit`` ``(B,)`` always present and ``phq_pred`` ``(B,)``
            when PHQ-8 regression is enabled.
        """
        embeddings = {
            "audio": self.encoders["audio"](batch["audio"], batch["audio_lengths"]),
            "video": self.encoders["video"](batch["video"], batch["video_lengths"]),
            "text": self.encoders["text"](batch["text"], batch["text_lengths"]),
        }
        fused = self.fusion(embeddings, batch["presence"] if presence is None else presence)

        outputs: dict[str, torch.Tensor] = {"logit": self.classifier(fused).squeeze(-1)}
        if self.regressor is not None:
            outputs["phq_pred"] = self.regressor(fused).squeeze(-1)
        return outputs
