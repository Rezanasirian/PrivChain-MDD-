"""Centralized modality dropout (Phase 1, ADR-0027).

The centralized model is trained on sessions that always carry all three
modalities, then handed to a federated population where most clients hold one or
two. That is a distribution shift the model has never seen, and it is a plausible
part of the ROC-AUC drop between the centralized and federated arms.

The fix is cheap: during training, draw a capability vector per sample and clear
the presence flags it declares absent. Fusion already treats a cleared flag as
"this branch contributes exactly zero", so no feature tensor has to be rewritten
and the batch is left untouched.

Two properties the implementation has to keep:

* **Per sample, not per batch.** One capability for a whole batch would correlate
  the dropout with whatever else that batch shares, and at batch size 32 on 107
  sessions there are only a handful of draws per epoch to average over.
* **Training only.** Evaluation must see what the split actually holds, or the
  reported number describes a different task than the one being claimed.
"""

from __future__ import annotations

import torch

from privchain.config import CAPABILITY_MODALITIES, ModalityDropoutConfig
from privchain.data.mock_daic_woz import Batch


class ModalityDropout:
    """Draws a per-sample capability pattern and masks presence accordingly.

    Args:
        config: Validated modality-dropout configuration.
        seed: Seed for the draw, so a run is reproducible.

    Raises:
        ValueError: If the configuration is enabled but has no patterns.
    """

    def __init__(self, config: ModalityDropoutConfig, seed: int) -> None:
        config.validated()
        self.config = config
        self._generator = torch.Generator().manual_seed(seed)
        self._capabilities = torch.tensor(
            [pattern.capability for pattern in config.patterns], dtype=torch.float32
        )
        self._weights = torch.tensor(
            [pattern.fraction for pattern in config.patterns], dtype=torch.float32
        )

    def __call__(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Return presence flags for one batch, with modalities randomly hidden.

        Args:
            batch: The collated batch (read only — never mutated, so the caller
                can still evaluate the same batch unmasked).

        Returns:
            ``{modality: (B,)}`` flags: the batch's own presence, with the drawn
            pattern's absent modalities cleared. A modality the sample never had
            stays absent.
        """
        presence = batch["presence"]
        modality = next(iter(CAPABILITY_MODALITIES))
        device = presence[modality].device
        batch_size = presence[modality].shape[0]

        if not self.config.enabled:
            return dict(presence)

        drawn = torch.multinomial(
            self._weights, batch_size, replacement=True, generator=self._generator
        )
        capability = self._capabilities[drawn].to(device)  # (B, M)
        return {
            name: presence[name].to(capability.dtype) * capability[:, index]
            for index, name in enumerate(CAPABILITY_MODALITIES)
        }
