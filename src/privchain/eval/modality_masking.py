"""Modality ablation at inference time (Phase 1, objectives H4/H5).

Answering "how much does this modality actually contribute?" requires hiding it
the same way a capability-restricted federated client would: zero the features
**and** clear the presence flag. Only clearing presence lets the features leak in
through an unmasked branch; only zeroing the features leaves fusion treating a
zero vector as a real observation.

Shrinking the input instead — the trick of loading one frame per session — is
not an ablation at all: that frame is real data, and the presence flag still says
the modality is there.

Promoted out of ``scripts/run_modality_ablation.py`` so the ablation arms of
different experiments cannot drift apart in what "absent" means.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from privchain.config import CAPABILITY_MODALITIES
from privchain.data.mock_daic_woz import Batch


class MaskedModalityModel(nn.Module):
    """Wraps a model, hiding every modality outside ``present``.

    Args:
        model: The full multimodal model.
        present: Modalities the model is allowed to see.
    """

    def __init__(self, model: nn.Module, present: frozenset[str]) -> None:
        super().__init__()
        self.model = model
        self.present = present

    def forward(
        self, batch: Batch, presence: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Blank the hidden modalities, then delegate to the wrapped model.

        Args:
            batch: A collated batch. Copied, never mutated, so the caller can
                reuse it for another ablation arm.
            presence: Presence flags to ablate; defaults to the batch's own. The
                ablation *removes* modalities, so the result is the incoming
                flags **times** the arm's mask — a sample that never had video
                must not come back marked as having it.

        Returns:
            The wrapped model's outputs.
        """
        masked: dict[str, Any] = dict(batch)
        incoming = batch["presence"] if presence is None else presence
        quality = batch.get("quality")
        masked_quality: dict[str, torch.Tensor] = dict(quality) if quality is not None else {}
        if quality is not None:
            masked["quality"] = masked_quality
        flags: dict[str, torch.Tensor] = {}
        for modality in CAPABILITY_MODALITIES:
            visible = modality in self.present
            if not visible:
                masked[modality] = torch.zeros_like(batch[modality])  # type: ignore[literal-required]
                if quality is not None:
                    # Zeroing quality clears its `valid` channel, so a segment
                    # gate sees the modality as absent rather than as present
                    # with a suspiciously flat signal.
                    masked_quality[modality] = torch.zeros_like(quality[modality])
            held = incoming[modality].to(batch["label"].device)
            flags[modality] = held.to(torch.float32) * float(visible)
        outputs: dict[str, torch.Tensor] = self.model(masked, flags)
        return outputs
