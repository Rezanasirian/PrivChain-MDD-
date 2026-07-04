"""Federated distillation for missing-modality clients (Phase 4, objective H2).

A missing-modality client (e.g. text-only) cannot learn cross-modal dependencies
from its own data alone. Federated distillation transfers that knowledge: the
frozen **global model at the start of the round** is the teacher, and each such
client's local objective gains a soft-target term matching the teacher's
predictions on the client's own (capability-masked) batches. Because the global
model's encoders are kept clean by capability-aware aggregation, its predictions
carry the cross-modal signal the client is missing.

This is the response-based (logit) form of knowledge distillation, adapted to the
binary depression head; the temperature ``T`` softens both sides and the term is
scaled by ``T²`` to keep gradient magnitudes comparable to the hard-label loss.
"""

from __future__ import annotations

import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def distillation_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Soft-target KD loss between a student and a (frozen) teacher logit.

    Args:
        student_logit: The student's raw classification logits ``(B,)``.
        teacher_logit: The teacher's raw classification logits ``(B,)`` (detached
            internally, so gradients never flow into the teacher).
        temperature: Softening temperature ``T > 0``.

    Returns:
        A scalar distillation loss (temperature-scaled BCE on soft targets).

    Raises:
        ValueError: If ``temperature`` is not positive.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft_target = torch.sigmoid(teacher_logit.detach() / temperature)
    return binary_cross_entropy_with_logits(student_logit / temperature, soft_target) * (
        temperature * temperature
    )
