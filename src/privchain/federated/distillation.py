"""Federated distillation for missing-modality clients (Phase 4, objective H2).

A missing-modality client (e.g. text-only) cannot learn cross-modal dependencies
from its own data alone. Federated distillation transfers that knowledge: the
frozen **global model at the start of the round** is the teacher, and each such
client's local objective gains a soft-target term matching the teacher's
predictions on data-free anchors synthesized from the global model. The teacher
sees every anchor modality while a client sees the same anchor through its
capability mask (ADR-0024).

This is the response-based (logit) form of knowledge distillation, adapted to the
binary depression head; the temperature ``T`` softens both sides and the term is
scaled by ``T²`` to keep gradient magnitudes comparable to the hard-label loss.
"""

from __future__ import annotations

import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from privchain.config import DistillationConfig
from privchain.data.mock_daic_woz import MODALITIES, Batch
from privchain.fusion.baseline_model import MultimodalDepressionModel


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


def synthesize_anchors(
    teacher: MultimodalDepressionModel,
    config: DistillationConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Batch:
    """Create full-modality anchors without reading private records.

    Args:
        teacher: Frozen current global model.
        config: Anchor synthesis hyperparameters.
        device: Device on which synthesis runs.
        generator: Seeded random generator.

    Returns:
        A detached full-modality batch. Labels are dummy values and are never
        used by the KD objective.
    """
    batch_size = config.anchor_batch_size
    sequence_length = config.anchor_sequence_length
    inputs = {
        modality: torch.randn(
            batch_size,
            sequence_length,
            teacher.input_dims[modality],
            generator=generator,
            device=device,
            requires_grad=True,
        )
        for modality in MODALITIES
    }
    parameters = list(inputs.values())
    optimizer = torch.optim.Adam(parameters, lr=config.anchor_learning_rate)
    lengths = {
        modality: torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)
        for modality in MODALITIES
    }
    presence = {
        modality: torch.ones(batch_size, dtype=torch.long, device=device) for modality in MODALITIES
    }

    teacher.eval()
    for _ in range(config.anchor_optimization_steps if config.mode == "anchor" else 0):
        optimizer.zero_grad(set_to_none=True)
        batch = Batch(
            audio=inputs["audio"],
            video=inputs["video"],
            text=inputs["text"],
            audio_lengths=lengths["audio"],
            video_lengths=lengths["video"],
            text_lengths=lengths["text"],
            presence=presence,
            phq8_score=torch.zeros(batch_size, dtype=torch.long, device=device),
            label=torch.zeros(batch_size, dtype=torch.long, device=device),
        )
        probabilities = torch.sigmoid(teacher(batch)["logit"])
        eps = torch.finfo(probabilities.dtype).eps
        sample_entropy = -(
            probabilities * torch.log(probabilities.clamp_min(eps))
            + (1.0 - probabilities) * torch.log((1.0 - probabilities).clamp_min(eps))
        ).mean()
        mean_probability = probabilities.mean()
        diversity_entropy = -(
            mean_probability * torch.log(mean_probability.clamp_min(eps))
            + (1.0 - mean_probability) * torch.log((1.0 - mean_probability).clamp_min(eps))
        )
        l2 = sum(tensor.square().mean() for tensor in parameters)
        loss = (
            sample_entropy
            - config.anchor_diversity_weight * diversity_entropy
            + config.anchor_l2_weight * l2
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    return Batch(
        audio=inputs["audio"].detach(),
        video=inputs["video"].detach(),
        text=inputs["text"].detach(),
        audio_lengths=lengths["audio"],
        video_lengths=lengths["video"],
        text_lengths=lengths["text"],
        presence=presence,
        phq8_score=torch.zeros(batch_size, dtype=torch.long, device=device),
        label=torch.zeros(batch_size, dtype=torch.long, device=device),
    )


def capability_masked_anchor(anchor: Batch, capability: tuple[int, int, int]) -> Batch:
    """Return an anchor view whose presence flags match one client."""
    presence = {
        modality: torch.full_like(anchor["presence"][modality], flag)
        for modality, flag in zip(MODALITIES, capability, strict=True)
    }
    return Batch(
        audio=anchor["audio"],
        video=anchor["video"],
        text=anchor["text"],
        audio_lengths=anchor["audio_lengths"],
        video_lengths=anchor["video_lengths"],
        text_lengths=anchor["text_lengths"],
        presence=presence,
        phq8_score=anchor["phq8_score"],
        label=anchor["label"],
    )
