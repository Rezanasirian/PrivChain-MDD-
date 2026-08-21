"""Unit tests for federated-distillation loss (Phase 4, H2)."""

from __future__ import annotations

import pytest
import torch

from privchain.config import (
    DistillationConfig,
    EncoderConfig,
    FusionConfig,
    HeadConfig,
    ModelConfig,
)
from privchain.federated.distillation import (
    capability_masked_anchor,
    distillation_loss,
    synthesize_anchors,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel


def test_loss_is_minimized_when_student_matches_teacher() -> None:
    teacher = torch.tensor([2.0, -1.5, 0.5, -3.0])
    matched = distillation_loss(teacher.clone(), teacher, temperature=2.0)
    mismatched = distillation_loss(-teacher, teacher, temperature=2.0)
    assert matched < mismatched


def test_gradient_flows_only_into_student() -> None:
    student = torch.zeros(3, requires_grad=True)
    teacher = torch.tensor([3.0, -3.0, 1.0], requires_grad=True)
    loss = distillation_loss(student, teacher, temperature=1.5)
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) > 0
    # Teacher is detached inside the loss -> no gradient leaks into it.
    assert teacher.grad is None


def test_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError):
        distillation_loss(torch.zeros(2), torch.zeros(2), temperature=0.0)


def _model() -> MultimodalDepressionModel:
    config = ModelConfig(
        encoder=EncoderConfig(type="mean", hidden_dim=4, out_dim=4, dropout=0.0),
        fusion=FusionConfig(hidden_dim=8, dropout=0.0),
        head=HeadConfig(hidden_dim=4, dropout=0.0),
        use_phq_regression=False,
    )
    return MultimodalDepressionModel({"audio": 3, "video": 2, "text": 5}, config)


def test_data_free_anchor_teacher_is_full_and_student_is_masked() -> None:
    teacher = _model().eval()
    config = DistillationConfig(
        mode="anchor",
        anchor_batch_size=4,
        anchor_sequence_length=2,
        anchor_optimization_steps=2,
    )
    anchor = synthesize_anchors(
        teacher,
        config,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )
    assert all(torch.all(mask == 1) for mask in anchor["presence"].values())

    student_anchor = capability_masked_anchor(anchor, (1, 0, 1))
    assert torch.all(student_anchor["presence"]["video"] == 0)
    assert torch.all(student_anchor["presence"]["audio"] == 1)

    student = _model()
    teacher_logit = teacher(anchor)["logit"].detach()
    loss = distillation_loss(student(student_anchor)["logit"], teacher_logit, 2.0)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in student.parameters()
    )
