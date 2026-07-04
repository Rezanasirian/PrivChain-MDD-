"""Unit tests for federated-distillation loss (Phase 4, H2)."""

from __future__ import annotations

import pytest
import torch

from privchain.federated.distillation import distillation_loss


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
