"""Per-modality DP-SGD via microbatch gradient clipping (Phase 3, objective H1).

Implements differentially-private SGD where **each modality is an independent DP
mechanism**: the per-sample gradient of each modality's parameter group is
clipped to ``C`` and perturbed with Gaussian noise scaled by that modality's own
multiplier ``σ_m`` (from :class:`~privchain.privacy.budget_allocator.PerModalityBudgetAllocator`).

This is mathematically what Opacus does (per-sample clipping + Gaussian noise +
RDP accounting); it is implemented here directly so Phase 3 runs offline without
the ``opacus`` dependency. ``opacus_engine`` documents/validates the Opacus path.

Parameter→modality grouping (by name prefix on ``MultimodalDepressionModel``):
``encoders.audio`` → audio, ``encoders.video`` → video, ``encoders.text`` → text,
everything else (fusion + heads) → ``shared``. The shared group sees all
modalities, so it conservatively takes the **largest** ``σ`` among modalities.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.training.objective import DepressionObjective, move_batch_to_device

SHARED_GROUP = "shared"


def map_parameter_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    """Group a model's parameters into per-modality groups + a shared group.

    Args:
        model: A :class:`MultimodalDepressionModel` (or compatible) instance.

    Returns:
        Mapping ``{group_name: [parameters]}`` with keys ``audio``, ``video``,
        ``text``, and ``shared``.
    """
    groups: dict[str, list[nn.Parameter]] = {"audio": [], "video": [], "text": [], SHARED_GROUP: []}
    for name, param in model.named_parameters():
        if name.startswith("encoders.audio"):
            groups["audio"].append(param)
        elif name.startswith("encoders.video"):
            groups["video"].append(param)
        elif name.startswith("encoders.text"):
            groups["text"].append(param)
        else:
            groups[SHARED_GROUP].append(param)
    return groups


def resolve_group_sigmas(modality_sigmas: dict[str, float]) -> dict[str, float]:
    """Add a ``shared``-group sigma (the max modality sigma) to the mapping.

    Args:
        modality_sigmas: ``{audio/video/text: σ_m}``.

    Returns:
        A copy with an added ``shared`` entry equal to ``max(σ_m)``.
    """
    sigmas = dict(modality_sigmas)
    sigmas[SHARED_GROUP] = max(modality_sigmas.values())
    return sigmas


def _group_grad_norm(params: list[nn.Parameter]) -> float:
    """L2 norm of the concatenated gradients of a parameter group."""
    sq = 0.0
    for param in params:
        if param.grad is not None:
            sq += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(sq)


def dp_train_epoch(
    model: nn.Module,
    dataset: torch.utils.data.Dataset[Sample],
    batches: list[list[int]],
    objective: DepressionObjective,
    *,
    groups: dict[str, list[nn.Parameter]],
    group_sigmas: dict[str, float],
    max_grad_norm: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    """Run one DP-SGD epoch over pre-formed logical batches.

    For each logical batch, per-sample gradients are computed by microbatching
    (one forward/backward per sample), each modality group is clipped to
    ``max_grad_norm`` per sample, summed, perturbed with group-specific Gaussian
    noise, averaged, and applied via ``optimizer.step()``.

    Args:
        model: The model to train.
        dataset: Dataset yielding individual :class:`Sample` items.
        batches: Logical batches as lists of dataset indices.
        objective: Loss object.
        groups: Parameter groups from :func:`map_parameter_groups`.
        group_sigmas: Noise multiplier per group (incl. ``shared``).
        max_grad_norm: Per-sample clipping bound ``C``.
        optimizer: Optimizer over all model parameters.
        device: Torch device.
        generator: RNG for reproducible Gaussian noise.

    Returns:
        Mean per-sample training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_indices in batches:
        accumulators: dict[nn.Parameter, torch.Tensor] = {
            param: torch.zeros_like(param) for params in groups.values() for param in params
        }

        for index in batch_indices:
            batch = move_batch_to_device(collate_fn([dataset[index]]), device)
            optimizer.zero_grad()
            loss = objective(model(batch), batch)
            loss.backward()
            total_loss += float(loss.item())
            total_samples += 1

            for params in groups.values():
                norm = _group_grad_norm(params)
                clip = min(1.0, max_grad_norm / (norm + 1e-6))
                for param in params:
                    if param.grad is not None:
                        accumulators[param].add_(param.grad.detach() * clip)

        batch_size = max(len(batch_indices), 1)
        optimizer.zero_grad()
        for group_name, params in groups.items():
            std = group_sigmas[group_name] * max_grad_norm
            for param in params:
                noisy = accumulators[param]
                if std > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=std,
                        size=param.shape,
                        generator=generator,
                        device=device,
                    )
                    noisy = noisy + noise
                param.grad = noisy / batch_size
        optimizer.step()

    return total_loss / max(total_samples, 1)
