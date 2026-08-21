"""Per-modality DP-SGD with Poisson subsampling (Phase 3, objective H1).

Implements differentially-private SGD where **each modality is an independent DP
mechanism**: the per-sample gradient of each modality's parameter group is
clipped to ``C`` and perturbed with Gaussian noise scaled by that modality's own
multiplier ``σ_m`` (from
:class:`~privchain.privacy.budget_allocator.PerModalityBudgetAllocator`).

Two properties make the reported ``ε`` actually valid (ADR-0004, ADR-0009):

* **Poisson subsampling.** The RDP accountant of
  :mod:`privchain.privacy.accountant` assumes each sample enters a step
  independently with probability ``q``. Shuffled fixed-size batches do not
  satisfy that, so :func:`poisson_batches` draws real Bernoulli(``q``) batches
  and the noisy sum is normalised by the **expected** batch size ``q·N``
  — dividing by the realised size would leak it.
* **Verifiable per-sample gradients.** The fast path uses
  :class:`opacus.GradSampleModule` (one backward pass for the whole batch); the
  ``microbatch`` path recomputes them one sample at a time. A unit test asserts
  the two agree, so the fast path is never trusted blindly.

Parameter→modality grouping (by name prefix on ``MultimodalDepressionModel``):
``encoders.audio`` → audio, ``encoders.video`` → video, ``encoders.text`` → text,
everything else (fusion + heads) → ``shared``. The shared group sees all
modalities, so it conservatively takes the **largest** ``σ`` among modalities.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from opacus import GradSampleModule
from torch import nn

from privchain.config import CAPABILITY_MODALITIES
from privchain.data.mock_daic_woz import Sample, collate_fn
from privchain.training.objective import DepressionObjective, move_batch_to_device

SHARED_GROUP = "shared"
PerSampleBackend = Literal["grad_sample", "microbatch"]

# Prefix `GradSampleModule` prepends to every wrapped parameter name.
_WRAPPER_PREFIX = "_module."


def _strip_wrapper_prefix(name: str) -> str:
    """Remove any ``GradSampleModule`` wrapper prefixes from a parameter name."""
    while name.startswith(_WRAPPER_PREFIX):
        name = name[len(_WRAPPER_PREFIX) :]
    return name


def map_parameter_groups(
    model: nn.Module, capability: tuple[int, int, int] | None = None
) -> dict[str, list[nn.Parameter]]:
    """Group a model's parameters into per-modality groups + a shared group.

    Works on both a bare model and one wrapped in
    :class:`opacus.GradSampleModule` (the wrapper's ``_module.`` prefix is
    stripped before matching).

    Args:
        model: A :class:`MultimodalDepressionModel` (or compatible) instance.
        capability: Optional ``[audio, video, text]`` availability flags. When
            supplied, absent encoder groups are omitted entirely.

    Returns:
        Mapping ``{group_name: [parameters]}`` with keys ``audio``, ``video``,
        ``text``, and ``shared``.
    """
    enabled = set(CAPABILITY_MODALITIES)
    if capability is not None:
        enabled = {
            modality
            for modality, flag in zip(CAPABILITY_MODALITIES, capability, strict=True)
            if flag == 1
        }
    groups: dict[str, list[nn.Parameter]] = {name: [] for name in enabled}
    groups[SHARED_GROUP] = []
    for raw_name, param in model.named_parameters():
        name = _strip_wrapper_prefix(raw_name)
        if name.startswith("encoders.audio") and "audio" in groups:
            groups["audio"].append(param)
        elif name.startswith("encoders.video") and "video" in groups:
            groups["video"].append(param)
        elif name.startswith("encoders.text") and "text" in groups:
            groups["text"].append(param)
        elif not name.startswith("encoders."):
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


def poisson_batches(
    num_items: int,
    sample_rate: float,
    steps: int,
    generator: torch.Generator,
) -> list[list[int]]:
    """Draw ``steps`` Poisson-subsampled batches of indices.

    Each of the ``num_items`` samples is included in each batch independently
    with probability ``sample_rate`` — the sampling scheme the RDP accountant
    assumes. Batches therefore have varying (occasionally zero) size.

    Args:
        num_items: Dataset size ``N``.
        sample_rate: Inclusion probability ``q`` in ``(0, 1]``.
        steps: Number of batches (mechanism applications) to draw.
        generator: RNG for reproducibility.

    Returns:
        ``steps`` lists of dataset indices.

    Raises:
        ValueError: If ``num_items``/``steps`` are negative or ``q`` is invalid.
    """
    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must be in (0, 1]")

    batches: list[list[int]] = []
    for _ in range(steps):
        mask = torch.rand(num_items, generator=generator, device=generator.device) < sample_rate
        batches.append([int(i) for i in torch.nonzero(mask, as_tuple=False).flatten().tolist()])
    return batches


def steps_for_epochs(num_items: int, batch_size: int, epochs: int) -> int:
    """Number of Poisson steps matching ``epochs`` passes at ``batch_size``.

    With Poisson sampling an "epoch" is a budgeting convention, not a traversal:
    it is the number of steps whose expected sample count equals ``epochs``
    passes over the data.

    Args:
        num_items: Dataset size ``N``.
        batch_size: Nominal batch size (``q = batch_size / N``).
        epochs: Number of nominal passes.

    Returns:
        The equivalent number of mechanism applications.
    """
    return int(math.ceil(num_items / max(batch_size, 1)) * epochs)


def _accumulate_microbatch(
    model: nn.Module,
    dataset: torch.utils.data.Dataset[Sample],
    batch_indices: list[int],
    objective: DepressionObjective,
    groups: dict[str, list[nn.Parameter]],
    accumulators: dict[nn.Parameter, torch.Tensor],
    max_grad_norm: float,
    device: torch.device,
) -> float:
    """Clip and accumulate per-sample gradients one sample at a time.

    Args:
        model: The (unwrapped) model.
        dataset: Dataset yielding individual samples.
        batch_indices: Indices in this batch.
        objective: Loss object.
        groups: Parameter groups.
        accumulators: Per-parameter running sums to add the clipped gradients to.
        max_grad_norm: Per-sample, per-group clipping bound ``C``.
        device: Torch device.

    Returns:
        Summed (unreduced) loss over the batch.
    """
    total_loss = 0.0
    for index in batch_indices:
        batch = move_batch_to_device(collate_fn([dataset[index]]), device)
        model.zero_grad(set_to_none=True)
        loss = objective(model(batch), batch)
        loss.backward()  # type: ignore[no-untyped-call]
        total_loss += float(loss.item())

        for params in groups.values():
            squared = 0.0
            for param in params:
                if param.grad is not None:
                    squared += float(param.grad.detach().pow(2).sum().item())
            coefficient = min(1.0, max_grad_norm / (math.sqrt(squared) + 1e-6))
            for param in params:
                if param.grad is not None:
                    accumulators[param].add_(param.grad.detach() * coefficient)
    model.zero_grad(set_to_none=True)
    return total_loss


def _accumulate_grad_sample(
    model: GradSampleModule,
    dataset: torch.utils.data.Dataset[Sample],
    batch_indices: list[int],
    objective: DepressionObjective,
    groups: dict[str, list[nn.Parameter]],
    accumulators: dict[nn.Parameter, torch.Tensor],
    max_grad_norm: float,
    device: torch.device,
) -> float:
    """Clip and accumulate per-sample gradients from a single backward pass.

    ``GradSampleModule`` defaults to ``loss_reduction="mean"``, so it rescales
    the backprops by the batch size; the loss below is therefore the *mean* loss
    and ``p.grad_sample[i]`` is exactly sample ``i``'s own gradient.

    Args:
        model: The model wrapped in :class:`opacus.GradSampleModule`.
        dataset: Dataset yielding individual samples.
        batch_indices: Indices in this batch.
        objective: Loss object.
        groups: Parameter groups.
        accumulators: Per-parameter running sums to add the clipped gradients to.
        max_grad_norm: Per-sample, per-group clipping bound ``C``.
        device: Torch device.

    Returns:
        Summed (unreduced) loss over the batch.
    """
    batch = move_batch_to_device(collate_fn([dataset[i] for i in batch_indices]), device)
    model.zero_grad(set_to_none=True)
    mean_loss = objective(model(batch), batch)
    mean_loss.backward()  # type: ignore[no-untyped-call]
    batch_size = len(batch_indices)

    # Per-sample, per-group L2 norms -> clipping coefficients.
    coefficients: dict[str, torch.Tensor] = {}
    for group_name, params in groups.items():
        squared = torch.zeros(batch_size, device=device)
        for param in params:
            grad_sample = getattr(param, "grad_sample", None)
            if grad_sample is None:
                continue
            squared = squared + grad_sample.detach().reshape(batch_size, -1).pow(2).sum(dim=1)
        norms = squared.sqrt()
        coefficients[group_name] = (max_grad_norm / (norms + 1e-6)).clamp(max=1.0)

    for group_name, params in groups.items():
        coefficient = coefficients[group_name]
        for param in params:
            grad_sample = getattr(param, "grad_sample", None)
            if grad_sample is None:
                continue
            scaled = grad_sample.detach() * coefficient.view(-1, *([1] * (grad_sample.dim() - 1)))
            accumulators[param].add_(scaled.sum(dim=0))

    model.zero_grad(set_to_none=True)
    return float(mean_loss.item()) * batch_size


def dp_train_steps(
    model: nn.Module,
    dataset: torch.utils.data.Dataset[Sample],
    batches: list[list[int]],
    objective: DepressionObjective,
    *,
    groups: dict[str, list[nn.Parameter]],
    group_sigmas: dict[str, float],
    max_grad_norm: float,
    expected_batch_size: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    generator: torch.Generator,
    backend: PerSampleBackend = "grad_sample",
) -> float:
    """Run per-modality DP-SGD over pre-drawn Poisson batches.

    For each batch, per-sample gradients are clipped to ``max_grad_norm``
    *within each modality group*, summed, perturbed with that group's Gaussian
    noise, normalised by ``expected_batch_size``, and applied via
    ``optimizer.step()``. Empty batches (possible under Poisson sampling) still
    consume a mechanism application: noise is added and a step is taken, which is
    what the accountant charges for.

    Args:
        model: The model to train — a :class:`opacus.GradSampleModule` for the
            ``grad_sample`` backend, a plain module for ``microbatch``.
        dataset: Dataset yielding individual :class:`Sample` items.
        batches: Poisson batches from :func:`poisson_batches`.
        objective: Loss object.
        groups: Parameter groups from :func:`map_parameter_groups`.
        group_sigmas: Noise multiplier per group (incl. ``shared``).
        max_grad_norm: Per-sample clipping bound ``C``.
        expected_batch_size: ``q·N`` — the normaliser. Must not depend on the
            realised batch size.
        optimizer: Optimizer over all model parameters.
        device: Torch device.
        generator: RNG for reproducible Gaussian noise (must match ``device``).
        backend: Per-sample gradient strategy.

    Returns:
        Mean per-sample training loss over the processed samples.

    Raises:
        ValueError: On an invalid ``expected_batch_size``, a backend/model
            mismatch, or a generator on the wrong device.
    """
    if expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be positive")
    if generator.device.type != device.type:
        raise ValueError(
            f"generator is on '{generator.device}' but training runs on '{device}'; "
            "create it with torch.Generator(device=device)"
        )
    if backend == "grad_sample" and not isinstance(model, GradSampleModule):
        raise ValueError(
            "backend='grad_sample' needs the model wrapped in GradSampleModule "
            "(use wrap_for_per_sample_grads)"
        )
    if backend == "microbatch" and isinstance(model, GradSampleModule):
        raise ValueError("backend='microbatch' expects an unwrapped model")

    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_indices in batches:
        accumulators: dict[nn.Parameter, torch.Tensor] = {
            param: torch.zeros_like(param) for params in groups.values() for param in params
        }

        if batch_indices:
            if backend == "grad_sample":
                assert isinstance(model, GradSampleModule)  # narrowed above
                total_loss += _accumulate_grad_sample(
                    model,
                    dataset,
                    batch_indices,
                    objective,
                    groups,
                    accumulators,
                    max_grad_norm,
                    device,
                )
            else:
                total_loss += _accumulate_microbatch(
                    model,
                    dataset,
                    batch_indices,
                    objective,
                    groups,
                    accumulators,
                    max_grad_norm,
                    device,
                )
            total_samples += len(batch_indices)

        optimizer.zero_grad(set_to_none=True)
        for group_name, params in groups.items():
            std = group_sigmas[group_name] * max_grad_norm
            for param in params:
                noisy = accumulators[param]
                if std > 0:
                    noisy = noisy + torch.normal(
                        mean=0.0,
                        std=std,
                        size=param.shape,
                        generator=generator,
                        device=device,
                    )
                param.grad = noisy / expected_batch_size
        optimizer.step()

    return total_loss / max(total_samples, 1)


def wrap_for_per_sample_grads(model: nn.Module) -> GradSampleModule:
    """Wrap a model so a single backward pass yields per-sample gradients.

    The wrapper shares parameter objects with ``model``, so an optimizer built
    over either sees the same tensors.

    Args:
        model: The model to wrap.

    Returns:
        The model wrapped in :class:`opacus.GradSampleModule` (returned as-is if
        it already is one).
    """
    if isinstance(model, GradSampleModule):
        return model
    return GradSampleModule(model)
