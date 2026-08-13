"""Per-modality differential-privacy budget allocation (Phase 3, objective H1).

This is the first core novelty of the thesis: instead of one uniform privacy
budget over the whole gradient vector, each modality ``m`` gets its own budget
``ε_m`` calibrated by its re-identification risk ``r_m`` (audio > video > text),
and a correspondingly calibrated noise multiplier ``σ_m``.

Formalization (the math destined for Chapter 3):

* Indices: client ``i``, modality ``m ∈ {audio, video, text}``, round/step ``t``.
* Parameters: per-modality risk ``r_m ∈ (0, 1]``, target ``δ``, sampling rate
  ``q``, planned steps ``T``; either explicit budgets ``ε_m`` or a total budget
  ``ε_total`` with sharpness ``γ``.
* Adaptive allocation (``inverse_risk`` mode):
  ``ε_m = ε_total · r_m^{-γ} / Σ_k r_k^{-γ}`` — higher risk ⇒ smaller budget.
* Decision variable: ``σ_{m} = min{ σ : ε_RDP(σ, q, T, δ) ≤ ε_m }`` via the RDP
  accountant (:func:`~privchain.privacy.accountant.get_noise_multiplier`).
* Auditable consumption after ``t`` steps: ``ε_m(t) = ε_RDP(σ_m, q, t, δ)`` — the
  per-modality budget each client reports (and later logs to the ledger).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from privchain.config import AllocationConfig, ModalityPrivacy
from privchain.privacy.accountant import (
    Mechanism,
    compose_epsilon,
    get_epsilon,
    get_noise_multiplier,
)


def allocate_target_epsilons(
    allocation: AllocationConfig, per_modality: dict[str, ModalityPrivacy]
) -> dict[str, float]:
    """Compute each modality's target ``ε_m`` under the configured mode.

    Args:
        allocation: Allocation mode + parameters.
        per_modality: Per-modality config (explicit ``epsilon`` and risk).

    Returns:
        Mapping ``{modality: target_epsilon}``.

    Raises:
        ValueError: If a risk is non-positive in ``inverse_risk`` mode.
    """
    if allocation.mode == "explicit":
        return {m: cfg.epsilon for m, cfg in per_modality.items()}

    # inverse_risk: distribute the total budget inversely to risk^gamma.
    gamma = allocation.risk_sharpness
    inv_weights: dict[str, float] = {}
    for modality, cfg in per_modality.items():
        if cfg.reidentification_risk <= 0.0:
            raise ValueError(f"risk for '{modality}' must be > 0 for inverse_risk allocation")
        inv_weights[modality] = cfg.reidentification_risk ** (-gamma)
    total_weight = sum(inv_weights.values())
    return {
        modality: allocation.total_epsilon * weight / total_weight
        for modality, weight in inv_weights.items()
    }


def scale_to_participant_epsilon(
    weights: Mapping[str, float],
    target_participant_epsilon: float,
    *,
    delta: float,
    sample_rate: float,
    steps: int,
    include_shared: bool = True,
    tolerance: float = 1.0e-3,
    max_iterations: int = 60,
) -> dict[str, float]:
    """Scale a budget *shape* until its composed participant ``ε`` hits a target.

    Comparing two allocations only means something if they cost the same
    privacy, and the obvious way to match them — giving both the same sum or
    mean of per-modality ``ε`` — does **not**. A participant contributing every
    modality is exposed to every mechanism, so what they actually spend is the
    RDP *composition* (:meth:`PerModalityBudgetAllocator.participant_epsilon`,
    ADR-0009), which is dominated by the loosest mechanism and is not linear in
    the individual budgets. Matching on the mean therefore hands more real
    privacy to whichever allocation is most uneven — in this project's case, the
    adaptive one, i.e. the hypothesis under test (ADR-0018).

    Composed ``ε`` is strictly increasing in the scale factor, so a bisection
    finds the multiplier that puts any shape on the same participant budget.

    Args:
        weights: Relative budget per modality; only the ratios matter.
        target_participant_epsilon: The composed budget every arm must spend.
        delta: Target ``δ``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Planned steps ``T``.
        include_shared: Whether the shared fusion/head group counts toward the
            composition, matching how the arms are actually trained.
        tolerance: Relative accuracy of the match.
        max_iterations: Bisection cap.

    Returns:
        Mapping ``{modality: epsilon}`` whose composed participant ``ε`` equals
        ``target_participant_epsilon`` to within ``tolerance``.

    Raises:
        ValueError: If ``weights`` is empty, any weight is non-positive, or the
            target is non-positive.
        RuntimeError: If no bracket containing the target can be found.
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    if any(weight <= 0.0 for weight in weights.values()):
        raise ValueError(f"every weight must be positive, got {dict(weights)}")
    if target_participant_epsilon <= 0.0:
        raise ValueError("target_participant_epsilon must be positive")

    # Normalize so the largest budget equals the scale factor; the composition is
    # then bounded below by the scale, which makes the initial bracket easy.
    largest = max(weights.values())
    shape = {modality: weight / largest for modality, weight in weights.items()}

    def composed(scale: float) -> float:
        allocator = PerModalityBudgetAllocator(
            {modality: scale * weight for modality, weight in shape.items()},
            dict.fromkeys(shape, 0.0),
            delta=delta,
            sample_rate=sample_rate,
            steps=steps,
        )
        return allocator.participant_epsilon(steps, include_shared=include_shared)

    # Every per-modality budget is at most `scale` and composing several
    # mechanisms costs more than any one of them, so `scale = target` is an upper
    # bracket up to the accountant's own slack. Expand either end until the
    # target is genuinely straddled rather than assuming it.
    high = target_participant_epsilon
    for _ in range(max_iterations):
        if composed(high) >= target_participant_epsilon:
            break
        high *= 2.0
    else:
        raise RuntimeError(
            f"could not bracket participant epsilon {target_participant_epsilon} "
            f"from above for shape {shape}"
        )

    low = high / 2.0
    for _ in range(max_iterations):
        if composed(low) < target_participant_epsilon:
            break
        low /= 2.0
    else:
        raise RuntimeError(
            f"could not bracket participant epsilon {target_participant_epsilon} "
            f"from below for shape {shape}"
        )

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        value = composed(middle)
        if abs(value - target_participant_epsilon) <= tolerance * target_participant_epsilon:
            break
        if value < target_participant_epsilon:
            low = middle
        else:
            high = middle
    else:
        middle = 0.5 * (low + high)

    return {modality: middle * weight for modality, weight in shape.items()}


@dataclass(frozen=True)
class ModalityAllocation:
    """Resolved per-modality budget and calibrated noise multiplier."""

    modality: str
    target_epsilon: float
    risk: float
    noise_multiplier: float


class PerModalityBudgetAllocator:
    """Calibrate and audit per-modality DP budgets.

    Args:
        target_epsilons: Per-modality target ``ε_m``.
        risks: Per-modality re-identification risk ``r_m`` (metadata for audit).
        delta: Target ``δ``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Planned number of steps ``T`` used to calibrate ``σ_m``.
    """

    def __init__(
        self,
        target_epsilons: dict[str, float],
        risks: dict[str, float],
        *,
        delta: float,
        sample_rate: float,
        steps: int,
    ) -> None:
        self.delta = delta
        self.sample_rate = sample_rate
        self.planned_steps = steps
        self.allocations: dict[str, ModalityAllocation] = {}
        for modality, target in target_epsilons.items():
            sigma = get_noise_multiplier(target, sample_rate, steps, delta)
            self.allocations[modality] = ModalityAllocation(
                modality=modality,
                target_epsilon=target,
                risk=risks.get(modality, float("nan")),
                noise_multiplier=sigma,
            )

    @classmethod
    def from_config(
        cls,
        allocation: AllocationConfig,
        per_modality: dict[str, ModalityPrivacy],
        *,
        delta: float,
        sample_rate: float,
        steps: int,
    ) -> PerModalityBudgetAllocator:
        """Build an allocator from validated config sections.

        Args:
            allocation: Allocation mode + parameters.
            per_modality: Per-modality config (epsilon + risk).
            delta: Target ``δ``.
            sample_rate: Poisson sampling rate ``q``.
            steps: Planned steps ``T``.

        Returns:
            A configured :class:`PerModalityBudgetAllocator`.
        """
        targets = allocate_target_epsilons(allocation, per_modality)
        risks = {m: cfg.reidentification_risk for m, cfg in per_modality.items()}
        return cls(targets, risks, delta=delta, sample_rate=sample_rate, steps=steps)

    def noise_multipliers(self) -> dict[str, float]:
        """Return the calibrated ``σ_m`` per modality."""
        return {m: a.noise_multiplier for m, a in self.allocations.items()}

    def consumed_epsilon(self, steps_done: int) -> dict[str, float]:
        """Per-modality ``ε`` actually consumed after ``steps_done`` steps.

        This is the auditable quantity each client reports (CLAUDE.md §7) — it
        must never be silently overwritten.

        Args:
            steps_done: Number of steps actually executed.

        Returns:
            Mapping ``{modality: epsilon_spent}``.
        """
        return {
            modality: get_epsilon(alloc.noise_multiplier, self.sample_rate, steps_done, self.delta)
            for modality, alloc in self.allocations.items()
        }

    def participant_epsilon(self, steps_done: int, *, include_shared: bool = True) -> float:
        """Composed ``ε`` for a subject whose data touches every group (ADR-0009).

        The per-modality budgets of :meth:`consumed_epsilon` are *per mechanism*.
        A participant contributing all three modalities is exposed to all three
        mechanisms plus the shared fusion/head group, so their true budget is the
        RDP composition of those mechanisms — always larger than any single
        ``ε_m``, and reported alongside them so the audit trail is honest.

        Args:
            steps_done: Number of steps actually executed.
            include_shared: Whether to include the shared (fusion + head) group,
                which conservatively runs at ``max_m σ_m``.

        Returns:
            The composed ``ε`` across all mechanisms the participant is exposed to.
        """
        mechanisms = [
            Mechanism(
                noise_multiplier=alloc.noise_multiplier,
                sample_rate=self.sample_rate,
                steps=steps_done,
                name=modality,
            )
            for modality, alloc in self.allocations.items()
        ]
        if include_shared and self.allocations:
            mechanisms.append(
                Mechanism(
                    noise_multiplier=max(a.noise_multiplier for a in self.allocations.values()),
                    sample_rate=self.sample_rate,
                    steps=steps_done,
                    name="shared",
                )
            )
        return compose_epsilon(mechanisms, self.delta)
