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

from dataclasses import dataclass

from privchain.config import AllocationConfig, ModalityPrivacy
from privchain.privacy.accountant import get_epsilon, get_noise_multiplier


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
            modality: get_epsilon(
                alloc.noise_multiplier, self.sample_rate, steps_done, self.delta
            )
            for modality, alloc in self.allocations.items()
        }
