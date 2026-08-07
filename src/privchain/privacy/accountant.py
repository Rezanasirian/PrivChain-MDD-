"""Rényi Differential Privacy accounting for the (subsampled) Gaussian mechanism.

Phase 3 (objective H1). A thin, typed wrapper over **Opacus**'s RDP analysis —
the accountant the thesis names — so every ``ε`` reported in Chapter 4 comes from
the reference implementation rather than a hand-rolled bound.

An earlier in-house implementation was replaced here (see ADR-0004): measured
against Opacus it over-reported ``ε`` by 15–22%, which is safe in direction but
calibrates ``σ`` larger than necessary and needlessly costs accuracy.

Two levels of accounting are exposed:

* :func:`get_epsilon` / :func:`get_noise_multiplier` — **one** mechanism, i.e. a
  single parameter group (one modality's encoder).
* :func:`compose_epsilon` — the **participant-level** budget: a subject whose
  data flows through several groups (audio, video, text and the shared fusion
  head) is exposed to all of those mechanisms, so their RDP curves must be summed
  before conversion to ``(ε, δ)``. See ADR-0009 for the privacy unit this
  implies.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from opacus.accountants.analysis import rdp as rdp_analysis
from opacus.accountants.utils import get_noise_multiplier as _opacus_noise_multiplier

# RDP orders searched over. Mirrors Opacus's own default grid (dense fractional
# orders below 11, which dominate at small ε, plus integers up to 64) and extends
# it with a few large orders so tight budgets do not hit the edge of the grid.
DEFAULT_ORDERS: tuple[float, ...] = (
    *(1.0 + x / 10.0 for x in range(1, 100)),
    *range(11, 64),
    64.0,
    96.0,
    128.0,
    256.0,
    512.0,
)


@dataclass(frozen=True)
class Mechanism:
    """One Sampled-Gaussian mechanism in a composition.

    Args:
        noise_multiplier: Gaussian noise multiplier ``σ`` (noise std / clip bound).
        sample_rate: Poisson sampling rate ``q`` in ``(0, 1]``.
        steps: Number of applications of this mechanism.
        name: Optional label (e.g. the modality) for reporting.
    """

    noise_multiplier: float
    sample_rate: float
    steps: int
    name: str = ""


def _rdp_curve(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    orders: Sequence[float],
) -> list[float]:
    """Return the total RDP at each order for ``steps`` applications.

    Args:
        noise_multiplier: Gaussian noise multiplier ``σ``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Number of mechanism applications.
        orders: Rényi orders.

    Returns:
        One RDP value per order (``0.0`` everywhere when ``steps == 0``).

    Raises:
        ValueError: If ``sample_rate`` is outside ``[0, 1]`` or ``steps < 0``.
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be in [0, 1]")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0 or sample_rate == 0.0:
        return [0.0] * len(orders)
    if noise_multiplier <= 0.0:
        return [float("inf")] * len(orders)
    curve = rdp_analysis.compute_rdp(
        q=sample_rate,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=list(orders),
    )
    return [float(value) for value in curve]


def get_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
) -> float:
    """Compute the ``ε`` spent by a single mechanism after ``steps`` steps.

    Args:
        noise_multiplier: Gaussian noise multiplier ``σ``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Number of mechanism applications (optimizer steps).
        delta: Target ``δ``.
        orders: Rényi orders to minimize over.

    Returns:
        The spent ``ε`` (``0.0`` when ``steps == 0``).

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if steps == 0:
        return 0.0
    curve = _rdp_curve(noise_multiplier, sample_rate, steps, orders)
    epsilon, _ = rdp_analysis.get_privacy_spent(orders=list(orders), rdp=curve, delta=delta)
    return float(epsilon)


def compose_epsilon(
    mechanisms: Iterable[Mechanism],
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
) -> float:
    """Compose several mechanisms into one participant-level ``(ε, δ)``.

    RDP composes additively at each order, so the per-group curves are summed
    *before* the conversion to ``(ε, δ)`` — summing the individual ``ε`` values
    instead would be needlessly loose. This is the budget a subject actually
    spends when their data touches every modality encoder plus the shared head.

    Args:
        mechanisms: The mechanisms to compose (e.g. one per parameter group).
        delta: Target ``δ``.
        orders: Rényi orders to minimize over.

    Returns:
        The composed ``ε`` (``0.0`` for an empty composition).

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    total = [0.0] * len(orders)
    any_mechanism = False
    for mechanism in mechanisms:
        any_mechanism = True
        curve = _rdp_curve(
            mechanism.noise_multiplier, mechanism.sample_rate, mechanism.steps, orders
        )
        total = [a + b for a, b in zip(total, curve, strict=True)]
    if not any_mechanism:
        return 0.0
    epsilon, _ = rdp_analysis.get_privacy_spent(orders=list(orders), rdp=total, delta=delta)
    return float(epsilon)


def get_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
    *,
    tolerance: float = 1e-3,
) -> float:
    """Find the smallest ``σ`` whose spent ``ε`` does not exceed ``target_epsilon``.

    Delegates to Opacus's calibration routine (binary search over the same RDP
    accountant used by :func:`get_epsilon`).

    Args:
        target_epsilon: Desired privacy budget ``ε``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Number of mechanism applications.
        delta: Target ``δ``.
        orders: Unused; kept so callers can pass a custom grid without breaking
            (Opacus's calibration uses its own default orders).
        tolerance: Accepted overshoot on ``ε`` during the search.

    Returns:
        A noise multiplier ``σ`` satisfying ``get_epsilon(σ, ...) <= target_epsilon``.

    Raises:
        ValueError: If ``target_epsilon <= 0`` or ``delta`` is out of range.
    """
    del orders  # Opacus calibrates against its own order grid.
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if steps <= 0:
        raise ValueError("steps must be positive to calibrate a noise multiplier")

    sigma = float(
        _opacus_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="rdp",
            epsilon_tolerance=tolerance,
        )
    )
    return sigma
