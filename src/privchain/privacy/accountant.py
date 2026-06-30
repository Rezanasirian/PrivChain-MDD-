"""Rényi Differential Privacy accountant for the (subsampled) Gaussian mechanism.

Phase 3 (objective H1). Pure-NumPy/SciPy implementation of the RDP of the
Sampled Gaussian Mechanism (Mironov et al., 2019) so per-modality privacy
budgets can be computed offline — the same accounting Opacus performs under the
hood. ``opacus_engine.cross_check_epsilon`` validates these numbers against
Opacus when it is installed.

Given a noise multiplier ``σ`` (noise std / clipping bound), sampling rate ``q``,
and ``T`` steps, we compute the RDP at a set of integer orders ``α`` and convert
to ``(ε, δ)``-DP via the standard bound
``ε = min_α  RDP_total(α) + log(1/δ)/(α-1)``.
"""

from __future__ import annotations

import math

from scipy.special import logsumexp

# Integer RDP orders to search over (the SGM integer-order bound).
DEFAULT_ORDERS: tuple[int, ...] = tuple(range(2, 65))


def _log_binom(n: int, k: int) -> float:
    """Natural log of the binomial coefficient ``C(n, k)``."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_sampled_gaussian(alpha: int, sample_rate: float, noise_multiplier: float) -> float:
    """RDP at integer order ``alpha`` of the Sampled Gaussian Mechanism.

    Args:
        alpha: Integer Rényi order (``>= 2``).
        sample_rate: Poisson sampling rate ``q`` in ``[0, 1]``.
        noise_multiplier: Gaussian noise multiplier ``σ`` (``> 0``).

    Returns:
        The RDP value ``ε_RDP(α)`` for a single step.

    Raises:
        ValueError: If ``alpha < 2`` or inputs are out of range.
    """
    if alpha < 2:
        raise ValueError("alpha must be >= 2")
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError("sample_rate must be in [0, 1]")
    if noise_multiplier <= 0.0:
        return math.inf
    if sample_rate == 0.0:
        return 0.0
    if sample_rate == 1.0:
        # Non-subsampled Gaussian: RDP(α) = α / (2 σ²).
        return alpha / (2.0 * noise_multiplier**2)

    log_terms = []
    for k in range(alpha + 1):
        log_coef = (
            _log_binom(alpha, k)
            + k * math.log(sample_rate)
            + (alpha - k) * math.log1p(-sample_rate)
        )
        log_terms.append(log_coef + (k * k - k) / (2.0 * noise_multiplier**2))
    return float(logsumexp(log_terms) / (alpha - 1))


def get_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: tuple[int, ...] = DEFAULT_ORDERS,
) -> float:
    """Compute the ``(ε, δ)`` spent after ``steps`` SGM steps.

    Args:
        noise_multiplier: Gaussian noise multiplier ``σ``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Number of mechanism applications (optimizer steps).
        delta: Target ``δ``.
        orders: Integer RDP orders to minimize over.

    Returns:
        The spent ``ε`` (``0.0`` when ``steps == 0``).

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if steps == 0:
        return 0.0

    log_inv_delta = math.log(1.0 / delta)
    best = math.inf
    for alpha in orders:
        rdp_total = steps * rdp_sampled_gaussian(alpha, sample_rate, noise_multiplier)
        epsilon = rdp_total + log_inv_delta / (alpha - 1)
        best = min(best, epsilon)
    return best


def get_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float,
    orders: tuple[int, ...] = DEFAULT_ORDERS,
    *,
    tolerance: float = 1e-3,
) -> float:
    """Find the smallest ``σ`` whose spent ``ε`` does not exceed ``target_epsilon``.

    ``ε`` is monotonically decreasing in ``σ``, so a binary search applies.

    Args:
        target_epsilon: Desired privacy budget ``ε``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Number of mechanism applications.
        delta: Target ``δ``.
        orders: Integer RDP orders.
        tolerance: Stop when the search bracket is narrower than this.

    Returns:
        A noise multiplier ``σ`` satisfying ``get_epsilon(σ, ...) <= target_epsilon``.

    Raises:
        ValueError: If ``target_epsilon <= 0``.
    """
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive")

    low, high = 1e-3, 1.0
    # Grow `high` until the budget is satisfied (ε decreases as σ grows).
    while get_epsilon(high, sample_rate, steps, delta, orders) > target_epsilon:
        high *= 2.0
        if high > 1e7:
            return high
    # If even a tiny σ already satisfies the budget, return it.
    if get_epsilon(low, sample_rate, steps, delta, orders) <= target_epsilon:
        return low

    while high - low > tolerance:
        mid = (low + high) / 2.0
        if get_epsilon(mid, sample_rate, steps, delta, orders) > target_epsilon:
            low = mid  # too little noise -> need more
        else:
            high = mid  # enough noise -> can try less
    return high
