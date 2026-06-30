"""Opacus bridge and accountant cross-check (Phase 3, objective H1/H4).

The thesis names **Opacus** for per-modality noise injection. ``opacus`` is an
optional dependency and is not installed in this offline environment, so Phase 3
implements the DP mechanism directly (:mod:`privchain.privacy.dp_sgd`) — which is
mathematically what Opacus does (per-sample clipping + Gaussian noise + RDP
accounting).

This module provides:

1. :func:`opacus_available` / :func:`cross_check_epsilon` — when ``opacus`` *is*
   installed, recompute ε with Opacus's ``RDPAccountant`` to validate our
   :mod:`privchain.privacy.accountant` numbers.
2. Documentation of how to attach a real ``opacus.PrivacyEngine`` per modality
   (see module notes) for the production path.

Production wiring (when ``opacus`` is installed): build one optimizer per
modality encoder, wrap each with its own ``PrivacyEngine`` configured with that
modality's ``noise_multiplier`` and ``max_grad_norm``, and step them together.
Because each modality is an independent DP mechanism, per-modality ε accounting
is exactly the per-group accounting used here.
"""

from __future__ import annotations


def opacus_available() -> bool:
    """Return whether the ``opacus`` package is importable."""
    import importlib.util

    return importlib.util.find_spec("opacus") is not None


def cross_check_epsilon(
    noise_multiplier: float, sample_rate: float, steps: int, delta: float
) -> float:
    """Recompute ε via Opacus's RDP accountant (for validation/tests).

    Args:
        noise_multiplier: Gaussian noise multiplier ``σ``.
        sample_rate: Poisson sampling rate ``q``.
        steps: Number of steps.
        delta: Target ``δ``.

    Returns:
        Opacus's spent ``ε`` for the same parameters.

    Raises:
        ImportError: If ``opacus`` is not installed.
    """
    try:
        from opacus.accountants import RDPAccountant
    except ImportError as exc:  # pragma: no cover - exercised only without opacus
        raise ImportError(
            "cross_check_epsilon requires 'opacus'. Install with `pip install opacus`."
        ) from exc

    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return float(accountant.get_epsilon(delta=delta))
