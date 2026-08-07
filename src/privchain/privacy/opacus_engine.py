"""Opacus integration notes for the per-modality DP mechanism (Phase 3, H1/H4).

The thesis names **Opacus** for per-vertex/per-modality noise injection, and the
project now uses it in two places:

1. :mod:`privchain.privacy.accountant` wraps Opacus's RDP analysis — every ``ε``
   reported in Chapter 4 comes from Opacus.
2. :mod:`privchain.privacy.dp_sgd` uses :class:`opacus.GradSampleModule` to
   obtain per-sample gradients in a single backward pass, then applies the
   **per-modality** clipping and noise that is the actual novelty of H1.

Why not ``opacus.PrivacyEngine`` directly? ``PrivacyEngine`` attaches one
mechanism to one optimizer, clipping the *whole* gradient vector to a single
bound with a single ``σ``. H1 requires a different clip/noise pair per modality
parameter group with a shared optimizer step, so the engine's outer loop is
replaced while its per-sample gradient machinery and accountant are reused.
The equivalent production wiring with stock Opacus would be one optimizer +
``PrivacyEngine`` per modality encoder, stepped together each iteration; that
composes identically (see :func:`privchain.privacy.accountant.compose_epsilon`).

Note that :class:`opacus.GradSampleModule` does not support ``nn.GRU``; the
sequence encoders therefore use :class:`opacus.layers.DPGRU` (see ADR-0004).
"""

from __future__ import annotations


def opacus_available() -> bool:
    """Return whether the ``opacus`` package is importable.

    ``opacus`` is a required dependency, so this is a diagnostic helper for
    environment checks rather than a feature gate.

    Returns:
        ``True`` when ``opacus`` can be imported.
    """
    import importlib.util

    return importlib.util.find_spec("opacus") is not None
