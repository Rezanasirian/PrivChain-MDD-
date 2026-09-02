"""The one place capability patterns are read from (Phase 4/1, ADR-0028).

Three separate things need the same list of modality-access patterns:

* the **centralized training schedule**, which shows each participant a different
  capability on different epochs;
* the **counterfactual evaluation**, which scores every held-out participant once
  per capability;
* the **federated deployment**, where each client's fixed capability decides what
  it can train on.

If any two of those drift apart, the centralized number stops predicting the
federated one and nobody finds out — the run still completes. So all three read
``federation.modality_patterns`` from ``configs/federated.yaml`` through this
module, and a contract test asserts they agree.

The patterns are deliberately *not* re-declared here. Adding ``video_only``
would widen the thesis's claim and add a test the deployment scenario never
runs; that belongs in the config, as a decision, not in a script's default.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from pathlib import Path

from privchain.config import CAPABILITY_MODALITIES, ModalityPattern, load_federated_config

#: Default location of the config that owns the patterns.
DEFAULT_FEDERATED_CONFIG = Path("configs/federated.yaml")


def load_modality_patterns(path: str | Path | None = None) -> list[ModalityPattern]:
    """Read the capability patterns from the federated config.

    Args:
        path: Config path; defaults to :data:`DEFAULT_FEDERATED_CONFIG`.

    Returns:
        The declared patterns, in config order.

    Raises:
        ValueError: If the config declares no patterns.
    """
    config = load_federated_config(DEFAULT_FEDERATED_CONFIG if path is None else path)
    patterns = list(config.federation.modality_patterns)
    if not patterns:
        raise ValueError(f"{path}: federation.modality_patterns is empty")
    return patterns


def capability_of(pattern: ModalityPattern) -> dict[str, int]:
    """Return one pattern's capability as a ``{modality: 0/1}`` mapping.

    Args:
        pattern: A declared modality pattern.

    Returns:
        Mapping keyed by :data:`~privchain.config.CAPABILITY_MODALITIES`.
    """
    return dict(zip(CAPABILITY_MODALITIES, pattern.capability, strict=True))


def cycle_counts(patterns: list[ModalityPattern]) -> list[int]:
    """Turn population fractions into whole visits per training cycle.

    The schedule must reproduce the deployment mix exactly, so the fractions are
    converted with exact rational arithmetic rather than by rounding: 0.4/0.3/
    0.2/0.1 becomes 4/3/2/1 visits in a 10-epoch cycle, not four floats that
    almost add up. Rounding would quietly under-represent the rarest pattern —
    which is the one the worst-capability metric is about.

    Args:
        patterns: The declared patterns.

    Returns:
        One positive visit count per pattern, in the same order, reduced by
        their greatest common divisor so the cycle is as short as it can be.

    Raises:
        ValueError: If the fractions do not sum to 1, or any pattern would get
            zero visits.
    """
    fractions = [Fraction(str(pattern.fraction)) for pattern in patterns]
    total = sum(fractions, Fraction(0))
    if total != 1:
        raise ValueError(
            f"modality_pattern fractions must sum to 1 for a training schedule, got {float(total)}"
        )

    denominator = 1
    for fraction in fractions:
        denominator = denominator * fraction.denominator // gcd(denominator, fraction.denominator)
    counts = [int(fraction * denominator) for fraction in fractions]

    divisor = 0
    for count in counts:
        divisor = gcd(divisor, count)
    counts = [count // divisor for count in counts]

    if any(count < 1 for count in counts):
        raise ValueError(f"every pattern needs at least one visit per cycle, got {counts}")
    return counts
