"""Byzantine-robust outlier filtering (Phase 5, objective H3).

A light, Sho-et-al.-2024-inspired defence layered on top of the reputation
weighting: before aggregation, flag clients whose shared-group update is a gross
statistical outlier (a corrupted or adversarial contribution) and drop them.

The score is the L2 norm of each client's update to the **shared** parameter
group (fusion + heads — the group every client touches). Outliers are detected
with a robust median / MAD rule so a single extreme client cannot mask others.
The filter is conservative: it never removes so many clients that aggregation
would be left empty, and it does nothing for very small cohorts.
"""

from __future__ import annotations

import math
import statistics
from collections import OrderedDict

import torch

from privchain.federated.aggregation import ClientUpdate
from privchain.federated.capability import SHARED_GROUP, param_group

# Minimum cohort size for outlier detection to be meaningful.
_MIN_FOR_FILTER = 3
# Consistency factor making MAD a standard-deviation-scale estimator on normals.
_MAD_TO_STD = 1.4826


def _shared_delta_norm(
    state: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    shared_keys: list[str],
) -> float:
    """L2 norm of a client's update to the shared parameter group."""
    squared = 0.0
    for key in shared_keys:
        delta = state[key].float() - global_state[key].float()
        squared += float(delta.pow(2).sum().item())
    return math.sqrt(squared)


def flag_byzantine_updates(
    updates: list[ClientUpdate],
    global_state: OrderedDict[str, torch.Tensor],
    *,
    z_thresh: float,
) -> set[int]:
    """Return the indices of updates whose shared-group step is an outlier.

    Args:
        updates: The round's client updates.
        global_state: The global ``state_dict`` this round started from.
        z_thresh: How many robust standard deviations from the median a norm may
            be before the client is flagged (larger ⇒ more permissive).

    Returns:
        Indices into ``updates`` to exclude. Empty for cohorts smaller than
        three, or when at least half the cohort would be flagged (treated as no
        reliable consensus rather than mass exclusion).
    """
    if len(updates) < _MIN_FOR_FILTER:
        return set()

    shared_keys = [key for key in global_state if param_group(key) == SHARED_GROUP]
    norms = [_shared_delta_norm(u.state, global_state, shared_keys) for u in updates]

    median = statistics.median(norms)
    mad = statistics.median([abs(value - median) for value in norms])
    if mad > 0.0:
        scale = _MAD_TO_STD * mad
    else:
        # Degenerate MAD (≥ half the norms identical): fall back to std.
        scale = statistics.pstdev(norms)
    if scale <= 0.0:
        return set()

    flagged = {i for i, value in enumerate(norms) if abs(value - median) > z_thresh * scale}
    # Never drop half or more of the cohort — that is not a robust consensus.
    if len(flagged) * 2 >= len(updates):
        return set()
    return flagged
