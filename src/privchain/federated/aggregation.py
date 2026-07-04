"""Federated aggregation strategies (Phase 2 FedAvg + Phase 4 capability-aware).

* :func:`fedavg` — the deliberately-naive Phase 2 baseline: a single
  sample-count-weighted average over the whole parameter vector, ignoring which
  modalities a client actually has.
* :func:`capability_aware_aggregate` — the Phase 4 protocol (objective H2): each
  modality encoder is averaged only over its *subgraph* (clients that declare
  that modality), and every group is weighted by a per-client reputation score.
  Parameters a subgraph cannot inform are left at their previous global value.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from privchain.federated.capability import SHARED_GROUP, has_modality, param_group


@dataclass(frozen=True)
class ClientUpdate:
    """One client's contribution to a round of aggregation.

    Args:
        client_id: Identifier for logging / reputation tracking.
        capability: ``[audio, video, text]`` 0/1 availability flags.
        num_samples: Local training-sample count (data-volume signal).
        state: The client's updated model ``state_dict``.
    """

    client_id: int
    capability: tuple[int, int, int]
    num_samples: int
    state: OrderedDict[str, torch.Tensor]


def fedavg(
    client_states: list[OrderedDict[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict[str, torch.Tensor]:
    """Weighted average of client ``state_dict``s (FedAvg).

    Args:
        client_states: Per-client model ``state_dict``s (identical keys/shapes).
        weights: Per-client weights (e.g., number of local training samples).

    Returns:
        The aggregated ``state_dict``.

    Raises:
        ValueError: If inputs are empty, mismatched, or sum to zero weight.
    """
    if not client_states:
        raise ValueError("no client states to aggregate")
    if len(client_states) != len(weights):
        raise ValueError("client_states and weights must have the same length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("sum of weights must be positive")

    keys = client_states[0].keys()
    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        stacked = torch.stack([state[key].float() for state in client_states], dim=0)
        coeffs = torch.tensor(weights, dtype=torch.float32) / total
        # Broadcast coeffs over the parameter's trailing dims.
        shape = [len(weights)] + [1] * (stacked.dim() - 1)
        weighted = (stacked * coeffs.view(shape)).sum(dim=0)
        aggregated[key] = weighted.to(client_states[0][key].dtype)
    return aggregated


def _participates(update: ClientUpdate, group: str) -> bool:
    """Whether a client contributes to a parameter group's aggregation."""
    return group == SHARED_GROUP or has_modality(update.capability, group)


def capability_aware_aggregate(
    updates: list[ClientUpdate],
    global_state: OrderedDict[str, torch.Tensor],
    group_weights: list[dict[str, float]],
) -> OrderedDict[str, torch.Tensor]:
    """Aggregate per-modality encoders over their subgraphs (Phase 4, H2).

    Each parameter is routed to a group via
    :func:`~privchain.federated.capability.param_group`. A modality encoder's
    parameters are averaged **only** over clients that declare that modality;
    shared (fusion + head) parameters are averaged over all clients. Weighting
    uses the per-client, per-group weights (reputation × volume). If a group has
    no contributing client this round, its parameters keep the incoming
    ``global_state`` value (never overwritten with a zero-signal average).

    Args:
        updates: The round's client updates.
        global_state: The current global ``state_dict`` (fallback for empty
            subgroups; also defines the parameter keys/shapes).
        group_weights: Per-client mapping ``{group: weight}``; a client appears
            in a group's key set iff it contributes to that group. Aligned with
            ``updates`` by position.

    Returns:
        The aggregated global ``state_dict``.

    Raises:
        ValueError: If ``updates`` is empty or misaligned with ``group_weights``.
    """
    if not updates:
        raise ValueError("no client updates to aggregate")
    if len(updates) != len(group_weights):
        raise ValueError("updates and group_weights must have the same length")

    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, global_value in global_state.items():
        group = param_group(key)
        members = [
            i
            for i, update in enumerate(updates)
            if _participates(update, group) and group_weights[i].get(group, 0.0) > 0.0
        ]
        total = sum(group_weights[i][group] for i in members)
        if not members or total <= 0.0:
            # No client can inform this group — keep the previous global value.
            aggregated[key] = global_value.clone()
            continue

        acc = torch.zeros_like(global_value, dtype=torch.float32)
        for i in members:
            coeff = group_weights[i][group] / total
            acc += updates[i].state[key].float() * coeff
        aggregated[key] = acc.to(global_value.dtype)
    return aggregated
