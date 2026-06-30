"""Federated aggregation strategies (Phase 2: FedAvg).

Plain FedAvg — a sample-count-weighted average of client model parameters. This
is the deliberately-naive Phase 2 baseline (no missing-modality handling, no
reputation, no Byzantine robustness); those arrive in Phase 4.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


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
