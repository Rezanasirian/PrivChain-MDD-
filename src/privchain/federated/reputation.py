"""Reputation-based aggregation weighting (Phase 4, objective H2).

Each round, every participating client earns a per-group reputation in ``[0, 1]``
that blends two auditable signals (see ADR-0005):

* **data volume** — the client's local sample count, relative to the largest in
  the same subgraph; and
* **update consistency** — the cosine agreement between the client's update for
  that group and the subgraph's volume-weighted consensus update. A client whose
  update points against the consensus (e.g. a corrupted / Byzantine client) earns
  low consistency and is down-weighted.

Reputation is smoothed across rounds with an EMA so a single noisy round does not
dominate. The resulting weight fed to
:func:`~privchain.federated.aggregation.capability_aware_aggregate` is
``max(reputation, min_reputation) × volume``. When ``use_reputation`` is False the
tracker degrades to pure volume weighting (FedAvg-style) while still respecting
subgraph membership.

The per-client, per-modality reputation snapshot is exactly what the Phase 5
``UpdateReputation`` chaincode function will persist to the ledger.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from privchain.config import CAPABILITY_MODALITIES, ReputationConfig
from privchain.federated.aggregation import ClientUpdate, _participates
from privchain.federated.capability import SHARED_GROUP, param_group

# All aggregation groups, in a fixed order for reproducible iteration.
GROUPS: tuple[str, ...] = (*CAPABILITY_MODALITIES, SHARED_GROUP)


def _flatten_delta(
    state: OrderedDict[str, torch.Tensor],
    global_state: OrderedDict[str, torch.Tensor],
    group_keys: list[str],
) -> torch.Tensor:
    """Flatten a client's update delta for one group into a 1-D vector."""
    if not group_keys:
        return torch.zeros(1)
    parts = [(state[key].float() - global_state[key].float()).reshape(-1) for key in group_keys]
    return torch.cat(parts)


class ReputationTracker:
    """Tracks and updates per-client, per-group reputation across rounds.

    Args:
        config: Reputation hyperparameters (volume weight, EMA decay, floor).
    """

    def __init__(self, config: ReputationConfig) -> None:
        """Initialize the tracker with its reputation hyperparameters."""
        self._config = config
        # client_id -> {group -> reputation in [0, 1]}
        self._reputation: dict[int, dict[str, float]] = {}

    def snapshot(self) -> dict[int, dict[str, float]]:
        """Return a copy of the current per-client, per-group reputation.

        Returns:
            Mapping ``{client_id: {group: reputation}}`` (ledger-ready).
        """
        return {cid: dict(groups) for cid, groups in self._reputation.items()}

    def compute_weights(
        self,
        updates: list[ClientUpdate],
        global_state: OrderedDict[str, torch.Tensor],
        *,
        use_reputation: bool = True,
    ) -> list[dict[str, float]]:
        """Update reputation from this round and return aggregation weights.

        Args:
            updates: The round's client updates.
            global_state: The global ``state_dict`` this round started from
                (baseline for the update deltas and group key layout).
            use_reputation: When False, weights reduce to raw sample counts
                (volume only), bypassing the reputation multiplier.

        Returns:
            Per-client mapping ``{group: weight}`` aligned with ``updates``; a
            client only has a key for a group it contributes to.
        """
        group_keys = {
            group: [k for k in global_state if param_group(k) == group] for group in GROUPS
        }
        weights: list[dict[str, float]] = [{} for _ in updates]

        for group in GROUPS:
            members = [i for i, u in enumerate(updates) if _participates(u, group)]
            if not members:
                continue

            keys = group_keys[group]
            volumes = {i: float(updates[i].num_samples) for i in members}
            max_volume = max(volumes.values()) or 1.0
            deltas = {
                i: _flatten_delta(updates[i].state, global_state, keys) for i in members
            }
            consensus = self._consensus(deltas, volumes)

            for i in members:
                norm_volume = volumes[i] / max_volume
                consistency = self._consistency(deltas[i], consensus, len(members))
                raw = self._config.volume_weight * norm_volume + (
                    1.0 - self._config.volume_weight
                ) * consistency
                reputation = self._blend(updates[i].client_id, group, raw)

                if use_reputation:
                    weights[i][group] = max(reputation, self._config.min_reputation) * volumes[i]
                else:
                    weights[i][group] = volumes[i]
        return weights

    def _blend(self, client_id: int, group: str, raw: float) -> float:
        """EMA-update and return a client's reputation for a group."""
        groups = self._reputation.setdefault(client_id, {})
        previous = groups.get(group, raw)
        decay = self._config.ema_decay
        updated = decay * previous + (1.0 - decay) * raw
        groups[group] = updated
        return updated

    @staticmethod
    def _consensus(
        deltas: dict[int, torch.Tensor], volumes: dict[int, float]
    ) -> torch.Tensor:
        """Volume-weighted mean update direction over a subgraph."""
        total = sum(volumes.values()) or 1.0
        stacked = torch.stack([deltas[i] * volumes[i] for i in deltas], dim=0)
        return stacked.sum(dim=0) / total

    @staticmethod
    def _consistency(delta: torch.Tensor, consensus: torch.Tensor, num_members: int) -> float:
        """Cosine agreement of an update with the consensus, clamped to [0, 1].

        A lone member (or a degenerate zero consensus) has nothing to disagree
        with, so it earns full consistency.
        """
        if num_members <= 1:
            return 1.0
        cos = torch.nn.functional.cosine_similarity(delta, consensus, dim=0, eps=1e-8)
        return float(torch.clamp(cos, min=0.0).item())
