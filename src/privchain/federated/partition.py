"""Heterogeneous federated data partitioning (Phase 2, objective H2).

Splits a dataset across N simulated clients and assigns each client a modality
capability vector drawn from the configured population mix (some clients have
all three modalities, some audio+text, some audio-only, etc.). A client that
lacks a modality has that modality **zeroed** (a length-1 zero sequence) — the
naive imputation that plain FedAvg must cope with, which is exactly the failure
mode the Phase 4 protocol is designed to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from privchain.config import CAPABILITY_MODALITIES, FederationConfig
from privchain.data.mock_daic_woz import Sample


@dataclass(frozen=True)
class ClientPartition:
    """One client's data slice and declared modality capability."""

    client_id: int
    pattern_name: str
    capability: tuple[int, int, int]  # [audio, video, text]
    indices: list[int]


def assign_capabilities(federation: FederationConfig, seed: int) -> list[tuple[str, tuple[int, int, int]]]:
    """Assign a (pattern_name, capability) to each client per the population mix.

    Counts are proportional to each pattern's ``fraction`` (rounded, then
    corrected so they sum to ``num_clients``), and the assignment order is
    shuffled reproducibly.

    Args:
        federation: Federation configuration.
        seed: Seed for the shuffle.

    Returns:
        A list of length ``num_clients`` of ``(pattern_name, capability)``.
    """
    num_clients = federation.num_clients
    patterns = federation.modality_patterns
    counts = [int(round(p.fraction * num_clients)) for p in patterns]

    # Correct rounding drift so counts sum exactly to num_clients.
    drift = num_clients - sum(counts)
    order = sorted(range(len(patterns)), key=lambda i: patterns[i].fraction, reverse=True)
    idx = 0
    while drift != 0 and patterns:
        target = order[idx % len(order)]
        if drift > 0:
            counts[target] += 1
            drift -= 1
        elif counts[target] > 0:
            counts[target] -= 1
            drift += 1
        idx += 1

    assignments: list[tuple[str, tuple[int, int, int]]] = []
    for pattern, count in zip(patterns, counts):
        cap = (pattern.capability[0], pattern.capability[1], pattern.capability[2])
        assignments.extend([(pattern.name, cap)] * count)

    rng = np.random.default_rng(seed)
    rng.shuffle(assignments)  # type: ignore[arg-type]
    return assignments


def partition_indices(num_items: int, num_clients: int, seed: int) -> list[list[int]]:
    """Shuffle and split item indices into ``num_clients`` near-equal shards (IID).

    Args:
        num_items: Total number of items.
        num_clients: Number of clients.
        seed: Seed for the shuffle.

    Returns:
        A list of ``num_clients`` index lists (each non-empty when
        ``num_items >= num_clients``).

    Raises:
        ValueError: If there are fewer items than clients.
    """
    if num_items < num_clients:
        raise ValueError(f"cannot partition {num_items} items across {num_clients} clients")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_items)
    return [sorted(int(i) for i in shard) for shard in np.array_split(shuffled, num_clients)]


def build_client_partitions(
    num_items: int, federation: FederationConfig, seed: int
) -> list[ClientPartition]:
    """Build per-client partitions (indices + capability) for the population.

    Args:
        num_items: Number of dataset items to distribute.
        federation: Federation configuration.
        seed: Base seed for capability assignment and index partition.

    Returns:
        A list of :class:`ClientPartition`, one per client.
    """
    capabilities = assign_capabilities(federation, seed)
    shards = partition_indices(num_items, federation.num_clients, seed + 1)
    return [
        ClientPartition(client_id=cid, pattern_name=name, capability=cap, indices=shard)
        for cid, ((name, cap), shard) in enumerate(zip(capabilities, shards))
    ]


class ModalityMaskedDataset(Dataset[Sample]):
    """View of a base dataset with absent modalities zeroed per a capability.

    Args:
        base: The underlying dataset (mock or real DAIC-WOZ).
        indices: Indices of ``base`` belonging to this client.
        capability: ``[audio, video, text]`` 0/1 availability flags.
    """

    def __init__(
        self, base: Dataset[Sample], indices: list[int], capability: tuple[int, int, int]
    ) -> None:
        self._base = base
        self._indices = indices
        self._capability = dict(zip(CAPABILITY_MODALITIES, capability))

    def __len__(self) -> int:
        """Return the number of items assigned to this client."""
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        """Return the masked sample at local position ``index``.

        Absent modalities are replaced by a length-1 zero sequence with the same
        feature dimension, so encoders still receive a valid (signal-free) input.

        Args:
            index: Local index in ``[0, len(self))``.

        Returns:
            The capability-masked :class:`Sample`.
        """
        sample = self._base[self._indices[index]]
        masked: Sample = dict(sample)  # type: ignore[assignment]
        for modality in CAPABILITY_MODALITIES:
            if self._capability[modality] == 0:
                feat_dim = sample[modality].shape[1]  # type: ignore[literal-required]
                masked[modality] = torch.zeros((1, feat_dim), dtype=sample[modality].dtype)  # type: ignore[literal-required]
        return masked
