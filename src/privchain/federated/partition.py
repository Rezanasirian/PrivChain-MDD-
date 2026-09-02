"""Heterogeneous federated data partitioning (Phase 2, objective H2).

Splits a dataset across N simulated clients and assigns each client a modality
capability vector drawn from the configured population mix (some clients have
all three modalities, some audio+text, some audio-only, etc.). A client that
lacks a modality has that modality **zeroed** (a length-1 zero sequence) — the
naive imputation that plain FedAvg must cope with, which is exactly the failure
mode the Phase 4 protocol is designed to fix.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def assign_capabilities(
    federation: FederationConfig, seed: int
) -> list[tuple[str, tuple[int, int, int]]]:
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
    for pattern, count in zip(patterns, counts, strict=True):
        cap = (pattern.capability[0], pattern.capability[1], pattern.capability[2])
        assignments.extend([(pattern.name, cap)] * count)

    rng = np.random.default_rng(seed)
    rng.shuffle(assignments)
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


def partition_indices_dirichlet(
    labels: Sequence[int], num_clients: int, alpha: float, seed: int
) -> list[list[int]]:
    """Split indices so each client sees a *different* class mix (non-IID).

    IID sharding gives every client roughly the corpus-wide prevalence, which is
    the easiest case for federated averaging and the least like real clinical
    federation: sites differ in referral pattern and severity mix. Drawing each
    client's class proportions from a Dirichlet makes that skew explicit, and it
    is the regime where capability-aware aggregation has the most to prove.

    ``alpha`` controls the skew: small values (0.1) give clients dominated by one
    class, large values approach the IID split.

    Args:
        labels: Binary label per item, index-aligned with the dataset.
        num_clients: Number of clients.
        alpha: Dirichlet concentration; must be positive.
        seed: Seed for the draw.

    Returns:
        ``num_clients`` index lists, together covering every index exactly once.
        Every list is non-empty.

    Raises:
        ValueError: If ``alpha`` is not positive, or there are fewer items than
            clients.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if len(labels) < num_clients:
        raise ValueError(f"cannot partition {len(labels)} items across {num_clients} clients")

    rng = np.random.default_rng(seed)
    shards: list[list[int]] = [[] for _ in range(num_clients)]
    for label in sorted(set(labels)):
        members = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(members)
        proportions = rng.dirichlet([alpha] * num_clients)
        # Cut points along the shuffled class members, so each client takes a
        # different share of this class.
        cuts = (np.cumsum(proportions) * len(members)).astype(int)[:-1]
        for client, chunk in enumerate(np.split(np.asarray(members), cuts)):
            shards[client].extend(int(i) for i in chunk)

    # A Dirichlet draw can leave a client with nothing, which would crash client
    # construction. Move one item from the largest shard rather than silently
    # dropping the client, so the requested population size is honoured.
    for client, shard in enumerate(shards):
        if shard:
            continue
        donor = max(range(num_clients), key=lambda c: len(shards[c]))
        shards[client].append(shards[donor].pop())

    return [sorted(shard) for shard in shards]


def build_client_partitions(
    num_items: int,
    federation: FederationConfig,
    seed: int,
    *,
    labels: Sequence[int] | None = None,
) -> list[ClientPartition]:
    """Build per-client partitions (indices + capability) for the population.

    Args:
        num_items: Number of dataset items to distribute.
        federation: Federation configuration.
        seed: Base seed for capability assignment and index partition.
        labels: Binary label per item, required when
            ``federation.partition.mode`` is ``dirichlet``.

    Returns:
        A list of :class:`ClientPartition`, one per client.

    Raises:
        ValueError: If a label-skewed partition is requested without labels.
    """
    capabilities = assign_capabilities(federation, seed)
    if federation.partition.mode == "dirichlet":
        if labels is None:
            raise ValueError("dirichlet partitioning needs per-item labels")
        shards = partition_indices_dirichlet(
            labels, federation.num_clients, federation.partition.dirichlet_alpha, seed + 1
        )
    else:
        shards = partition_indices(num_items, federation.num_clients, seed + 1)
    return [
        ClientPartition(client_id=cid, pattern_name=name, capability=cap, indices=shard)
        for cid, ((name, cap), shard) in enumerate(zip(capabilities, shards, strict=True))
    ]


def apply_capability_mask(sample: Sample, capability: dict[str, int]) -> Sample:
    """Return a copy of ``sample`` with the modalities it does not hold blanked.

    Shared by the fixed-capability client view (:class:`ModalityMaskedDataset`)
    and the scheduled training view
    (:class:`~privchain.training.capability_schedule.ScheduledCapabilityDataset`).
    Two implementations of "blank a modality" is one too many: they would drift,
    and the centralized number would stop describing the federated one.

    How an absent modality is blanked depends on what the sample is:

    * **Segment-aligned samples** (they carry ``quality``) keep their length.
      Segment ``k`` must mean the same stretch of interview in every branch, so
      collapsing one modality to a single row would break the alignment the
      architecture rests on. The quality vectors are zeroed too, which sets
      ``valid = 0`` at every segment.
    * **Frame-level samples** collapse to a single zero frame. Nothing aligns
      them, and keeping 3000 zero frames per absent modality would make every
      capability-restricted client pay to encode padding.

    Args:
        sample: The unmasked sample.
        capability: ``{modality: 0/1}`` availability flags.

    Returns:
        A new sample; the input is never mutated.
    """
    masked: Sample = dict(sample)  # type: ignore[assignment]
    masked["presence"] = dict(sample["presence"])
    aligned = "quality" in sample
    if aligned:
        masked["quality"] = dict(sample["quality"])
    for modality in CAPABILITY_MODALITIES:
        if capability[modality] == 0:
            features = sample[modality]  # type: ignore[literal-required]
            blank = (
                torch.zeros_like(features)
                if aligned
                else torch.zeros((1, features.shape[1]), dtype=features.dtype)
            )
            masked[modality] = blank  # type: ignore[literal-required]
            masked["presence"][modality] = torch.tensor(0, dtype=torch.long)
            if aligned:
                masked["quality"][modality] = torch.zeros_like(sample["quality"][modality])
    return masked


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
        self._capability = dict(zip(CAPABILITY_MODALITIES, capability, strict=True))

    def __len__(self) -> int:
        """Return the number of items assigned to this client."""
        return len(self._indices)

    def __getitem__(self, index: int) -> Sample:
        """Return the masked sample at local position ``index``.

        How an absent modality is blanked depends on what the sample is:

        * **Segment-aligned samples** (they carry ``quality``) keep their length.
          Segment ``k`` must mean the same stretch of interview in every branch,
          so collapsing one modality to a single row would break the alignment
          the architecture rests on. The quality vectors are zeroed too, which
          sets ``valid = 0`` at every segment.
        * **Frame-level samples** collapse to a single zero frame, as before.
          Nothing aligns them, and keeping 3000 zero frames per absent modality
          would make every capability-restricted client pay to encode padding.

        Args:
            index: Local index in ``[0, len(self))``.

        Returns:
            The capability-masked :class:`Sample`.
        """
        return apply_capability_mask(self._base[self._indices[index]], self._capability)
