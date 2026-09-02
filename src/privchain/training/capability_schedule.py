"""Deterministic per-participant capability schedule (Phase 1, ADR-0028).

The model has to work for a client that holds only audio, so training must show
it that case. The obvious way — forward every participant under all four masks
each step — breaks the privacy accounting: those four views are not independent
observations, and Opacus would clip a gradient that is already a sum over them,
so the per-participant sensitivity the accountant assumes is no longer what the
optimizer enforces.

So each participant contributes **one** masked view per optimizer step, and the
capability varies across epochs instead. Over a full cycle every participant sees
the deployment mix exactly: with ``4:3:2:1`` that is a 10-epoch cycle of 4 full,
3 audio+text, 2 audio-only, 1 text-only.

The cycle is shuffled per (participant, cycle) rather than drawn independently
per epoch. Independent draws leave the rarest pattern to chance — over 10 epochs
a participant could plausibly never be seen text-only — while a shuffled bag
guarantees the mix and still varies the order between participants and cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset

from privchain.config import ModalityPattern
from privchain.data.mock_daic_woz import Sample
from privchain.federated.capability_patterns import capability_of, cycle_counts
from privchain.federated.partition import apply_capability_mask
from privchain.seeding import derive_seed


@dataclass(frozen=True)
class ScheduledCapability:
    """One participant's capability for one epoch.

    Attributes:
        name: The pattern's name, for reporting.
        capability: ``{modality: 0/1}`` availability flags.
    """

    name: str
    capability: dict[str, int]


class CapabilityScheduler:
    """Assigns each (participant, epoch) exactly one capability pattern.

    Args:
        patterns: The declared modality patterns, from
            :func:`~privchain.federated.capability_patterns.load_modality_patterns`.
        seed: Base seed; combined with participant and cycle index so the
            schedule is reproducible and independent of iteration order.

    Raises:
        ValueError: If the patterns' fractions cannot form a whole-number cycle.
    """

    def __init__(self, patterns: list[ModalityPattern], seed: int) -> None:
        self._patterns = list(patterns)
        self._seed = seed
        counts = cycle_counts(self._patterns)
        # The bag holds one entry per visit, so shuffling it yields the mix.
        self._bag: list[int] = [index for index, count in enumerate(counts) for _ in range(count)]
        self.cycle_length = len(self._bag)
        self.counts = dict(zip((p.name for p in self._patterns), counts, strict=True))

    def for_epoch(self, participant_index: int, epoch: int) -> ScheduledCapability:
        """Return the capability this participant trains under at ``epoch``.

        Args:
            participant_index: Stable index of the participant within the split.
            epoch: Zero-based epoch number.

        Returns:
            The scheduled capability.

        Raises:
            ValueError: If ``epoch`` or ``participant_index`` is negative.
        """
        if epoch < 0 or participant_index < 0:
            raise ValueError("participant_index and epoch must be non-negative")
        cycle, position = divmod(epoch, self.cycle_length)
        rng = np.random.default_rng(derive_seed(self._seed, participant_index, cycle))
        order = rng.permutation(self.cycle_length)
        pattern = self._patterns[self._bag[int(order[position])]]
        return ScheduledCapability(name=pattern.name, capability=capability_of(pattern))

    def mix_over(self, participant_index: int, epochs: int) -> dict[str, int]:
        """Count how often each pattern is scheduled over ``epochs`` epochs.

        Exposed so a run can assert it trained the mix it claimed rather than
        assuming the schedule did what the docstring says.

        Args:
            participant_index: Stable index of the participant.
            epochs: Number of epochs to count over.

        Returns:
            ``{pattern_name: visits}``.
        """
        counts = {pattern.name: 0 for pattern in self._patterns}
        for epoch in range(epochs):
            counts[self.for_epoch(participant_index, epoch).name] += 1
        return counts


class ScheduledCapabilityDataset(Dataset[Sample]):
    """A training view whose capability mask changes with the epoch.

    The epoch is set from outside (:meth:`set_epoch`) rather than counted here,
    because a DataLoader with workers copies the dataset into each worker
    process: a counter incremented inside ``__getitem__`` would advance
    independently in every copy, and two participants in the same batch could be
    masked as though they were in different epochs.

    Args:
        base: The unmasked dataset.
        scheduler: The capability schedule.

    Attributes:
        epoch: The epoch currently being served.
    """

    def __init__(self, base: Dataset[Sample], scheduler: CapabilityScheduler) -> None:
        self._base = base
        self._scheduler = scheduler
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Point the view at ``epoch``.

        Args:
            epoch: Zero-based epoch number.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of participants in the underlying split."""
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Sample:
        """Return the participant at ``index`` under this epoch's capability.

        Args:
            index: Participant index within the split.

        Returns:
            The masked sample.
        """
        scheduled = self._scheduler.for_epoch(index, self.epoch)
        return apply_capability_mask(self._base[index], scheduled.capability)

    def scheduled_name(self, index: int) -> str:
        """Return the pattern name this participant trains under right now.

        Args:
            index: Participant index within the split.

        Returns:
            The pattern's name.
        """
        return self._scheduler.for_epoch(index, self.epoch).name
