"""The capability schedule and its single source of patterns (ADR-0028).

Two failure modes are covered here because neither raises on its own: a schedule
that quietly under-represents the rarest capability (the one the primary metric
is about), and a training/evaluation/deployment triple that drifts onto three
different pattern lists.
"""

from __future__ import annotations

import pytest

from privchain.config import ModalityPattern, load_federated_config
from privchain.federated.capability_patterns import (
    DEFAULT_FEDERATED_CONFIG,
    capability_of,
    cycle_counts,
    load_modality_patterns,
)
from privchain.training.capability_schedule import CapabilityScheduler


def _patterns() -> list[ModalityPattern]:
    return load_modality_patterns()


# ── Cycle construction ───────────────────────────────────────────────────────


def test_committed_fractions_become_whole_visits() -> None:
    """0.4/0.3/0.2/0.1 is 4:3:2:1, not four floats that almost add up."""
    assert cycle_counts(_patterns()) == [4, 3, 2, 1]


def test_fractions_are_converted_exactly_not_rounded() -> None:
    """Rounding would zero out a rare pattern; exact rationals cannot."""
    patterns = [
        ModalityPattern(name="a", capability=[1, 0, 0], fraction=0.99),
        ModalityPattern(name="b", capability=[0, 0, 1], fraction=0.01),
    ]

    assert cycle_counts(patterns) == [99, 1]


def test_fractions_that_do_not_sum_to_one_are_refused() -> None:
    patterns = [
        ModalityPattern(name="a", capability=[1, 0, 0], fraction=0.5),
        ModalityPattern(name="b", capability=[0, 0, 1], fraction=0.2),
    ]

    with pytest.raises(ValueError, match="must sum to 1"):
        cycle_counts(patterns)


def test_cycle_is_reduced_to_its_shortest_form() -> None:
    """2:2 is a 2-epoch cycle, not a 10-epoch one that repeats itself five times."""
    patterns = [
        ModalityPattern(name="a", capability=[1, 0, 0], fraction=0.5),
        ModalityPattern(name="b", capability=[0, 0, 1], fraction=0.5),
    ]

    assert cycle_counts(patterns) == [1, 1]


# ── The schedule ─────────────────────────────────────────────────────────────


def test_each_participant_sees_the_deployment_mix_every_cycle() -> None:
    """The property independent per-epoch draws would only satisfy on average."""
    scheduler = CapabilityScheduler(_patterns(), seed=42)

    for participant in range(20):
        assert scheduler.mix_over(participant, scheduler.cycle_length) == {
            "full": 4,
            "audio_text": 3,
            "audio_only": 2,
            "text_only": 1,
        }


def test_the_rarest_capability_is_never_skipped_in_a_cycle() -> None:
    """Over 10 independent draws a participant could plausibly never see it."""
    scheduler = CapabilityScheduler(_patterns(), seed=7)

    for participant in range(50):
        names = {
            scheduler.for_epoch(participant, epoch).name for epoch in range(scheduler.cycle_length)
        }
        assert "text_only" in names


def test_order_varies_between_participants_and_cycles() -> None:
    """A fixed order would correlate every participant's mask with the epoch."""
    scheduler = CapabilityScheduler(_patterns(), seed=42)
    first = [scheduler.for_epoch(0, e).name for e in range(scheduler.cycle_length)]
    other = [scheduler.for_epoch(1, e).name for e in range(scheduler.cycle_length)]
    later = [
        scheduler.for_epoch(0, e).name
        for e in range(scheduler.cycle_length, 2 * scheduler.cycle_length)
    ]

    assert first != other
    assert first != later


def test_schedule_is_reproducible() -> None:
    a = CapabilityScheduler(_patterns(), seed=42)
    b = CapabilityScheduler(_patterns(), seed=42)

    assert [a.for_epoch(3, e).name for e in range(20)] == [
        b.for_epoch(3, e).name for e in range(20)
    ]


def test_a_different_seed_gives_a_different_schedule() -> None:
    a = CapabilityScheduler(_patterns(), seed=42)
    b = CapabilityScheduler(_patterns(), seed=43)

    assert [a.for_epoch(0, e).name for e in range(10)] != [
        b.for_epoch(0, e).name for e in range(10)
    ]


def test_exactly_one_capability_per_participant_step() -> None:
    """Several masks per step would break the participant-level DP sensitivity."""
    scheduler = CapabilityScheduler(_patterns(), seed=42)

    scheduled = scheduler.for_epoch(0, 0)

    assert set(scheduled.capability) == {"audio", "video", "text"}
    assert all(flag in (0, 1) for flag in scheduled.capability.values())
    assert sum(scheduled.capability.values()) >= 1


def test_negative_coordinates_are_refused() -> None:
    scheduler = CapabilityScheduler(_patterns(), seed=42)

    with pytest.raises(ValueError, match="non-negative"):
        scheduler.for_epoch(0, -1)


# ── The contract the three consumers share ───────────────────────────────────


def test_training_evaluation_and_federation_read_one_list() -> None:
    """If these drift, the centralized number stops predicting the federated one."""
    federated = load_federated_config(DEFAULT_FEDERATED_CONFIG).federation.modality_patterns
    loaded = load_modality_patterns()
    scheduler = CapabilityScheduler(loaded, seed=0)

    assert [(p.name, p.capability, p.fraction) for p in loaded] == [
        (p.name, p.capability, p.fraction) for p in federated
    ]
    # The schedule's own view of the mix must match those fractions too.
    assert scheduler.counts == dict(
        zip((p.name for p in federated), cycle_counts(federated), strict=True)
    )


def test_capability_mapping_keeps_the_project_wide_modality_order() -> None:
    """`[audio, video, text]`; a silent reorder would mask the wrong branch."""
    pattern = ModalityPattern(name="audio_only", capability=[1, 0, 0], fraction=1.0)

    assert capability_of(pattern) == {"audio": 1, "video": 0, "text": 0}


def test_committed_patterns_cover_every_kind_of_missingness() -> None:
    """The claim needs a full case, a partial case and a text-free case."""
    names = {pattern.name for pattern in _patterns()}

    assert {"full", "audio_text", "audio_only", "text_only"} <= names
