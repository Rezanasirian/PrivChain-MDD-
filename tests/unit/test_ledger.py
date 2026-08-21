"""Unit tests for the in-memory audit ledger (Phase 5, H3).

These assert the MockLedger enforces exactly the invariants the Go chaincode
does: one-shot registration, append-only privacy budget, immutable subgraph,
updatable reputation — and the same access control, so an authorization bug
cannot pass offline only to surface against real Fabric.
"""

from __future__ import annotations

import pytest

from privchain.chain_client import (
    Capability,
    LedgerError,
    LedgerIdentity,
    MockLedger,
    build_ledger,
    record_round,
)
from privchain.config import LedgerConfig

CLIENT_0 = LedgerIdentity("ClientOrgMSP", "client-0-cert")
CLIENT_1 = LedgerIdentity("ClientOrgMSP", "client-1-cert")
ATTACKER = LedgerIdentity("OtherOrgMSP", "attacker-cert")


def test_register_and_get_client() -> None:
    ledger = MockLedger()
    ledger.register_client("client-0", (1, 0, 1))
    record = ledger.get_client("client-0")
    assert record is not None
    assert record.capability == (1, 0, 1)
    assert ledger.get_client("missing") is None


def test_register_rejects_duplicate_and_empty_capability() -> None:
    ledger = MockLedger()
    ledger.register_client("client-0", (1, 1, 1))
    with pytest.raises(LedgerError):
        ledger.register_client("client-0", (1, 0, 0))
    with pytest.raises(LedgerError):
        ledger.register_client("client-1", (0, 0, 0))


def test_privacy_budget_is_append_only() -> None:
    ledger = MockLedger()
    ledger.register_client("client-0", (1, 0, 1))
    ledger.log_privacy_budget("client-0", "audio", 1, 0.5)
    # Same (client, modality, round) must not be overwritten.
    with pytest.raises(LedgerError):
        ledger.log_privacy_budget("client-0", "audio", 1, 0.9)
    # A later round is fine.
    ledger.log_privacy_budget("client-0", "audio", 2, 0.8)
    history = ledger.budget_history("client-0", "audio")
    assert [r.round for r in history] == [1, 2]


def test_privacy_budget_validates_inputs() -> None:
    ledger = MockLedger()
    ledger.register_client("client-0", (1, 0, 1))
    with pytest.raises(LedgerError):
        ledger.log_privacy_budget("client-0", "smell", 1, 0.5)  # unknown modality
    with pytest.raises(LedgerError):
        ledger.log_privacy_budget("client-0", "audio", 1, -1.0)  # negative epsilon
    with pytest.raises(LedgerError):
        ledger.log_privacy_budget("ghost", "audio", 1, 0.5)  # unregistered client


def test_subgraph_is_immutable() -> None:
    ledger = MockLedger()
    ledger.publish_subgraph(1, ["client-0", "client-1"])
    with pytest.raises(LedgerError):
        ledger.publish_subgraph(1, ["client-2"])  # re-publish rejected
    ledger.publish_subgraph(2, ["client-0"])
    subgraph = ledger.get_subgraph(1)
    assert subgraph is not None
    assert subgraph.client_ids == ("client-0", "client-1")
    with pytest.raises(LedgerError):
        ledger.publish_subgraph(3, [])  # empty subgraph rejected


def test_reputation_is_updatable_and_bounded() -> None:
    ledger = MockLedger()
    ledger.register_client("client-0", (1, 0, 1))
    ledger.update_reputation("client-0", "audio", 0.7, 1)
    ledger.update_reputation("client-0", "audio", 0.9, 2)  # overwrite allowed
    record = ledger.get_reputation("client-0", "audio")
    assert record is not None
    assert record.score == 0.9 and record.round == 2
    with pytest.raises(LedgerError):
        ledger.update_reputation("client-0", "audio", 1.5, 3)  # out of [0, 1]


# ── access control (parity with the chaincode's identity checks) ────────────


def test_registration_binds_the_client_to_its_registrant() -> None:
    ledger = MockLedger()
    with ledger.acting_as(CLIENT_0):
        ledger.register_client("client-0", (1, 0, 1))
    record = ledger.get_client("client-0")
    assert record is not None
    assert record.owner == CLIENT_0


def test_only_owner_or_coordinator_may_log_a_budget() -> None:
    ledger = MockLedger()
    with ledger.acting_as(CLIENT_0):
        ledger.register_client("client-0", (1, 0, 1))
        ledger.log_privacy_budget("client-0", "audio", 1, 0.5)  # owner: allowed

    # A foreign organization cannot forge this client's privacy accounting.
    with ledger.acting_as(ATTACKER), pytest.raises(LedgerError, match="does not own"):
        ledger.log_privacy_budget("client-0", "audio", 2, 99.0)

    # Nor can a different identity inside the same organization.
    with ledger.acting_as(CLIENT_1), pytest.raises(LedgerError, match="does not own"):
        ledger.log_privacy_budget("client-0", "audio", 2, 99.0)

    # The coordinator may (backfill path).
    ledger.log_privacy_budget("client-0", "audio", 2, 0.6)
    assert len(ledger.budget_history("client-0", "audio")) == 2


def test_reputation_is_coordinator_only() -> None:
    """A client that could write its own reputation could set its own weight."""
    ledger = MockLedger()
    with ledger.acting_as(CLIENT_0):
        ledger.register_client("client-0", (1, 0, 1))
        with pytest.raises(LedgerError, match="only the coordinator"):
            ledger.update_reputation("client-0", "audio", 1.0, 1)

    ledger.update_reputation("client-0", "audio", 0.7, 1)  # coordinator: allowed
    record = ledger.get_reputation("client-0", "audio")
    assert record is not None and record.score == 0.7


def test_subgraph_publication_is_coordinator_only() -> None:
    ledger = MockLedger()
    with ledger.acting_as(CLIENT_0), pytest.raises(LedgerError, match="only the coordinator"):
        ledger.publish_subgraph(1, ["client-0"])
    ledger.publish_subgraph(1, ["client-0"])  # coordinator: allowed


def test_acting_as_restores_the_previous_caller() -> None:
    ledger = MockLedger()
    original = ledger.caller
    with ledger.acting_as(CLIENT_0):
        assert ledger.caller == CLIENT_0
    assert ledger.caller == original


def test_budget_history_is_ordered_beyond_round_ten() -> None:
    """Mirrors the chaincode's zero-padded round key."""
    ledger = MockLedger()
    ledger.register_client("client-0", (1, 0, 1))
    for round_num in (12, 2, 10, 1):
        ledger.log_privacy_budget("client-0", "audio", round_num, 0.1 * round_num)
    rounds = [record.round for record in ledger.budget_history("client-0", "audio")]
    assert rounds == [1, 2, 10, 12]


def test_build_ledger_selects_backend() -> None:
    assert isinstance(build_ledger(LedgerConfig(backend="mock")), MockLedger)
    # fabric_rest constructs without a live network (calls would fail, not init).
    fabric = build_ledger(LedgerConfig(backend="fabric_rest", gateway_url="http://localhost:9"))
    assert fabric.__class__.__name__ == "FabricRestLedger"


def test_registration_is_idempotent_against_a_persistent_ledger() -> None:
    """A second run must not fail on clients an earlier run already registered.

    ``record_round`` tracks registrations in a set, but that set is per-process
    while the ledger is not. Against the real Fabric network the chaincode
    rejects a duplicate ``RegisterClient``, so the second audited run aborted
    with an endorsement failure (ADR-0022).
    """
    ledger = MockLedger()
    participants: list[tuple[str, Capability]] = [("0", (1, 0, 1)), ("1", (1, 1, 1))]

    record_round(ledger, round_num=1, participants=participants, registered=set())
    # A fresh set is exactly what a newly started process brings.
    record_round(ledger, round_num=2, participants=participants, registered=set())

    assert ledger.get_client("0") is not None
    assert ledger.get_client("1") is not None


def test_record_round_accounts_rejected_participants_outside_subgraph() -> None:
    ledger = MockLedger()
    participants: list[tuple[str, Capability]] = [("0", (1, 0, 1)), ("1", (1, 1, 1))]
    spend = {
        client_id: {
            "incremental": {"shared": 0.1},
            "cumulative": {"shared": 0.1},
        }
        for client_id, _ in participants
    }

    record_round(
        ledger,
        round_num=1,
        participants=participants,
        consumed_epsilon=spend,
        subgraph_client_ids=["0"],
        registered=set(),
    )

    assert ledger.get_subgraph(1).client_ids == ("0",)
    assert ledger.budget_history("1", "shared")[0].epsilon_incremental == pytest.approx(0.1)
