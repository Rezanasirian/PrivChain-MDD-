"""Unit tests for the in-memory audit ledger (Phase 5, H3).

These assert the MockLedger enforces exactly the invariants the Go chaincode
does: one-shot registration, append-only privacy budget, immutable subgraph,
updatable reputation.
"""

from __future__ import annotations

import pytest

from privchain.chain_client import LedgerError, MockLedger, build_ledger
from privchain.config import LedgerConfig


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


def test_build_ledger_selects_backend() -> None:
    assert isinstance(build_ledger(LedgerConfig(backend="mock")), MockLedger)
    # fabric_rest constructs without a live network (calls would fail, not init).
    fabric = build_ledger(LedgerConfig(backend="fabric_rest", gateway_url="http://localhost:9"))
    assert fabric.__class__.__name__ == "FabricRestLedger"
