"""CLI: capability-aware federation with a blockchain audit trail (Phase 5, H3).

Runs the capability-aware protocol and records each round to the audit ledger —
the per-round aggregation subgraph, per-modality consumed privacy budget (H1),
and per-modality reputation (H2) — then reads the trail back and writes an
``audit_report.json``. With the default ``mock`` backend this runs fully offline
against the in-memory ``MockLedger`` (which enforces the same invariants as the
Go chaincode); point ``configs/blockchain.yaml`` at a Fabric REST gateway
(``backend: fabric_rest``) to exercise the real network.

Usage:
    python scripts/run_federated_with_ledger.py
    python scripts/run_federated_with_ledger.py --rounds 5 --num-clients 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from privchain.chain_client import LedgerError, build_ledger
from privchain.config import (
    CAPABILITY_MODALITIES,
    load_baseline_config,
    load_blockchain_config,
    load_federated_config,
    load_privacy_config,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.federated.client import ClientDPConfig
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import build_federated_clients, run_capability_aware_simulation
from privchain.fusion.factory import build_depression_model
from privchain.privacy.budget_allocator import allocate_target_epsilons
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.loaders import split_dataset


def main() -> None:
    """Run capability-aware federation, record to the ledger, and audit it."""
    parser = argparse.ArgumentParser(description="Federated training with a blockchain audit log.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--blockchain-config", type=Path, default=Path("configs/blockchain.yaml"))
    parser.add_argument("--rounds", type=int, default=None, help="Override num_rounds.")
    parser.add_argument("--num-clients", type=int, default=None, help="Override num_clients.")
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    privacy = load_privacy_config(args.privacy_config)
    blockchain = load_blockchain_config(args.blockchain_config)
    seed_everything(base.seed)

    federation = fed.federation
    if args.rounds is not None:
        federation = federation.model_copy(update={"num_rounds": args.rounds})
    if args.num_clients is not None:
        federation = federation.model_copy(
            update={
                "num_clients": args.num_clients,
                "clients_per_round": min(federation.clients_per_round, args.num_clients),
            }
        )

    full_dataset = MockDaicWozDataset(base.data, seed=base.seed)
    train_subset, val_subset = split_dataset(full_dataset, base.train.val_fraction, base.seed)
    val_loader: DataLoader[Sample] = DataLoader(
        val_subset, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )

    partitions = build_client_partitions(len(train_subset), federation, base.seed)
    input_dims = modality_input_dims(base.data)
    clients = build_federated_clients(
        train_subset,
        partitions,
        input_dims=input_dims,
        model_config=base.model,
        batch_size=base.train.batch_size,
        local_epochs=federation.local_epochs,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        seed=base.seed,
        client_dp=ClientDPConfig(
            target_epsilons=allocate_target_epsilons(
                privacy.privacy.allocation, privacy.privacy.per_modality
            ),
            delta=privacy.privacy.delta,
            max_grad_norm=privacy.privacy.max_grad_norm,
            batch_size=base.train.batch_size,
            num_rounds=federation.num_rounds,
            seed=base.seed,
        ),
    )
    global_model = build_depression_model(input_dims, base.model)

    ledger = build_ledger(blockchain.ledger)
    run_dir = create_run_dir(base.train.output_dir, "phase5", "phase5_federated_with_ledger")
    save_config(
        run_dir,
        {
            "baseline": base.model_dump(),
            "federated": fed.model_dump(),
            "privacy": privacy.model_dump(),
            "blockchain": blockchain.model_dump(),
        },
    )
    print(f"Ledger backend: {blockchain.ledger.backend}")

    run_capability_aware_simulation(
        global_model,
        clients,
        val_loader,
        aggregation=fed.aggregation,
        num_rounds=federation.num_rounds,
        clients_per_round=federation.clients_per_round,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        run_dir=run_dir,
        seed=base.seed,
        ledger=ledger,
    )

    audit = _audit(ledger, federation.num_rounds)
    (run_dir / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"\nRun dir: {run_dir}")
    print(f"Rounds with a published subgraph: {audit['rounds_published']}/{federation.num_rounds}")
    sample = audit["sample_client"]
    if sample is not None:
        print(f"Sample client {sample['client_id']} capability={sample['capability']}")
        for modality, entries in sample["budget_history"].items():
            spent = entries[-1]["epsilon_spent"] if entries else None
            rep = sample["reputation"].get(modality)
            print(f"  {modality:<6} consumed_eps(final)={spent}  reputation={rep}")
    print("\n(On mock noise the metrics are meaningless; this demonstrates the audit trail.)")


def _audit(ledger: object, num_rounds: int) -> dict[str, object]:
    """Read the ledger back into a JSON-serializable audit report.

    Args:
        ledger: The ledger client that was written to.
        num_rounds: Number of rounds executed.

    Returns:
        A report with the published-subgraph count and a per-modality budget /
        reputation trail for one sample client.
    """
    from privchain.chain_client import LedgerClient

    assert isinstance(ledger, LedgerClient)
    rounds_published = 0
    first_subgraph = None
    for round_num in range(1, num_rounds + 1):
        subgraph = ledger.get_subgraph(round_num)
        if subgraph is not None:
            rounds_published += 1
            if first_subgraph is None:
                first_subgraph = subgraph

    sample: dict[str, object] | None = None
    if first_subgraph is not None:
        client_id = first_subgraph.client_ids[0]
        client = ledger.get_client(client_id)
        capability = list(client.capability) if client is not None else None
        budget_history: dict[str, list[dict[str, object]]] = {}
        reputation: dict[str, float] = {}
        for modality in CAPABILITY_MODALITIES:
            history = ledger.budget_history(client_id, modality)
            budget_history[modality] = [
                {
                    "round": r.round,
                    "epsilon_incremental": r.epsilon_incremental,
                    "epsilon_cumulative": r.epsilon_cumulative,
                }
                for r in history
            ]
            rep = ledger.get_reputation(client_id, modality)
            if rep is not None:
                reputation[modality] = rep.score
        sample = {
            "client_id": client_id,
            "capability": capability,
            "budget_history": budget_history,
            "reputation": reputation,
        }

    return {"rounds_published": rounds_published, "sample_client": sample}


if __name__ == "__main__":
    try:
        main()
    except LedgerError as exc:  # a live-Fabric misconfiguration should be legible
        raise SystemExit(f"ledger error: {exc}") from exc
