"""CLI: run heterogeneous federated training (Phase 2).

Combines ``configs/baseline.yaml`` (data + model + local-training hyperparams)
with ``configs/federated.yaml`` (population mix + rounds), partitions the mock
dataset across N clients with heterogeneous modality access, and runs FedAvg.

Backends:
  * ``sim``    — in-house simulator (default; runs offline, no extra deps)
  * ``flower`` — Flower ``start_simulation`` (requires ``pip install flwr``)

Usage:
    python scripts/run_federated.py
    python scripts/run_federated.py --rounds 5 --num-clients 6
    python scripts/run_federated.py --backend flower
"""

from __future__ import annotations

import argparse
from pathlib import Path

from privchain.config import (
    load_baseline_config,
    load_federated_config,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import build_federated_clients, run_simulation
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.loaders import split_dataset
from torch.utils.data import DataLoader


def main() -> None:
    """Parse args, build the federated population, and run a FedAvg simulation."""
    parser = argparse.ArgumentParser(description="Run heterogeneous federated training.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--backend", choices=["sim", "flower"], default="sim")
    parser.add_argument("--rounds", type=int, default=None, help="Override num_rounds.")
    parser.add_argument("--num-clients", type=int, default=None, help="Override num_clients.")
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
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

    # Hold out a full-modality validation set; partition the rest across clients.
    full_dataset = MockDaicWozDataset(base.data, seed=base.seed)
    train_subset, val_subset = split_dataset(full_dataset, base.train.val_fraction, base.seed)
    val_loader: DataLoader[Sample] = DataLoader(
        val_subset, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )

    partitions = build_client_partitions(len(train_subset), federation, base.seed)
    input_dims = modality_input_dims(base.data)
    global_model = MultimodalDepressionModel(input_dims, base.model)

    run_dir = create_run_dir(base.train.output_dir, "phase2", "phase2_fedavg_heterogeneous")
    save_config(run_dir, {"baseline": base.model_dump(), "federated": fed.model_dump()})

    pattern_summary = {p.pattern_name: 0 for p in partitions}
    for p in partitions:
        pattern_summary[p.pattern_name] += 1
    print(f"Clients by pattern: {pattern_summary}")

    if args.backend == "flower":
        from privchain.federated.flower_app import run_flower_simulation

        run_flower_simulation(
            train_subset,
            partitions,
            val_loader,
            input_dims=input_dims,
            model_config=base.model,
            global_model=global_model,
            num_rounds=federation.num_rounds,
            clients_per_round=federation.clients_per_round,
            batch_size=base.train.batch_size,
            local_epochs=federation.local_epochs,
            learning_rate=base.train.learning_rate,
            weight_decay=base.train.weight_decay,
            phq8_max=base.data.phq8_max,
            phq_loss_weight=base.model.phq_loss_weight,
            seed=base.seed,
        )
        print(f"Flower simulation complete. Run dir: {run_dir}")
        return

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
    )
    history = run_simulation(
        global_model,
        clients,
        val_loader,
        num_rounds=federation.num_rounds,
        clients_per_round=federation.clients_per_round,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        run_dir=run_dir,
        seed=base.seed,
    )

    final = history[-1]
    print(f"Run dir: {run_dir}")
    print(
        f"Final global (round {final['round']}) — "
        f"F1={final['val_f1']:.4f}  ROC-AUC={final['val_roc_auc']:.4f}  "
        f"acc={final['val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
