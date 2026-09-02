"""CLI: does the Flower backend agree with the in-house simulator? (Phase 2, H2)

The thesis names Flower as the federated-orchestration framework, but every
federated number in the project comes from the in-house simulator; the Flower
adapter was written and never executed (ADR-0003). Two implementations of the
same protocol, only one of which has ever run, is a claim waiting to fail review.

This is a **parity check, not a second set of results**. Both backends get the
same splits, the same client partitions, the same initial global parameters and
the same seed, and are run for the same small number of rounds. Their final
selection-split metrics should agree closely; if they do not, one of the two is
wrong and the federated results cannot be trusted until that is resolved.

Exact equality is not expected — Flower's simulation samples clients through its
own strategy and RNG — so the tolerance is stated up front rather than fitted
after seeing the numbers.

Usage:
    python scripts/run_flower_parity.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path

import torch

from privchain.config import load_baseline_config, load_federated_config, resolve_device
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import build_federated_clients, run_simulation
from privchain.fusion.factory import build_depression_model
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.objective import (
    build_objective,
    evaluate_model,
)
from privchain.training.protocol import build_splits, labels_of, make_loader

# Both backends run the identical protocol on identical data, so anything beyond
# this is a real disagreement rather than orchestration noise.
TOLERANCE = 0.05


def main() -> None:
    """Run both backends on one matched configuration and compare."""
    parser = argparse.ArgumentParser(description="Flower vs in-house simulator parity check.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=10, help="Kept small; this is a smoke check.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    seed = args.seed if args.seed is not None else base.seed
    seed_everything(seed)

    splits, input_dims = build_splits(base, args.daic_config)
    # Both backends run on the same device — the one the real results were
    # produced on. Flower needs Ray workers to be granted a GPU slice for this to
    # work (see `flower_app.run_flower_simulation`); without it every client
    # fails silently and the comparison scores an untrained model.
    device = resolve_device(base.train.device)
    torch_device = torch.device(device)
    selection_loader = make_loader(
        splits.selection, batch_size=base.train.batch_size, shuffle=False
    )
    objective = build_objective(base.model, base.data.phq8_max).to(torch_device)

    federation = fed.federation.model_copy(update={"num_rounds": args.rounds})
    partitions = build_client_partitions(
        len(splits.train),  # type: ignore[arg-type]
        federation,
        base.seed,
        labels=labels_of(splits.train),
    )

    # One starting point, so any divergence is orchestration and not init.
    seed_everything(seed)
    init_model = build_depression_model(input_dims, base.model, splits.quality_dims)
    init_state: OrderedDict[str, torch.Tensor] = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in init_model.state_dict().items()
    )

    run_dir = create_run_dir(base.train.output_dir, "phase2", "phase2_flower_parity")
    save_config(run_dir, {"baseline": base.model_dump(), "federated": fed.model_dump()})
    client_kwargs = {
        "input_dims": input_dims,
        "model_config": base.model,
        "batch_size": base.train.batch_size,
        "local_epochs": federation.local_epochs,
        "learning_rate": base.train.learning_rate,
        "weight_decay": base.train.weight_decay,
        "phq8_max": base.data.phq8_max,
        "phq_loss_weight": base.model.phq_loss_weight,
        "seed": seed,
        "device": device,
    }

    # ── In-house simulator ───────────────────────────────────────────────────
    sim_dir = run_dir / "in_house"
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim_model = build_depression_model(input_dims, base.model, splits.quality_dims)
    sim_model.load_state_dict(copy.deepcopy(init_state))
    run_simulation(
        sim_model,
        build_federated_clients(splits.train, partitions, **client_kwargs),  # type: ignore[arg-type]
        selection_loader,
        num_rounds=federation.num_rounds,
        clients_per_round=federation.clients_per_round,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        run_dir=sim_dir,
        seed=seed,
        device=device,
    )
    in_house = evaluate_model(sim_model, selection_loader, objective, torch_device, threshold=None)

    # ── Flower ───────────────────────────────────────────────────────────────
    from privchain.federated.flower_app import run_flower_simulation

    flower_model = build_depression_model(input_dims, base.model, splits.quality_dims)
    flower_model.load_state_dict(copy.deepcopy(init_state))
    _history, final_state = run_flower_simulation(
        splits.train,
        partitions,
        selection_loader,
        global_model=flower_model,
        num_rounds=federation.num_rounds,
        clients_per_round=federation.clients_per_round,
        **client_kwargs,  # type: ignore[arg-type]
    )
    if final_state is None:
        raise RuntimeError("Flower returned no aggregated parameters; every client round failed")
    flower_model.load_state_dict(final_state)
    flower = evaluate_model(
        flower_model.to(torch_device), selection_loader, objective, torch_device, threshold=None
    )

    # ── Verdict ──────────────────────────────────────────────────────────────
    compared = ("roc_auc", "f1", "accuracy")
    deltas = {key: abs(in_house[key] - flower[key]) for key in compared}
    agree = all(delta <= TOLERANCE for delta in deltas.values())

    print(f"\nrounds={federation.num_rounds}  clients={len(partitions)}  seed={seed}")
    print(f"{'metric':10s} {'in_house':>10s} {'flower':>10s} {'|delta|':>10s}")
    for key in compared:
        print(f"{key:10s} {in_house[key]:>10.4f} {flower[key]:>10.4f} {deltas[key]:>10.4f}")
    print(
        f"\nverdict: {'AGREE' if agree else 'DISAGREE'} at tolerance {TOLERANCE}"
        + ("" if agree else " — the federated results are blocked until this is resolved")
    )

    (run_dir / "flower_parity.json").write_text(
        json.dumps(
            {
                "tolerance": TOLERANCE,
                "rounds": federation.num_rounds,
                "seed": seed,
                "in_house": in_house,
                "flower": flower,
                "abs_deltas": deltas,
                "agree": agree,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {run_dir / 'flower_parity.json'}")
    raise SystemExit(0 if agree else 1)


if __name__ == "__main__":
    main()
