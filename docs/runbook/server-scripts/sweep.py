"""Hyperparameter sweep for the Phase 1 stats-encoder baseline (ADR-0012).

Datasets are built once and reused across configurations; the on-disk feature
cache makes each run cheap. Selection is on dev F1, with ROC-AUC reported
alongside so a degenerate high-AUC/zero-F1 config cannot win silently.
"""

from __future__ import annotations

import itertools
import sys

import torch
from torch.utils.data import DataLoader

from privchain.config import load_baseline_config, load_yaml, resolve_device
from privchain.data.daic_woz import build_daic_woz_dataset
from privchain.data.mock_daic_woz import collate_fn
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    move_batch_to_device,
    positive_class_weight,
)

EPOCHS = 200
PATIENCE = 40
SEEDS = (42, 7, 2024)


def main() -> None:
    """Run the grid and print a table ordered by mean dev F1."""
    base = load_baseline_config("configs/baseline.yaml")
    daic_cfg = load_yaml("configs/daic_woz.yaml")

    train_set = build_daic_woz_dataset(daic_cfg, split="train")
    dev_set = build_daic_woz_dataset(daic_cfg, split="dev")
    dims = train_set.feature_dims
    device = torch.device(resolve_device(base.train.device))

    train_loader: DataLoader = DataLoader(
        train_set, batch_size=base.train.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader: DataLoader = DataLoader(
        dev_set, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )
    pos_weight = positive_class_weight(train_loader)
    print(f"device={device}  pos_weight={pos_weight:.3f}  dims={dims}", flush=True)

    grid = list(itertools.product((3e-4, 1e-3, 3e-3), (32, 64, 128), (0.3, 0.5)))
    results: list[tuple[float, float, float, dict[str, float]]] = []

    print(f"\n{'lr':>7} {'hidden':>7} {'drop':>5} {'F1':>16} {'AUC':>16}", flush=True)
    for lr, hidden, dropout in grid:
        f1s: list[float] = []
        aucs: list[float] = []
        for seed in SEEDS:
            seed_everything(seed)
            model_cfg = base.model.model_copy(deep=True)
            model_cfg.encoder.hidden_dim = hidden
            model_cfg.encoder.out_dim = hidden
            model_cfg.encoder.dropout = dropout
            model_cfg.fusion.dropout = dropout
            model_cfg.head.dropout = dropout

            model = MultimodalDepressionModel(dims, model_cfg).to(device)
            objective = DepressionObjective(
                base.data.phq8_max, model_cfg.phq_loss_weight, pos_weight
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=base.train.weight_decay
            )

            best_f1, best_auc, stale = 0.0, 0.0, 0
            for _ in range(EPOCHS):
                model.train()
                for raw in train_loader:
                    batch = move_batch_to_device(raw, device)
                    optimizer.zero_grad()
                    objective(model(batch), batch).backward()
                    optimizer.step()
                metrics = evaluate_model(model, dev_loader, objective, device)
                if metrics["f1"] > best_f1:
                    best_f1, best_auc, stale = metrics["f1"], metrics["roc_auc"], 0
                else:
                    stale += 1
                    if stale >= PATIENCE:
                        break
            f1s.append(best_f1)
            aucs.append(best_auc)

        mean_f1 = sum(f1s) / len(f1s)
        spread_f1 = max(f1s) - min(f1s)
        mean_auc = sum(aucs) / len(aucs)
        results.append(
            (mean_f1, mean_auc, spread_f1, {"lr": lr, "hidden": hidden, "drop": dropout})
        )
        print(
            f"{lr:>7.0e} {hidden:>7} {dropout:>5.1f} "
            f"{mean_f1:>9.3f}+-{spread_f1:<5.3f} {mean_auc:>16.3f}",
            flush=True,
        )
        sys.stdout.flush()

    print(f"\n{'=' * 60}\nTop 5 by mean dev F1 over {len(SEEDS)} seeds:")
    for mean_f1, mean_auc, spread, params in sorted(results, reverse=True)[:5]:
        print(f"  F1={mean_f1:.3f} (spread {spread:.3f})  AUC={mean_auc:.3f}  {params}")


if __name__ == "__main__":
    main()
