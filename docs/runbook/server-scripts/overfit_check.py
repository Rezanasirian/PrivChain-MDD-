"""Diagnostic: can the baseline model overfit a tiny real-data subset?

If training loss does not fall toward zero on ~20 sessions with regularization
disabled, the defect is structural (architecture / gradient flow) rather than a
data-quantity or hyperparameter problem. See ADR-0011.
"""

from __future__ import annotations

import sys

import torch
from torch.utils.data import DataLoader, Subset

from privchain.config import load_baseline_config, load_yaml, resolve_device
from privchain.data.daic_woz import build_daic_woz_dataset
from privchain.data.mock_daic_woz import collate_fn
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.seeding import seed_everything
from privchain.training.objective import DepressionObjective

N_SESSIONS = 20
EPOCHS = 120


def run(lr: float, encoder_type: str, hidden: int) -> tuple[float, float]:
    """Train on a fixed tiny subset and return (first_loss, best_loss)."""
    seed_everything(42)
    config = load_baseline_config("configs/baseline.yaml")
    daic_cfg = load_yaml("configs/daic_woz.yaml")

    full = build_daic_woz_dataset(daic_cfg, split="train")
    # Take a class-balanced handful so the task is learnable in principle.
    pos = [i for i, r in enumerate(full._records) if r["label"] == 1][: N_SESSIONS // 2]
    neg = [i for i, r in enumerate(full._records) if r["label"] == 0][: N_SESSIONS // 2]
    subset = Subset(full, pos + neg)

    loader: DataLoader = DataLoader(
        subset, batch_size=len(pos) + len(neg), shuffle=True, collate_fn=collate_fn
    )

    model_cfg = config.model.model_copy(deep=True)
    model_cfg.encoder.type = encoder_type  # type: ignore[assignment]
    model_cfg.encoder.hidden_dim = hidden
    model_cfg.encoder.out_dim = hidden
    model_cfg.encoder.dropout = 0.0  # regularization off: we WANT overfitting
    model_cfg.fusion.dropout = 0.0
    model_cfg.head.dropout = 0.0

    device = torch.device(resolve_device(config.train.device))
    model = MultimodalDepressionModel(full.feature_dims, model_cfg).to(device)
    objective = DepressionObjective(config.data.phq8_max, 0.0).to(device)  # no PHQ aux term
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # no weight decay

    losses: list[float] = []
    for _ in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for raw in loader:
            batch = {k: v.to(device) for k, v in raw.items()}
            optimizer.zero_grad()
            loss = objective(model(batch), batch)  # type: ignore[arg-type]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        losses.append(epoch_loss / batches)
    return losses[0], min(losses)


if __name__ == "__main__":
    print(f"Overfit check: {N_SESSIONS} balanced sessions, {EPOCHS} epochs, no regularization\n")
    print(f"{'encoder':>8} {'hidden':>7} {'lr':>8} {'first':>9} {'best':>9}  verdict")
    for encoder_type in ("gru", "mean"):
        for lr in (1e-3, 1e-2):
            first, best = run(lr, encoder_type, 64)
            verdict = "CAN overfit" if best < 0.1 else ("partial" if best < 0.4 else "CANNOT fit")
            print(f"{encoder_type:>8} {64:>7} {lr:>8.0e} {first:>9.4f} {best:>9.4f}  {verdict}")
            sys.stdout.flush()
