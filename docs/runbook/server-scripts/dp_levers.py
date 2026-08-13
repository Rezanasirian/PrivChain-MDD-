"""Diagnostic: which lever recovers DP-SGD utility on 107 sessions? (ADR-0013)

Three levers are swept together at a fixed target epsilon:

* ``q``      — Poisson sampling rate. Larger q averages more per-sample gradients
               per step, so the signal grows while the per-coordinate noise does
               not; the accountant charges more per step in exchange.
* ``C``      — per-sample clipping norm. Bounds the signal *and* scales the noise
               (sigma * C), so it trades bias against variance.
* ``hidden`` — model width. Noise enters per parameter, so the noise vector grows
               as sqrt(#params) while the signal stays bounded by C.

Reported against the epsilon = infinity control for the same lever setting, so a
config that simply cannot train is distinguishable from one that DP breaks.
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
from privchain.privacy.budget_allocator import PerModalityBudgetAllocator
from privchain.privacy.dp_sgd import (
    dp_train_steps,
    map_parameter_groups,
    poisson_batches,
    resolve_group_sigmas,
    wrap_for_per_sample_grads,
)
from privchain.seeding import seed_everything
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    positive_class_weight,
)

MODALITIES = ("audio", "video", "text")
TARGET_EPS = 8.0
STEPS = 200
EVAL_EVERY = 10


def run(
    *,
    train_set: object,
    dev_loader: DataLoader,
    dims: dict[str, int],
    base: object,
    delta: float,
    device: torch.device,
    pos_weight: float | None,
    q: float,
    clip: float,
    hidden: int,
    private: bool,
) -> tuple[float, float]:
    """Train once under these levers; return (best dev F1, best dev AUC)."""
    seed_everything(42)
    n_train = len(train_set)  # type: ignore[arg-type]
    expected_batch = q * n_train

    if private:
        allocator = PerModalityBudgetAllocator(
            dict.fromkeys(MODALITIES, TARGET_EPS),
            dict.fromkeys(MODALITIES, 0.5),
            delta=delta,
            sample_rate=q,
            steps=STEPS,
        )
        sigmas = resolve_group_sigmas(allocator.noise_multipliers())
    else:
        sigmas = resolve_group_sigmas(dict.fromkeys(MODALITIES, 0.0))

    model_cfg = base.model.model_copy(deep=True)  # type: ignore[attr-defined]
    model_cfg.encoder.hidden_dim = hidden
    model_cfg.encoder.out_dim = hidden

    model = MultimodalDepressionModel(dims, model_cfg).to(device)
    dp_model = wrap_for_per_sample_grads(model)
    groups = map_parameter_groups(dp_model)
    objective = DepressionObjective(
        base.data.phq8_max,  # type: ignore[attr-defined]
        model_cfg.phq_loss_weight,
        pos_weight,
    ).to(device)
    optimizer = torch.optim.Adam(
        dp_model.parameters(),
        lr=base.train.learning_rate,  # type: ignore[attr-defined]
        weight_decay=base.train.weight_decay,  # type: ignore[attr-defined]
    )
    generator = torch.Generator(device=device).manual_seed(42)
    batches = poisson_batches(n_train, q, STEPS, generator)

    # Tracked independently: under heavy noise every sigmoid output can fall
    # below 0.5, giving F1 = 0 while the model still *ranks* cases correctly.
    # Gating AUC on an F1 improvement would then report AUC = 0 and hide that.
    best_f1, best_auc = 0.0, 0.0
    for start in range(0, len(batches), EVAL_EVERY):
        dp_train_steps(
            dp_model,
            train_set,  # type: ignore[arg-type]
            batches[start : start + EVAL_EVERY],
            objective,
            groups=groups,
            group_sigmas=sigmas,
            max_grad_norm=clip,
            expected_batch_size=expected_batch,
            optimizer=optimizer,
            device=device,
            generator=generator,
        )
        metrics = evaluate_model(model, dev_loader, objective, device)
        best_f1 = max(best_f1, metrics["f1"])
        if not torch.isnan(torch.tensor(metrics["roc_auc"])):
            best_auc = max(best_auc, metrics["roc_auc"])
    return best_f1, best_auc


def main() -> None:
    """Sweep q x C x width, private and non-private, and print a table."""
    base = load_baseline_config("configs/baseline.yaml")
    daic_cfg = load_yaml("configs/daic_woz.yaml")
    delta = 1.0e-5

    train_set = build_daic_woz_dataset(daic_cfg, split="train")
    dev_set = build_daic_woz_dataset(daic_cfg, split="dev")
    dims = train_set.feature_dims
    device = torch.device(resolve_device(base.train.device))

    dev_loader: DataLoader = DataLoader(
        dev_set, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )
    weight_loader: DataLoader = DataLoader(
        train_set, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )
    pos_weight = positive_class_weight(weight_loader)

    print(f"target_eps={TARGET_EPS}  steps={STEPS}  device={device}  n_train={len(train_set)}")
    print(f"\n{'q':>5} {'C':>5} {'hidden':>7} {'sigma':>8} "
          f"{'F1@8':>7} {'AUC@8':>7} | {'F1@inf':>7} {'AUC@inf':>8}", flush=True)

    common = dict(
        train_set=train_set, dev_loader=dev_loader, dims=dims, base=base,
        delta=delta, device=device, pos_weight=pos_weight,
    )
    for q, clip, hidden in itertools.product((32 / 107, 1.0), (0.1, 1.0), (32, 128)):
        sigma = PerModalityBudgetAllocator(
            dict.fromkeys(MODALITIES, TARGET_EPS), dict.fromkeys(MODALITIES, 0.5),
            delta=delta, sample_rate=q, steps=STEPS,
        ).noise_multipliers()["audio"]

        f1_dp, auc_dp = run(**common, q=q, clip=clip, hidden=hidden, private=True)  # type: ignore[arg-type]
        f1_inf, auc_inf = run(**common, q=q, clip=clip, hidden=hidden, private=False)  # type: ignore[arg-type]
        print(f"{q:>5.2f} {clip:>5.1f} {hidden:>7} {sigma:>8.3f} "
              f"{f1_dp:>7.3f} {auc_dp:>7.3f} | {f1_inf:>7.3f} {auc_inf:>8.3f}", flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
