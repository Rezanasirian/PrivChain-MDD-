"""Tune the DP-SGD clipping norm C on the selection split (ADR-0013 follow-up).

C is not a privacy parameter: the guarantee depends on the noise *multiplier*
sigma, and the injected noise scales as sigma*C alongside a signal bounded by C.
So C trades bias (clipping distortion) against variance (noise relative to
signal) at a fixed epsilon, and can be tuned freely.

It is tuned on the **selection** split, like every other hyperparameter under
ADR-0015; the dev split is not touched here.
"""

from __future__ import annotations

import sys

import torch

from privchain.config import load_baseline_config, load_privacy_config, load_yaml, resolve_device
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.budget_allocator import PerModalityBudgetAllocator
from privchain.privacy.dp_sgd import (
    dp_train_steps,
    map_parameter_groups,
    poisson_batches,
    resolve_group_sigmas,
    steps_for_epochs,
    wrap_for_per_sample_grads,
)
from privchain.seeding import seed_everything
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    positive_class_weight,
)
from privchain.training.protocol import build_splits, format_aggregate, make_loader

MODALITIES = ("audio", "video", "text")
CLIP_NORMS = (0.01, 0.05, 0.1, 0.5, 1.0)
TARGET_EPSILONS = (float("inf"), 8.0)


def main() -> None:
    """Sweep C at a couple of budgets and print selection-split metrics."""
    base = load_baseline_config("configs/baseline.yaml")
    priv = load_privacy_config("configs/privacy.yaml").privacy
    daic = "configs/daic_woz.yaml"

    splits, dims = build_splits(base, __import__("pathlib").Path(daic))
    device = torch.device(resolve_device(base.train.device))
    selection_loader = make_loader(
        splits.selection, batch_size=base.train.batch_size, shuffle=False
    )
    pos_weight = positive_class_weight(
        make_loader(splits.train, batch_size=base.train.batch_size, shuffle=False)
    )
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight, pos_weight).to(
        device
    )

    n_train = len(splits.train)  # type: ignore[arg-type]
    q = min(1.0, base.train.batch_size / n_train)
    expected_batch = q * n_train
    planned_steps = steps_for_epochs(n_train, base.train.batch_size, priv.sweep.epochs)
    steps_per_epoch = max(1, planned_steps // priv.sweep.epochs)

    print(f"n_train={n_train}  q={q:.3f}  steps={planned_steps}  device={device}")
    print(f"\n{'eps':>6} {'C':>7} {'sigma':>8}  selection-split metrics", flush=True)

    for target_eps in TARGET_EPSILONS:
        private = target_eps != float("inf")
        if private:
            allocator = PerModalityBudgetAllocator(
                dict.fromkeys(MODALITIES, target_eps),
                {m: priv.per_modality[m].reidentification_risk for m in MODALITIES},
                delta=priv.delta,
                sample_rate=q,
                steps=planned_steps,
            )
            sigmas = resolve_group_sigmas(allocator.noise_multipliers())
            sigma_shown = sigmas["audio"]
        else:
            sigmas = resolve_group_sigmas(dict.fromkeys(MODALITIES, 0.0))
            sigma_shown = 0.0

        for clip in CLIP_NORMS:
            per_seed: list[dict[str, float]] = []
            for seed in base.train.seeds:
                seed_everything(seed)
                model = MultimodalDepressionModel(dims, base.model).to(device)
                dp_model = wrap_for_per_sample_grads(model)
                optimizer = torch.optim.Adam(
                    dp_model.parameters(),
                    lr=base.train.learning_rate,
                    weight_decay=base.train.weight_decay,
                )
                generator = torch.Generator(device=device).manual_seed(seed)
                batches = poisson_batches(n_train, q, planned_steps, generator)

                best: dict[str, float] | None = None
                for start in range(0, len(batches), steps_per_epoch):
                    dp_train_steps(
                        dp_model,
                        splits.train,
                        batches[start : start + steps_per_epoch],
                        objective,
                        groups=map_parameter_groups(dp_model),
                        group_sigmas=sigmas,
                        max_grad_norm=clip,
                        expected_batch_size=expected_batch,
                        optimizer=optimizer,
                        device=device,
                        generator=generator,
                    )
                    metrics = evaluate_model(
                        model, selection_loader, objective, device, threshold=None
                    )
                    if best is None or metrics["roc_auc"] > best["roc_auc"]:
                        best = metrics
                assert best is not None
                per_seed.append(best)

            mean_auc = sum(m["roc_auc"] for m in per_seed) / len(per_seed)
            mean_f1 = sum(m["f1"] for m in per_seed) / len(per_seed)
            label = "   inf" if not private else f"{target_eps:6.1f}"
            print(
                f"{label} {clip:>7.2f} {sigma_shown:>8.3f}  "
                f"auc={mean_auc:.3f}  f1={mean_f1:.3f}",
                flush=True,
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
