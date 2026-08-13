"""Is the DP arm's deficit the cost of clipping, or a difference in the path?

With sigma = 0 and C large enough to clip nothing, DP-SGD reduces to SGD on the
same data. If that configuration still trails the plain baseline, the gap is not
the price of privacy machinery but a systematic difference between the two code
paths — Poisson subsampling, the step budget, or the GradSampleModule wrapper.

Both arms use the same splits, the same seeds, and report on the same untouched
split (ADR-0015).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from privchain.config import load_baseline_config, load_privacy_config, resolve_device
from privchain.fusion.baseline_model import MultimodalDepressionModel
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
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import build_splits, make_loader
from privchain.training.trainer import CentralizedTrainer

MODALITIES = ("audio", "video", "text")
NO_CLIP = 1.0e6  # far above any real per-sample gradient norm


def main() -> None:
    """Run both arms under matched conditions and print the comparison."""
    base = load_baseline_config("configs/baseline.yaml")
    priv = load_privacy_config("configs/privacy.yaml").privacy
    splits, dims = build_splits(base, Path("configs/daic_woz.yaml"))
    device = torch.device(resolve_device(base.train.device))

    selection_loader = make_loader(
        splits.selection, batch_size=base.train.batch_size, shuffle=False
    )
    report_loader = make_loader(splits.report, batch_size=base.train.batch_size, shuffle=False)
    pos_weight = positive_class_weight(
        make_loader(splits.train, batch_size=base.train.batch_size, shuffle=False)
    )
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight, pos_weight).to(
        device
    )

    n_train = len(splits.train)  # type: ignore[arg-type]
    q = min(1.0, base.train.batch_size / n_train)
    planned_steps = steps_for_epochs(n_train, base.train.batch_size, priv.sweep.epochs)
    steps_per_epoch = max(1, planned_steps // priv.sweep.epochs)

    def dp_arm(seed: int, clip: float) -> dict[str, float]:
        """DP code path with no noise and the given clipping norm."""
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

        best_score, best_state = -float("inf"), None
        for start in range(0, len(batches), steps_per_epoch):
            dp_train_steps(
                dp_model,
                splits.train,
                batches[start : start + steps_per_epoch],
                objective,
                groups=map_parameter_groups(dp_model),
                group_sigmas=resolve_group_sigmas(dict.fromkeys(MODALITIES, 0.0)),
                max_grad_norm=clip,
                expected_batch_size=q * n_train,
                optimizer=optimizer,
                device=device,
                generator=generator,
            )
            metrics = evaluate_model(model, selection_loader, objective, device, threshold=None)
            if metrics[base.train.selection_metric] > best_score:
                best_score = metrics[base.train.selection_metric]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        return evaluate_with_selected_threshold(
            model, selection_loader, report_loader, objective, device
        )

    def baseline_arm(seed: int) -> dict[str, float]:
        """The plain Phase 1 training path."""
        seed_everything(seed)
        train_loader = make_loader(
            splits.train, batch_size=base.train.batch_size, shuffle=True, seed=seed
        )
        model = MultimodalDepressionModel(dims, base.model)
        trainer = CentralizedTrainer(
            model,
            learning_rate=base.train.learning_rate,
            weight_decay=base.train.weight_decay,
            phq8_max=base.data.phq8_max,
            phq_loss_weight=base.model.phq_loss_weight,
            device=str(device),
            pos_weight=pos_weight,
        )
        run_dir = Path("/tmp/dp_path_check") / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        trainer.fit(
            train_loader,
            selection_loader,
            epochs=base.train.epochs,
            run_dir=run_dir,
            selection_metric=base.train.selection_metric,
            early_stopping_patience=base.train.early_stopping_patience,
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
        return evaluate_with_selected_threshold(
            model, selection_loader, report_loader, objective, device
        )

    print(f"n_train={n_train}  q={q:.3f}  dp_steps={planned_steps}  device={device}\n")
    arms = {
        "baseline (shuffled, no clip)": lambda s: baseline_arm(s),
        f"DP path, sigma=0, C={NO_CLIP:.0e}": lambda s: dp_arm(s, NO_CLIP),
        "DP path, sigma=0, C=0.1": lambda s: dp_arm(s, 0.1),
    }
    for name, arm in arms.items():
        results = [arm(seed) for seed in base.train.seeds]
        auc = sum(r["roc_auc"] for r in results) / len(results)
        f1 = sum(r["f1"] for r in results) / len(results)
        print(f"{name:34s} report-split auc={auc:.3f}  f1={f1:.3f}", flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
