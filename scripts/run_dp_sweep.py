"""CLI: per-modality DP budget report + accuracy-vs-epsilon sweep (Phase 3).

Satisfies the Phase 3 Definition of Done:
  1. Reports each modality's calibrated noise multiplier and consumed ε budget
     under the configured allocation (``configs/privacy.yaml``).
  2. Sweeps a list of target ε values, trains per-modality DP-SGD at each, and
     plots accuracy/F1 vs ε (the accuracy-vs-ε curve).

Usage:
    python scripts/run_dp_sweep.py
    python scripts/run_dp_sweep.py --config configs/baseline.yaml \
        --privacy-config configs/privacy.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from privchain.config import (
    load_baseline_config,
    load_privacy_config,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
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
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.loaders import split_dataset
from privchain.training.objective import DepressionObjective, evaluate_model

MODALITIES = ("audio", "video", "text")


def main() -> None:
    """Run the per-modality DP allocation report and the accuracy-vs-ε sweep."""
    parser = argparse.ArgumentParser(description="Per-modality DP sweep (Phase 3).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    priv = load_privacy_config(args.privacy_config).privacy
    seed_everything(base.seed)

    full = MockDaicWozDataset(base.data, seed=base.seed)
    train_subset, val_subset = split_dataset(full, base.train.val_fraction, base.seed)
    val_loader: DataLoader[Sample] = DataLoader(
        val_subset, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )

    batch_size = base.train.batch_size
    n_train = len(train_subset)
    # Poisson subsampling: each sample enters a step with probability q, which is
    # exactly the assumption the RDP accountant makes (ADR-0004).
    sample_rate = min(1.0, batch_size / n_train)
    expected_batch_size = sample_rate * n_train
    input_dims = modality_input_dims(base.data)
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight)
    device = torch.device("cpu")

    run_dir = create_run_dir(base.train.output_dir, "phase3", "phase3_dp_budget_sweep")
    save_config(run_dir, {"baseline": base.model_dump(), "privacy": priv.model_dump()})

    # ── 1. Per-modality allocation report (configured allocation) ────────────
    planned_steps = steps_for_epochs(n_train, batch_size, priv.sweep.epochs)
    configured = PerModalityBudgetAllocator.from_config(
        priv.allocation,
        priv.per_modality,
        delta=priv.delta,
        sample_rate=sample_rate,
        steps=planned_steps,
    )
    consumed = configured.consumed_epsilon(planned_steps)
    participant_epsilon = configured.participant_epsilon(planned_steps)
    report = {
        "accountant": "opacus-rdp",
        "sampling": "poisson",
        "allocation_mode": priv.allocation.mode,
        "delta": priv.delta,
        "sample_rate": sample_rate,
        "expected_batch_size": expected_batch_size,
        "planned_steps": planned_steps,
        "per_modality": {
            m: {
                "target_epsilon": a.target_epsilon,
                "reidentification_risk": a.risk,
                "noise_multiplier": a.noise_multiplier,
                "consumed_epsilon": consumed[m],
            }
            for m, a in configured.allocations.items()
        },
        # The budget a subject contributing every modality actually spends: the
        # RDP composition of the three encoders plus the shared head (ADR-0009).
        "participant_epsilon": participant_epsilon,
    }
    (run_dir / "allocation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Per-modality DP allocation (configured):")
    for m, a in configured.allocations.items():
        print(
            f"  {m:5s}  risk={a.risk:.2f}  target_eps={a.target_epsilon:.2f}  "
            f"sigma={a.noise_multiplier:.3f}  consumed_eps={consumed[m]:.3f}"
        )
    print(f"  composed participant epsilon (all groups): {participant_epsilon:.3f}")

    # ── 2. Accuracy-vs-epsilon sweep (uniform per-modality target) ───────────
    curve: list[dict[str, float]] = []
    for target_eps in priv.sweep.target_epsilons:
        seed_everything(base.seed)
        allocator = PerModalityBudgetAllocator(
            {m: target_eps for m in MODALITIES},
            {m: priv.per_modality[m].reidentification_risk for m in MODALITIES}
            if all(m in priv.per_modality for m in MODALITIES)
            else {m: 0.5 for m in MODALITIES},
            delta=priv.delta,
            sample_rate=sample_rate,
            steps=planned_steps,
        )
        group_sigmas = resolve_group_sigmas(allocator.noise_multipliers())

        model = MultimodalDepressionModel(input_dims, base.model).to(device)
        dp_model = wrap_for_per_sample_grads(model)
        groups = map_parameter_groups(dp_model)
        optimizer = torch.optim.SGD(dp_model.parameters(), lr=base.train.learning_rate)
        generator = torch.Generator(device=device).manual_seed(base.seed)

        batches = poisson_batches(n_train, sample_rate, planned_steps, generator)
        dp_train_steps(
            dp_model,
            train_subset,
            batches,
            objective,
            groups=groups,
            group_sigmas=group_sigmas,
            max_grad_norm=priv.max_grad_norm,
            expected_batch_size=expected_batch_size,
            optimizer=optimizer,
            device=device,
            generator=generator,
        )

        metrics = evaluate_model(model, val_loader, objective, device)
        consumed_eps = allocator.consumed_epsilon(planned_steps)["audio"]
        point = {
            "target_epsilon": target_eps,
            "consumed_epsilon": consumed_eps,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }
        curve.append(point)
        print(
            f"eps={target_eps:5.2f} -> acc={metrics['accuracy']:.3f}  "
            f"F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}"
        )

    with (run_dir / "sweep_curve.jsonl").open("w", encoding="utf-8") as handle:
        for point in curve:
            handle.write(json.dumps(point) + "\n")

    _plot_curve(curve, run_dir / "accuracy_vs_epsilon.png")
    print(f"Run dir: {run_dir}")


def _plot_curve(curve: list[dict[str, float]], path: Path) -> None:
    """Plot accuracy/F1/ROC-AUC vs epsilon (no-op if matplotlib is absent)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote sweep_curve.jsonl only.")
        return

    eps = [p["target_epsilon"] for p in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC")):
        ax.plot(eps, [p[key] for p in curve], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε (per modality, log scale)")
    ax.set_ylabel("Validation metric")
    ax.set_title("Accuracy vs. privacy budget (per-modality DP-SGD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
