"""CLI: per-modality DP budget report + accuracy-vs-epsilon sweep (Phase 3).

Satisfies the Phase 3 Definition of Done:
  1. Reports each modality's calibrated noise multiplier and consumed ε budget
     under the configured allocation (``configs/privacy.yaml``).
  2. Sweeps a list of target ε values, trains per-modality DP-SGD at each, and
     plots accuracy/F1 vs ε (the accuracy-vs-ε curve).

Usage:
    python scripts/run_dp_sweep.py                                  # mock data
    python scripts/run_dp_sweep.py --daic-config configs/daic_woz.yaml   # real data
    python scripts/run_dp_sweep.py --config configs/baseline.yaml \
        --privacy-config configs/privacy.yaml

On real data the protocol matches the non-private Phase 1 baseline (ADR-0012):
training on the official train split, evaluation on the official dev split, and
the same class-weighted objective — otherwise the privacy-utility gap would
conflate the cost of DP with a change of protocol.
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
    load_yaml,
    modality_input_dims,
    resolve_device,
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
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    positive_class_weight,
)

MODALITIES = ("audio", "video", "text")


def main() -> None:
    """Run the per-modality DP allocation report and the accuracy-vs-ε sweep."""
    parser = argparse.ArgumentParser(description="Per-modality DP sweep (Phase 3).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument(
        "--daic-config",
        type=Path,
        default=None,
        help="Optional real DAIC-WOZ config; when set, sweeps on real data.",
    )
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    priv = load_privacy_config(args.privacy_config).privacy
    seed_everything(base.seed)

    if args.daic_config is not None:
        # Imported lazily so the mock path has no real-data dependencies.
        from privchain.data.daic_woz import build_daic_woz_dataset

        daic_cfg = load_yaml(args.daic_config)
        train_subset: torch.utils.data.Dataset[Sample] = build_daic_woz_dataset(
            daic_cfg, split="train"
        )
        val_subset: torch.utils.data.Dataset[Sample] = build_daic_woz_dataset(daic_cfg, split="dev")
        # Real feature dims are inferred from the corpus, as in Phase 1.
        input_dims = train_subset.feature_dims
    else:
        full = MockDaicWozDataset(base.data, seed=base.seed)
        train_subset, val_subset = split_dataset(full, base.train.val_fraction, base.seed)
        input_dims = modality_input_dims(base.data)

    val_loader: DataLoader[Sample] = DataLoader(
        val_subset, batch_size=base.train.batch_size, shuffle=False, collate_fn=collate_fn
    )

    batch_size = base.train.batch_size
    n_train = len(train_subset)  # type: ignore[arg-type]
    # Poisson subsampling: each sample enters a step with probability q, which is
    # exactly the assumption the RDP accountant makes (ADR-0004).
    sample_rate = min(1.0, batch_size / n_train)
    expected_batch_size = sample_rate * n_train
    device = torch.device(resolve_device(base.train.device))

    # Match the non-private baseline's objective, including class weighting, so
    # the privacy-utility curve isolates the cost of DP (ADR-0013).
    pos_weight = None
    if base.train.class_weighting:
        weight_loader: DataLoader[Sample] = DataLoader(
            train_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        pos_weight = positive_class_weight(weight_loader)
    objective = DepressionObjective(
        base.data.phq8_max, base.model.phq_loss_weight, pos_weight
    ).to(device)

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
        "dataset": "daic_woz" if args.daic_config is not None else "mock",
        "n_train": n_train,
        "n_eval": len(val_subset),  # type: ignore[arg-type]
        "device": str(device),
        "pos_weight": pos_weight,
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
    # ε = ∞ reference: the same architecture, data, and step budget with the DP
    # mechanism switched off. Without it the curve shows absolute numbers but not
    # the *cost of privacy*, which is what Chapter 4 actually claims (ADR-0013).
    curve: list[dict[str, float]] = []
    for target_eps in [float("inf"), *priv.sweep.target_epsilons]:
        seed_everything(base.seed)
        private = target_eps != float("inf")
        if private:
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
        else:
            # σ = 0 disables the noise; clipping stays on so the only difference
            # from the private runs is the perturbation itself.
            group_sigmas = resolve_group_sigmas(dict.fromkeys(MODALITIES, 0.0))

        model = MultimodalDepressionModel(input_dims, base.model).to(device)
        dp_model = wrap_for_per_sample_grads(model)
        groups = map_parameter_groups(dp_model)
        # Adam, matching the non-private baseline. The "SGD" in DP-SGD refers to
        # the noised, per-sample-clipped *gradient*, not to the optimizer that
        # consumes it; plain SGD at the baseline's lr does not move this model at
        # all, which showed up as F1 = 0 even at ε = ∞ (ADR-0013).
        optimizer = torch.optim.Adam(
            dp_model.parameters(),
            lr=base.train.learning_rate,
            weight_decay=base.train.weight_decay,
        )
        generator = torch.Generator(device=device).manual_seed(base.seed)

        batches = poisson_batches(n_train, sample_rate, planned_steps, generator)

        # Evaluate every epoch and keep the best, matching the non-private
        # baseline's model-selection protocol. Comparing the baseline's *best*
        # epoch against DP's *last* epoch would charge DP for a difference in
        # protocol rather than in privacy (ADR-0013). Evaluating a released model
        # is post-processing, so it costs no additional privacy budget.
        # Selection runs over trained epochs only. Including a pre-training
        # evaluation would let an untrained model win, which silently reported
        # the initial weights' metrics identically at every epsilon.
        steps_per_epoch = max(1, len(batches) // priv.sweep.epochs)
        metrics: dict[str, float] | None = None
        for start in range(0, len(batches), steps_per_epoch):
            dp_train_steps(
                dp_model,
                train_subset,
                batches[start : start + steps_per_epoch],
                objective,
                groups=groups,
                group_sigmas=group_sigmas,
                max_grad_norm=priv.max_grad_norm,
                expected_batch_size=expected_batch_size,
                optimizer=optimizer,
                device=device,
                generator=generator,
            )
            # threshold=None: clipping normalizes away the class weighting, so a
            # fixed 0.5 cut reports F1 = 0 for models that rank perfectly well.
            # Picking the cut is post-processing and costs no budget (ADR-0013).
            epoch_metrics = evaluate_model(model, val_loader, objective, device, threshold=None)
            selector = base.train.selection_metric
            if metrics is None or epoch_metrics[selector] > metrics[selector]:
                metrics = epoch_metrics
        assert metrics is not None  # at least one epoch always runs
        consumed_eps = (
            allocator.consumed_epsilon(planned_steps)["audio"] if private else float("inf")
        )
        point = {
            "target_epsilon": target_eps,
            "consumed_epsilon": consumed_eps,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }
        curve.append(point)
        label = "  inf" if not private else f"{target_eps:5.2f}"
        print(
            f"eps={label} -> acc={metrics['accuracy']:.3f}  "
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

    # ε = ∞ cannot sit on a log axis, so it becomes a horizontal reference line.
    finite = [p for p in curve if p["target_epsilon"] != float("inf")]
    non_private = next((p for p in curve if p["target_epsilon"] == float("inf")), None)

    eps = [p["target_epsilon"] for p in finite]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (("accuracy", "Accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC")):
        line = ax.plot(eps, [p[key] for p in finite], marker="o", label=label)[0]
        if non_private is not None:
            ax.axhline(
                non_private[key], color=line.get_color(), linestyle=":", linewidth=1, alpha=0.7
            )
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε (per modality, log scale)")
    ax.set_ylabel("Validation metric")
    ax.set_title("Accuracy vs. privacy budget (per-modality DP-SGD)")
    if non_private is not None:
        ax.plot([], [], color="grey", linestyle=":", label="ε = ∞ (no noise)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
