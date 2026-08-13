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

Every arm here follows the shared evaluation protocol
(:mod:`privchain.training.protocol`, ADR-0015) — the same splits, the same
epoch/threshold selection on a held-back slice of train, the same reporting on
the untouched dev split, and the same seeds — as the non-private baseline. Any
other arrangement would charge DP for a difference in protocol rather than in
privacy, which is exactly what earlier revisions of this script did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from privchain.config import (
    load_baseline_config,
    load_privacy_config,
    resolve_device,
)
from privchain.privacy.budget_allocator import PerModalityBudgetAllocator
from privchain.privacy.dp_sgd import resolve_group_sigmas
from privchain.privacy.dp_training import build_dp_arm_config, train_dp_arm
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.protocol import (
    RunResult,
    build_splits,
    format_aggregate,
    repeat_over_seeds,
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

    # Exactly the splits the non-private baseline uses, from the same helper, so
    # the two arms differ only in the privacy mechanism (ADR-0015).
    splits, input_dims = build_splits(base, args.daic_config)
    device = torch.device(resolve_device(base.train.device))
    # The DP arm itself lives in privchain.privacy.dp_training so the sweep, the
    # allocation comparison and the attacker harness cannot drift apart.
    arm = build_dp_arm_config(base, priv, splits, input_dims, device)
    sample_rate, planned_steps = arm.sample_rate, arm.planned_steps

    run_dir = create_run_dir(base.train.output_dir, "phase3", "phase3_dp_budget_sweep")
    save_config(run_dir, {"baseline": base.model_dump(), "privacy": priv.model_dump()})

    # ── 1. Per-modality allocation report (configured allocation) ────────────
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
        "n_train": arm.n_train,
        "n_selection": len(splits.selection),  # type: ignore[arg-type]
        "n_report": len(splits.report),  # type: ignore[arg-type]
        "seeds": list(base.train.seeds),
        "device": str(device),
        "pos_weight": arm.pos_weight,
        "allocation_mode": priv.allocation.mode,
        "delta": priv.delta,
        "sample_rate": sample_rate,
        "expected_batch_size": arm.expected_batch_size,
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
    def train_at(target_eps: float, seed: int) -> dict[str, float]:
        """Train once at one budget and seed; report on the untouched split."""
        if target_eps == float("inf"):
            # σ = 0 disables the noise; clipping stays on so the only difference
            # from the private runs is the perturbation itself.
            group_sigmas = resolve_group_sigmas(dict.fromkeys(MODALITIES, 0.0))
        else:
            allocator = PerModalityBudgetAllocator(
                dict.fromkeys(MODALITIES, target_eps),
                {m: priv.per_modality[m].reidentification_risk for m in MODALITIES}
                if all(m in priv.per_modality for m in MODALITIES)
                else dict.fromkeys(MODALITIES, 0.5),
                delta=priv.delta,
                sample_rate=sample_rate,
                steps=planned_steps,
            )
            group_sigmas = resolve_group_sigmas(allocator.noise_multipliers())
        return train_dp_arm(arm, group_sigmas, seed)

    curve: list[dict[str, Any]] = []
    for target_eps in [float("inf"), *priv.sweep.target_epsilons]:
        private = target_eps != float("inf")
        aggregate, _ = repeat_over_seeds(
            lambda seed, eps=target_eps: RunResult(metrics=train_at(eps, seed)),  # type: ignore[misc]
            base.train.seeds,
        )
        consumed_eps = (
            PerModalityBudgetAllocator(
                dict.fromkeys(MODALITIES, target_eps),
                {m: priv.per_modality[m].reidentification_risk for m in MODALITIES},
                delta=priv.delta,
                sample_rate=sample_rate,
                steps=planned_steps,
            ).consumed_epsilon(planned_steps)["audio"]
            if private
            else float("inf")
        )
        curve.append({"target_epsilon": target_eps, "consumed_epsilon": consumed_eps, **aggregate})
        label = "  inf" if not private else f"{target_eps:5.2f}"
        print(f"eps={label} -> {format_aggregate(aggregate, ('f1', 'roc_auc', 'accuracy'))}")

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
        means = [p[f"{key}_mean"] for p in finite]
        stds = [p[f"{key}_std"] for p in finite]
        # Error bars over seeds: on a 34-session report split the spread is the
        # difference between a trend and a coincidence (ADR-0015).
        line = ax.errorbar(eps, means, yerr=stds, marker="o", capsize=3, label=label)[0]
        if non_private is not None:
            ax.axhline(
                non_private[f"{key}_mean"],
                color=line.get_color(),
                linestyle=":",
                linewidth=1,
                alpha=0.7,
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
