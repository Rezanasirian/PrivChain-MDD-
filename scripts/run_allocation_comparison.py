"""CLI: does per-modality DP allocation beat a uniform budget? (Phase 3, H1)

H1's central claim is that splitting the privacy budget **per modality**, guided
by re-identification risk, buys more utility than one uniform budget. Testing it
requires the arms to cost the same privacy — and the obvious way to arrange that
is wrong.

A participant contributing every modality is exposed to every mechanism, so what
they actually spend is the RDP *composition* of the three encoders plus the
shared head (ADR-0009), which is dominated by the loosest mechanism and is not
linear in the individual budgets. Matching arms on the sum or mean of their
per-modality ε therefore hands more real privacy to whichever allocation is most
uneven — the adaptive one, i.e. the hypothesis under test. Here every arm is
scaled to the same **composed participant ε** instead
(:func:`~privchain.privacy.budget_allocator.scale_to_participant_epsilon`), and
the run refuses to report anything if that match does not hold.

Three arms, all at that same participant budget:

* ``uniform`` — one budget for all three modalities; what H1 argues against.
* ``adaptive`` — ε ∝ 1/risk using the **measured** risks (ADR-0017).
* ``anti_adaptive`` — ε ∝ risk, the deliberately wrong allocation. Without this
  control a two-arm gap inside the seed spread reads as a result when it is not.

Each arm also reports its per-modality ε, which is the privacy half of the
claim: at equal participant cost, does the adaptive arm really give the
high-risk modalities a tighter budget?

Writes ``allocation_comparison.json`` + ``allocation_comparison.png`` under
``experiments/phase3/<run-id>/``. See ADR-0018.

Usage:
    python scripts/run_allocation_comparison.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from privchain.config import load_baseline_config, load_privacy_config, resolve_device
from privchain.privacy.budget_allocator import (
    PerModalityBudgetAllocator,
    scale_to_participant_epsilon,
)
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
REPORTED = ("f1", "roc_auc", "accuracy")
# The arms are matched to this accuracy; anything looser and the comparison is
# not between allocations but between privacy costs.
MATCH_TOLERANCE = 1.0e-2


def budget_shapes(risks: dict[str, float], sharpness: float) -> dict[str, dict[str, float]]:
    """Return the relative per-modality budget of each arm.

    Only ratios matter here — the absolute level is set later by scaling every
    shape to a common participant ε.

    Args:
        risks: Measured re-identification risk per modality.
        sharpness: Exponent ``γ``; 0 collapses every arm onto uniform.

    Returns:
        ``{arm_name: {modality: relative_budget}}``.
    """
    return {
        "uniform": dict.fromkeys(risks, 1.0),
        # Higher risk -> smaller budget -> more noise. The H1 mechanism.
        "adaptive": {m: risk ** (-sharpness) for m, risk in risks.items()},
        # Higher risk -> larger budget. The control: if this scores like the
        # others, the allocation axis is not doing anything.
        "anti_adaptive": {m: risk**sharpness for m, risk in risks.items()},
    }


def main() -> None:
    """Compare allocation strategies at a matched participant privacy budget."""
    parser = argparse.ArgumentParser(description="Adaptive vs uniform DP allocation (Phase 3).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--daic-config", type=Path, default=None)
    parser.add_argument(
        "--participant-epsilon",
        type=float,
        default=None,
        help="Composed budget every arm must spend; defaults to the configured value.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    priv = load_privacy_config(args.privacy_config).privacy
    seeds = args.seeds if args.seeds else base.train.seeds
    seed_everything(base.seed)

    splits, input_dims = build_splits(base, args.daic_config)
    device = torch.device(resolve_device(base.train.device))
    arm = build_dp_arm_config(base, priv, splits, input_dims, device)

    target = args.participant_epsilon or priv.allocation.total_participant_epsilon
    risks = {m: priv.per_modality[m].reidentification_risk for m in MODALITIES}
    shapes = budget_shapes(risks, priv.allocation.risk_sharpness)

    run_dir = create_run_dir(base.train.output_dir, "phase3", "phase3_allocation_comparison")
    save_config(run_dir, {"baseline": base.model_dump(), "privacy": priv.model_dump()})

    print(f"Run dir: {run_dir}")
    print(
        f"device={device}  n_train={arm.n_train}  q={arm.sample_rate:.3f}  "
        f"steps={arm.planned_steps}  seeds={list(seeds)}"
    )
    print(f"matching every arm to participant epsilon = {target:.3f}\n")

    # ── Scale each shape to the same composed participant epsilon ────────────
    allocations: dict[str, PerModalityBudgetAllocator] = {}
    for name, shape in shapes.items():
        epsilons = scale_to_participant_epsilon(
            shape,
            target,
            delta=priv.delta,
            sample_rate=arm.sample_rate,
            steps=arm.planned_steps,
        )
        allocations[name] = PerModalityBudgetAllocator(
            epsilons, risks, delta=priv.delta, sample_rate=arm.sample_rate, steps=arm.planned_steps
        )

    # The whole comparison rests on this, so it is enforced at runtime rather
    # than merely asserted in a test.
    achieved = {
        name: alloc.participant_epsilon(arm.planned_steps) for name, alloc in allocations.items()
    }
    spread = max(achieved.values()) - min(achieved.values())
    if spread > MATCH_TOLERANCE * target:
        raise RuntimeError(
            f"arms are not budget-matched: participant epsilons {achieved} span {spread:.4f}, "
            f"above the {MATCH_TOLERANCE:.1%} tolerance on {target:.3f}"
        )

    header = f"{'arm':14s} " + " ".join(f"{m:>8s}" for m in MODALITIES) + f" {'participant':>12s}"
    print(header)
    print("-" * len(header))
    for name, alloc in allocations.items():
        budgets = " ".join(f"{alloc.allocations[m].target_epsilon:>8.3f}" for m in MODALITIES)
        print(f"{name:14s} {budgets} {achieved[name]:>12.3f}")

    # ── Train each arm under the shared protocol ─────────────────────────────
    print()
    results: dict[str, dict[str, float]] = {}
    for name, alloc in allocations.items():
        group_sigmas = resolve_group_sigmas(alloc.noise_multipliers())
        aggregate, _ = repeat_over_seeds(
            lambda seed, sigmas=group_sigmas: RunResult(  # type: ignore[misc]
                metrics=train_dp_arm(arm, sigmas, seed)
            ),
            seeds,
        )
        results[name] = aggregate
        print(f"{name:14s} {format_aggregate(aggregate, REPORTED)}")

    payload: dict[str, Any] = {
        "matching": {
            "quantity": "composed participant epsilon (RDP over 3 encoders + shared head)",
            "target": target,
            "achieved": achieved,
            "tolerance": MATCH_TOLERANCE,
            "why": (
                "matching on the sum or mean of per-modality epsilon would give the "
                "most uneven allocation more real privacy; see ADR-0018"
            ),
        },
        "risks": risks,
        "risk_sharpness": priv.allocation.risk_sharpness,
        "seeds": list(seeds),
        "n_train": arm.n_train,
        "n_report": len(splits.report),  # type: ignore[arg-type]
        "arms": {
            name: {
                "epsilon_per_modality": {
                    m: alloc.allocations[m].target_epsilon for m in MODALITIES
                },
                "sigma_per_modality": alloc.noise_multipliers(),
                "participant_epsilon": achieved[name],
                "metrics": results[name],
            }
            for name, alloc in allocations.items()
        },
    }
    (run_dir / "allocation_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _plot(results, run_dir / "allocation_comparison.png", target)

    _print_verdict(results)
    print(f"\nWrote {run_dir / 'allocation_comparison.json'}")


def _print_verdict(results: dict[str, dict[str, float]]) -> None:
    """State whether the arms actually separate, judged against the seed spread."""
    key = "roc_auc"
    ranked = sorted(results, key=lambda name: results[name][f"{key}_mean"], reverse=True)
    best, worst = ranked[0], ranked[-1]
    gap = results[best][f"{key}_mean"] - results[worst][f"{key}_mean"]
    widest_std = max(results[name][f"{key}_std"] for name in results)
    print(f"\nordering by {key}: " + " > ".join(ranked))
    print(
        f"spread {gap:.3f} against a largest per-arm std of {widest_std:.3f} — "
        + (
            "the arms separate."
            if gap > 2 * widest_std
            else "NOT separable at this sample size; report as no measured difference."
        )
    )


def _plot(results: dict[str, dict[str, float]], path: Path, target: float) -> None:
    """Grouped bar chart of each arm's metrics (no-op without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote the JSON only.")
        return

    arms = list(results)
    width = 0.8 / len(REPORTED)
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, key in enumerate(REPORTED):
        ax.bar(
            [i + offset * width for i in range(len(arms))],
            [results[a][f"{key}_mean"] for a in arms],
            width,
            yerr=[results[a][f"{key}_std"] for a in arms],
            capsize=3,
            label=key,
        )
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(arms))])
    ax.set_xticklabels(arms)
    ax.set_ylabel("Dev-split metric")
    ax.set_title(f"DP budget allocation at matched participant ε = {target:.1f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
