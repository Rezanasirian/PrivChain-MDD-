"""CLI: paired participant-level comparison of the official-test arms (Phase 7, H5).

The campaign reports each arm's mean and spread over seeds, which cannot answer
"is this arm better than that one". This script reads the predictions the
campaign persisted (``official_test_scores.json``), averages each participant's
score across seeds as the pre-registration requires, and bootstraps every pairwise
AUC difference at participant level.

It reads no data and trains nothing, so it can be rerun freely: the official test
split is read once, by the campaign, and this is arithmetic on its output.

Usage:
    python scripts/analyze_official_test.py --run-dir experiments/phase7/<run-id>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from privchain.eval.metrics import paired_bootstrap_auc_difference


def average_scores_over_seeds(
    runs: list[dict[str, Any]], order: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Average one arm's per-seed predictions into one score per participant.

    Seeds are repetitions of the same fit, not independent samples, so they are
    averaged before the participant enters the bootstrap.

    Args:
        runs: One entry per seed, each with ``indices``, ``scores``, ``labels``.
        order: Participant indices defining the output order.

    Returns:
        ``(scores, labels)`` aligned to ``order``.

    Raises:
        ValueError: If a run does not cover exactly ``order``.
    """
    if not runs:
        raise ValueError("no runs to average")
    stacked: list[list[float]] = []
    labels: dict[int, int] = {}
    for run in runs:
        by_index = dict(zip(run["indices"], run["scores"], strict=True))
        if set(by_index) != set(order):
            raise ValueError("a seed scored a different participant set")
        labels = dict(zip(run["indices"], run["labels"], strict=True))
        stacked.append([float(by_index[i]) for i in order])
    return (
        np.asarray(stacked, dtype=np.float64).mean(axis=0),
        np.asarray([labels[i] for i in order], dtype=np.int_),
    )


def main() -> None:
    """Write the pairwise paired-bootstrap table for one campaign run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = json.loads((args.run_dir / "official_test_scores.json").read_text(encoding="utf-8"))
    order = list(payload[next(iter(payload))][0]["indices"])
    scores: dict[str, NDArray[np.float64]] = {}
    labels: NDArray[np.int_] | None = None
    for method, runs in payload.items():
        scores[method], labels = average_scores_over_seeds(runs, order)
    assert labels is not None

    comparisons: dict[str, dict[str, float]] = {}
    for first, second in itertools.combinations(scores, 2):
        comparisons[f"{first}_minus_{second}"] = paired_bootstrap_auc_difference(
            labels, scores[first], scores[second], n_resamples=args.resamples, seed=args.seed
        )
    report = {
        "participants": len(order),
        "positives": int(labels.sum()),
        "seeds": len(payload[next(iter(payload))]),
        "comparisons": comparisons,
    }
    (args.run_dir / "official_test_paired.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Official test — paired participant-level comparisons",
        "",
        f"{len(order)} participants ({int(labels.sum())} positive), "
        f"{report['seeds']} seeds averaged per participant before resampling.",
        "",
        "| comparison | Δ ROC-AUC | 95% CI | p | significant |",
        "|---|---|---|---|:--:|",
    ]
    for name, delta in comparisons.items():
        lines.append(
            f"| {name.replace('_minus_', ' − ')} | {delta['difference']:+.3f} | "
            f"[{delta['low']:+.3f}, {delta['high']:+.3f}] | {delta['p_two_sided']:.3f} | "
            f"{'yes' if delta['significant'] else 'no'} |"
        )
    (args.run_dir / "official_test_paired.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
