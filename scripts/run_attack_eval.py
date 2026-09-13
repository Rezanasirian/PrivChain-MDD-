"""CLI: privacy-attacker evaluation (Phase 6, objective H5).

Produces the Chapter-4 table of **attack success rate per modality and per
privacy budget**. For each target ε the pipeline trains a *real* per-modality
DP-SGD model at that budget and attacks it — no non-private model with noise
pasted on afterwards.

The two attacks answer questions about two different mechanisms, and the report
keeps them apart (ADR-0007):

1. **Membership inference** against the DP-SGD-trained model. This is the attack
   DP-SGD actually bounds, so its advantage should collapse toward 0 as ε
   shrinks, while the non-private reference run still leaks.
2. **Re-identification** (speaker-id / face / text de-anonymisation) against the
   **DP release** of an embedding: clipped to a bounded norm and perturbed by the
   Gaussian mechanism at the same ε. DP-SGD alone does *not* prevent this — an
   encoder can map an unseen subject to a distinctive point no matter how it was
   trained — so the release mechanism is what the curve measures.

Both are also run at each modality's *adaptive* ε (the H1 allocation), the
headline that the highest-risk modality (audio, smallest ε) ends up best
protected.

Outputs ``attack_success.json`` (the table), ``attack_curve.jsonl``, and
``attack_success_vs_epsilon.png`` under ``experiments/phase6/<run-id>/``. On mock
noise the depression labels are meaningless, but subject identity is a real
signal, so the re-identification curve is informative; see ADR-0007.

Usage:
    python scripts/run_attack_eval.py
    python scripts/run_attack_eval.py --train-epochs 8
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import (
    ModelConfig,
    load_attack_config,
    load_baseline_config,
    load_privacy_config,
    modality_input_dims,
    resolve_device,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.eval.attackers import (
    MembershipInferenceAttacker,
    ReidentificationAttacker,
    add_gaussian_noise,
    release_embeddings_dp,
)
from privchain.eval.benchmark import stratified_held_out_split
from privchain.eval.embeddings import extract_subject_embeddings, split_enroll_probe
from privchain.eval.session_views import build_views, concat_views
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.fusion.factory import require_baseline_architecture
from privchain.privacy.budget_allocator import (
    PerModalityBudgetAllocator,
    allocate_target_epsilons,
)
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
from privchain.training.objective import build_objective, move_batch_to_device
from privchain.training.protocol import build_splits, labels_of

MODALITIES = ("audio", "video", "text")


def _train_non_private(
    model: MultimodalDepressionModel,
    members: Dataset[Sample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phq8_max: int,
    model_config: ModelConfig,
    device: torch.device,
) -> None:
    """Fit the model on the member split with no privacy (the ε = ∞ reference)."""
    loader: DataLoader[Sample] = DataLoader(
        members, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    objective = build_objective(model_config, phq8_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for raw in loader:
            batch = move_batch_to_device(raw, device)
            optimizer.zero_grad()
            objective(model(batch), batch).backward()
            optimizer.step()


def _train_private(
    model: MultimodalDepressionModel,
    members: Dataset[Sample],
    *,
    target_epsilon: float,
    priv: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phq8_max: int,
    model_config: ModelConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Fit the model with per-modality DP-SGD at a uniform target ``ε``.

    This is what makes the Chapter-4 curve a statement about the *mechanism*:
    the attacked embeddings come from a model actually trained under DP-SGD at
    that budget, not from a non-private model with noise pasted on afterwards.

    Args:
        model: Model to train in place.
        members: The member (training) split.
        target_epsilon: Per-modality target ``ε`` for this point of the sweep.
        priv: Validated privacy config section (``delta``, ``max_grad_norm``).
        epochs: Nominal passes, converted to Poisson steps.
        batch_size: Nominal batch size, setting the sampling rate ``q``.
        learning_rate: SGD learning rate.
        phq8_max: Maximum PHQ-8 score (loss normalisation).
        model_config: Model configuration, for the PHQ-8 loss the arms share.
        device: Torch device.
        seed: Seed for the Poisson draws and the DP noise.

    Returns:
        The per-group noise multipliers actually used.
    """
    num_items = len(members)  # type: ignore[arg-type]
    sample_rate = min(1.0, batch_size / num_items)
    expected_batch_size = sample_rate * num_items
    steps = steps_for_epochs(num_items, batch_size, epochs)

    allocator = PerModalityBudgetAllocator(
        {m: target_epsilon for m in MODALITIES},
        {m: priv.per_modality[m].reidentification_risk for m in MODALITIES},
        delta=priv.delta,
        sample_rate=sample_rate,
        steps=steps,
    )
    group_sigmas = resolve_group_sigmas(allocator.noise_multipliers())

    dp_model = wrap_for_per_sample_grads(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    dp_train_steps(
        dp_model,
        members,
        poisson_batches(num_items, sample_rate, steps, generator),
        build_objective(model_config, phq8_max),
        groups=map_parameter_groups(dp_model),
        group_sigmas=group_sigmas,
        max_grad_norm=priv.max_grad_norm,
        expected_batch_size=expected_batch_size,
        optimizer=torch.optim.SGD(dp_model.parameters(), lr=learning_rate),
        device=device,
        generator=generator,
    )
    return group_sigmas


@torch.no_grad()
def _membership_scores(
    model: MultimodalDepressionModel, subset: Dataset[Sample], device: torch.device
) -> NDArray[np.float64]:
    """Return per-sample membership scores (negative BCE loss; higher ⇒ member)."""
    model.eval()
    loader: DataLoader[Sample] = DataLoader(subset, batch_size=8, collate_fn=collate_fn)
    scores: list[NDArray[np.float64]] = []
    for raw in loader:
        batch = move_batch_to_device(raw, device)
        logits = model(batch)["logit"]
        loss = binary_cross_entropy_with_logits(logits, batch["label"].float(), reduction="none")
        scores.append((-loss).cpu().numpy().astype(np.float64))
    return np.concatenate(scores)


def _pool_labels(dataset: Dataset[Sample]) -> list[int]:
    """Read binary labels through ``Subset`` wrappers without decoding features.

    ``labels_of`` reads a dataset's in-memory records, but a ``Subset`` hides
    them and falls back to indexing, which would decode every session's audio
    and video just to read an integer. Unwrapping first keeps the membership
    split cheap.

    Args:
        dataset: A dataset or a (possibly nested) ``Subset`` of one.

    Returns:
        One binary label per sample, in dataset order.
    """
    if isinstance(dataset, Subset):
        base = _pool_labels(dataset.dataset)
        return [base[i] for i in dataset.indices]
    labels: list[int] = labels_of(dataset)
    return labels


def _balanced_membership_pool(
    datasets: Sequence[Dataset[Sample]], seed: int
) -> tuple[Dataset[Sample], Dataset[Sample]]:
    """Draw equal-sized member / non-member halves from one pooled corpus.

    The ``official`` alternative uses the corpus's own train and dev splits, so
    membership is confounded with whichever split a participant happens to sit
    in, and the groups are unequal (86 against 34), which makes the attacker's
    accuracy a statement about the class prior rather than about leakage. Here
    every participant comes from one pool and a fresh stratified draw per seed
    decides who is trained on, so membership is the only difference between the
    groups. See ADR-0029.

    Args:
        datasets: The datasets to pool (train, selection and report).
        seed: Seed for this repetition's member draw.

    Returns:
        ``(members, nonmembers)``, stratified on the depression label.
    """
    pooled: Dataset[Sample] = ConcatDataset(list(datasets))
    labels = [label for dataset in datasets for label in _pool_labels(dataset)]
    member_idx, nonmember_idx = stratified_held_out_split(labels, held_out_fraction=0.5, seed=seed)
    return Subset(pooled, member_idx), Subset(pooled, nonmember_idx)


def _aggregate_over_seeds(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Summarise one attack point across seeds as mean, std and a 95% interval.

    Args:
        rows: One mapping of metric name to value per seed.

    Returns:
        ``<metric>_mean``, ``<metric>_std`` and ``<metric>_ci95_low/high`` per
        metric, plus ``num_seeds``. The interval is a normal approximation on
        the seed-to-seed standard error, so it describes the spread of the
        estimate and not a per-participant confidence bound.
    """
    aggregate: dict[str, float] = {"num_seeds": float(len(rows))}
    keys = sorted({key for row in rows for key in row})
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        half = 1.96 * std / float(np.sqrt(values.size)) if values.size > 1 else 0.0
        aggregate[f"{key}_mean"] = mean
        aggregate[f"{key}_std"] = std
        aggregate[f"{key}_ci95_low"] = mean - half
        aggregate[f"{key}_ci95_high"] = mean + half
    return aggregate


def main() -> None:
    """Run the attacker evaluation and write the attack-success table."""
    parser = argparse.ArgumentParser(description="Privacy-attacker evaluation (Phase 6).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--attack-config", type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument(
        "--daic-config",
        type=Path,
        default=None,
        help="Real DAIC-WOZ config; uses disjoint session segments instead of jittered views.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=25,
        help="Member-split fit epochs; enough to overfit so membership inference has a signal.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to repeat the whole evaluation over; defaults to the config's single seed.",
    )
    parser.add_argument(
        "--membership-split",
        choices=("official", "balanced"),
        default="official",
        help=(
            "How members and non-members are drawn: 'official' uses the corpus train/dev "
            "splits (unequal, split-confounded); 'balanced' draws equal stratified halves "
            "from the pooled participants, redrawn per seed (ADR-0029)."
        ),
    )
    parser.add_argument(
        "--mia-only",
        action="store_true",
        help=(
            "Skip the re-identification attacks and the adaptive-allocation section, "
            "leaving only the membership-inference sweep. Re-identification risk is "
            "measured at more seeds by scripts/run_reid_risk.py."
        ),
    )
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    # This evaluation trains and probes the encode-then-fuse encoders directly.
    require_baseline_architecture(base.model, "the attacker evaluation")
    priv = load_privacy_config(args.privacy_config).privacy
    atk = load_attack_config(args.attack_config).attack
    seeds = list(args.seeds) if args.seeds else [base.seed]
    if args.mia_only and not atk.membership_inference.enabled:
        raise ValueError("--mia-only requires attack.membership_inference.enabled")
    seed_everything(base.seed)

    device = torch.device(resolve_device(base.train.device))
    full: Dataset[Sample] | None
    view_datasets: list[Dataset[Sample]] = []
    official_members: Dataset[Sample]
    official_nonmembers: Dataset[Sample]
    if args.daic_config is None:
        full = MockDaicWozDataset(base.data, seed=base.seed)
        official_members, official_nonmembers = split_dataset(
            full, base.train.val_fraction, base.seed
        )
        pool_datasets: list[Dataset[Sample]] = [official_members, official_nonmembers]
        input_dims = modality_input_dims(base.data)
        num_subjects = len(full)
    else:
        full = None
        splits, input_dims = build_splits(base, args.daic_config)
        official_members, official_nonmembers = splits.train, splits.report
        view_datasets = [splits.train, splits.selection, splits.report]
        pool_datasets = list(view_datasets)
        num_subjects = sum(len(dataset) for dataset in view_datasets)  # type: ignore[arg-type]
    chance = ReidentificationAttacker.chance_accuracy(num_subjects)
    run_dir = create_run_dir(base.train.output_dir, "phase6", "phase6_attack_eval")
    save_config(
        run_dir,
        {
            "baseline": base.model_dump(),
            "privacy": priv.model_dump(),
            "attack": atk.model_dump(),
            "dataset": "daic_woz" if args.daic_config is not None else "mock",
            "seeds": seeds,
            "membership_split": args.membership_split,
            "mia_only": args.mia_only,
            "train_epochs": args.train_epochs,
        },
    )

    attacker = MembershipInferenceAttacker()

    def run_seed(seed: int) -> dict[str, Any]:
        """Run the whole attacker evaluation once at one seed.

        Args:
            seed: Seed for the member draw, model init, DP noise and the
                attacker's own calibration split.

        Returns:
            This repetition's membership-inference sweep, its non-private
            reference, and — unless ``--mia-only`` — the re-identification
            table and the adaptive-allocation headline.
        """
        if args.membership_split == "balanced":
            members, nonmembers = _balanced_membership_pool(pool_datasets, seed)
        else:
            members, nonmembers = official_members, official_nonmembers

        def train_at(target_epsilon: float | None) -> MultimodalDepressionModel:
            """Train a fresh model at a given parameter-group ε (``None`` = no privacy)."""
            seed_everything(seed)
            model = MultimodalDepressionModel(input_dims, base.model).to(device)
            if target_epsilon is None:
                _train_non_private(
                    model,
                    members,
                    epochs=args.train_epochs,
                    batch_size=base.train.batch_size,
                    learning_rate=base.train.learning_rate,
                    phq8_max=base.data.phq8_max,
                    model_config=base.model,
                    device=device,
                )
            else:
                _train_private(
                    model,
                    members,
                    target_epsilon=target_epsilon,
                    priv=priv,
                    epochs=args.train_epochs,
                    batch_size=base.train.batch_size,
                    learning_rate=base.train.learning_rate,
                    phq8_max=base.data.phq8_max,
                    model_config=base.model,
                    device=device,
                    seed=seed,
                )
            return model

        def attack_model(
            model: MultimodalDepressionModel, release_epsilon: float | None
        ) -> dict[str, float]:
            """Run the three re-identification attackers on one trained model.

            Args:
                model: The (DP-)trained model whose encoders produce the embeddings.
                release_epsilon: Budget for the DP *release* of each embedding, or
                    ``None`` to release it in the clear.

            Returns:
                Top-1 re-identification success per modality.
            """
            successes: dict[str, float] = {}
            for modality in MODALITIES:
                if full is not None:
                    embeddings, subjects, views = extract_subject_embeddings(
                        model,
                        full,
                        modality,
                        num_views=atk.num_views,
                        jitter=atk.jitter,
                        seed=seed,
                        device=device,
                    )
                    enroll_views = atk.enroll_views
                else:
                    offset = 0
                    parts = []
                    for dataset in view_datasets:
                        parts.append(
                            build_views(
                                dataset,
                                modality,
                                num_segments=atk.segments.num_segments,
                                encoder_type=base.model.encoder_for(modality).type,
                                encoder=model.encoders[modality],
                                device=device,
                                subject_offset=offset,
                            )
                        )
                        offset += len(dataset)  # type: ignore[arg-type]
                    segmented = concat_views(*parts)
                    embeddings, subjects, views = (
                        segmented.features,
                        segmented.subject_ids,
                        segmented.view_ids,
                    )
                    enroll_views = atk.segments.enroll_segments
                if release_epsilon is not None:
                    embeddings = release_embeddings_dp(
                        embeddings,
                        target_epsilon=release_epsilon,
                        delta=atk.delta,
                        clip_norm=atk.embedding_clip_norm,
                        rng=np.random.default_rng(seed),
                    )
                successes[modality] = _reid_success(
                    split_enroll_probe(embeddings, subjects, views, enroll_views), seed
                )
            return successes

        def membership_result(model: MultimodalDepressionModel) -> dict[str, float]:
            """Run the membership-inference attack against one trained model."""
            return attacker.attack(
                _membership_scores(model, members, device),
                _membership_scores(model, nonmembers, device),
                rng=np.random.default_rng(seed),
            )

        # ── 1. Sweep: train a real DP-SGD model per ε, then attack *that* model ───
        seed_reid: dict[str, list[dict[str, float]]] = {m: [] for m in MODALITIES}
        seed_curve: list[dict[str, float]] = []
        seed_mia: list[dict[str, float]] = []
        for epsilon in atk.target_epsilons:
            model = train_at(epsilon)
            row: dict[str, float] = {"target_epsilon": epsilon}
            if not args.mia_only:
                successes = attack_model(model, epsilon)
                for modality in MODALITIES:
                    seed_reid[modality].append(
                        {"target_epsilon": epsilon, "success_rate": successes[modality]}
                    )
                    row[f"reid_{modality}"] = successes[modality]
                seed_curve.append(row)
            if atk.membership_inference.enabled:
                mia_row = membership_result(model)
                seed_mia.append({"target_epsilon": epsilon, **mia_row})
            reid_text = (
                ""
                if args.mia_only
                else "  reid: " + "  ".join(f"{m}={row[f'reid_{m}']:.3f}" for m in MODALITIES)
            )
            mia_text = (
                f"  mia: auc={seed_mia[-1]['auc']:.3f} advantage={seed_mia[-1]['advantage']:+.3f}"
                if seed_mia
                else ""
            )
            print(f"seed={seed:<5d} eps={epsilon:6.2f}{mia_text}{reid_text}", flush=True)

        # ── 2. The ε = ∞ reference: same attacks on a non-private model ──────────
        non_private_model = train_at(None)
        non_private = {} if args.mia_only else attack_model(non_private_model, None)
        non_private_mia = (
            membership_result(non_private_model) if atk.membership_inference.enabled else {}
        )
        parts_text = []
        if non_private_mia:
            parts_text.append(
                f"  mia: auc={non_private_mia['auc']:.3f} "
                f"advantage={non_private_mia['advantage']:+.3f}"
            )
        if non_private:
            parts_text.append(
                "  reid: " + "  ".join(f"{m}={non_private[m]:.3f}" for m in MODALITIES)
            )
        print(f"seed={seed:<5d} eps=   inf{''.join(parts_text)}  (no DP)", flush=True)

        # ── 3. Adaptive allocation headline (parameter-group ε from H1) ─────────
        # Each modality is attacked on a model trained at *its own* budget, which is
        # the claim under test: higher risk -> smaller ε -> better protected.
        adaptive: dict[str, dict[str, float]] = {}
        if not args.mia_only:
            adaptive_eps = allocate_target_epsilons(priv.allocation, priv.per_modality)
            for modality in MODALITIES:
                epsilon = adaptive_eps[modality]
                successes = attack_model(train_at(epsilon), epsilon)
                adaptive[modality] = {
                    "epsilon": epsilon,
                    "reidentification_risk": priv.per_modality[modality].reidentification_risk,
                    "success_rate": successes[modality],
                }

        return {
            "seed": seed,
            "num_members": len(members),  # type: ignore[arg-type]
            "num_nonmembers": len(nonmembers),  # type: ignore[arg-type]
            "reidentification": seed_reid,
            "curve": seed_curve,
            "membership_inference": seed_mia,
            "non_private_reference": {
                "reidentification": non_private,
                "membership_inference": non_private_mia,
            },
            "adaptive_allocation": adaptive,
        }

    per_seed = [run_seed(seed) for seed in seeds]
    last = per_seed[-1]

    mia_summary: list[dict[str, float]] = []
    for index, epsilon in enumerate(atk.target_epsilons):
        rows = [
            {k: v for k, v in run["membership_inference"][index].items() if k != "target_epsilon"}
            for run in per_seed
            if index < len(run["membership_inference"])
        ]
        if rows:
            mia_summary.append({"target_epsilon": epsilon, **_aggregate_over_seeds(rows)})
    non_private_rows = [
        run["non_private_reference"]["membership_inference"]
        for run in per_seed
        if run["non_private_reference"]["membership_inference"]
    ]
    non_private_summary = _aggregate_over_seeds(non_private_rows) if non_private_rows else {}

    report: dict[str, Any] = {
        "mechanisms": {
            "membership_inference": (
                "per-modality DP-SGD training at the swept epsilon (Poisson sampling, "
                "Opacus RDP accounting)"
            ),
            "reidentification": (
                "DP release of the embedding: clipped to embedding_clip_norm, then the "
                "Gaussian mechanism at the swept epsilon"
            ),
        },
        "seeds": seeds,
        "membership_split": args.membership_split,
        "mia_only": args.mia_only,
        "num_members": last["num_members"],
        "num_nonmembers": last["num_nonmembers"],
        "embedding_clip_norm": atk.embedding_clip_norm,
        "num_subjects": num_subjects,
        "chance_accuracy": chance,
        "delta": priv.delta,
        "reidentification": last["reidentification"],
        "non_private_reference": last["non_private_reference"],
        "adaptive_allocation": {
            "mode": priv.allocation.mode,
            "per_modality": last["adaptive_allocation"],
        },
        "membership_inference": last["membership_inference"],
        "membership_inference_summary": {
            "per_epsilon": mia_summary,
            "non_private_reference": non_private_summary,
        },
        "per_seed": per_seed,
    }
    (run_dir / "attack_success.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.mia_only:
        with (run_dir / "attack_curve.jsonl").open("w", encoding="utf-8") as handle:
            for row in last["curve"]:
                handle.write(json.dumps(row) + "\n")
        _plot(last["curve"], chance, run_dir / "attack_success_vs_epsilon.png")

    if mia_summary:
        print(
            f"\nMembership inference over {len(seeds)} seed(s), "
            f"{args.membership_split} split "
            f"({last['num_members']} members vs {last['num_nonmembers']} non-members):"
        )
        print(f"{'epsilon':>8}  {'AUC':>16}  {'AUC 95% CI':>20}  {'advantage':>16}")
        for row in mia_summary:
            _print_mia_row(f"{row['target_epsilon']:8.2f}", row)
        if non_private_summary:
            _print_mia_row(f"{'inf':>8}", non_private_summary)

    if not args.mia_only:
        print(f"\nchance re-identification accuracy = {chance:.3f}")
        print("Adaptive allocation (per-modality budget from H1; higher risk -> smaller eps):")
        for modality in MODALITIES:
            info = last["adaptive_allocation"][modality]
            print(
                f"  {modality:5s}  risk={info['reidentification_risk']:.2f}  "
                f"eps={info['epsilon']:.2f}  reid_success={info['success_rate']:.3f}"
            )
    print(f"\nRun dir: {run_dir}")
    if args.daic_config is None:
        print("(On mock data the depression labels are noise; subject identity is real.)")


def _print_mia_row(label: str, row: Mapping[str, float]) -> None:
    """Print one aggregated membership-inference row of the summary table.

    Args:
        label: Left-hand column, already width-formatted (an ε or ``inf``).
        row: Aggregate produced by :func:`_aggregate_over_seeds`.
    """
    auc = f"{row['auc_mean']:.3f}±{row['auc_std']:.3f}"
    interval = f"[{row['auc_ci95_low']:.3f}, {row['auc_ci95_high']:.3f}]"
    advantage = f"{row['advantage_mean']:+.3f}±{row['advantage_std']:.3f}"
    print(f"{label}  {auc:>16}  {interval:>20}  {advantage:>16}")


def _reid_success(enroll_probe: tuple, seed: int, noise_std: float = 0.0) -> float:
    """Run one re-identification attack on a trained model's embeddings.

    Args:
        enroll_probe: ``(enroll_emb, enroll_ids, probe_emb, probe_ids)``.
        seed: Seed for any additional embedding noise.
        noise_std: Optional extra Gaussian noise on the released embeddings. The
            main sweep leaves this at 0 — the protection under test comes from
            DP-SGD training, not from post-hoc perturbation.

    Returns:
        Top-1 re-identification accuracy.
    """
    enroll_emb, enroll_ids, probe_emb, probe_ids = enroll_probe
    rng = np.random.default_rng(seed)
    attacker = ReidentificationAttacker()
    attacker.enroll(add_gaussian_noise(enroll_emb, noise_std, rng), enroll_ids)
    return attacker.attack(add_gaussian_noise(probe_emb, noise_std, rng), probe_ids)


def _plot(rows: list[dict[str, float]], chance: float, path: Path) -> None:
    """Plot re-identification success vs ε per modality (no-op without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote attack_curve.jsonl only.")
        return

    eps = [r["target_epsilon"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for modality in MODALITIES:
        ax.plot(eps, [r[f"reid_{modality}"] for r in rows], marker="o", label=modality)
    ax.axhline(chance, linestyle="--", color="grey", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget ε (per modality, log scale)")
    ax.set_ylabel("Re-identification success rate")
    ax.set_title("Attacker success vs. privacy budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
