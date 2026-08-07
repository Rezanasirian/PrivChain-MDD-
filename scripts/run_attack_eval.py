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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, Dataset

from privchain.config import (
    load_attack_config,
    load_baseline_config,
    load_privacy_config,
    modality_input_dims,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.eval.attackers import (
    MembershipInferenceAttacker,
    ReidentificationAttacker,
    add_gaussian_noise,
    release_embeddings_dp,
)
from privchain.eval.embeddings import extract_subject_embeddings, split_enroll_probe
from privchain.fusion.baseline_model import MultimodalDepressionModel
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
from privchain.training.objective import DepressionObjective, move_batch_to_device

MODALITIES = ("audio", "video", "text")


def _train_non_private(
    model: MultimodalDepressionModel,
    members: Dataset[Sample],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phq8_max: int,
    phq_loss_weight: float,
    device: torch.device,
) -> None:
    """Fit the model on the member split with no privacy (the ε = ∞ reference)."""
    loader: DataLoader[Sample] = DataLoader(
        members, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    objective = DepressionObjective(phq8_max, phq_loss_weight)
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
    phq_loss_weight: float,
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
        phq_loss_weight: Weight of the PHQ-8 regression term.
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
        DepressionObjective(phq8_max, phq_loss_weight),
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


def main() -> None:
    """Run the attacker evaluation and write the attack-success table."""
    parser = argparse.ArgumentParser(description="Privacy-attacker evaluation (Phase 6).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--attack-config", type=Path, default=Path("configs/attack.yaml"))
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=25,
        help="Member-split fit epochs; enough to overfit so membership inference has a signal.",
    )
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    priv = load_privacy_config(args.privacy_config).privacy
    atk = load_attack_config(args.attack_config).attack
    seed_everything(base.seed)

    device = torch.device("cpu")
    full = MockDaicWozDataset(base.data, seed=base.seed)
    members, nonmembers = split_dataset(full, base.train.val_fraction, base.seed)

    input_dims = modality_input_dims(base.data)
    num_subjects = len(full)
    chance = ReidentificationAttacker.chance_accuracy(num_subjects)
    run_dir = create_run_dir(base.train.output_dir, "phase6", "phase6_attack_eval")
    save_config(
        run_dir,
        {"baseline": base.model_dump(), "privacy": priv.model_dump(), "attack": atk.model_dump()},
    )

    def train_at(target_epsilon: float | None) -> MultimodalDepressionModel:
        """Train a fresh model at a given per-modality ε (``None`` = no privacy)."""
        seed_everything(base.seed)
        model = MultimodalDepressionModel(input_dims, base.model).to(device)
        if target_epsilon is None:
            _train_non_private(
                model,
                members,
                epochs=args.train_epochs,
                batch_size=base.train.batch_size,
                learning_rate=base.train.learning_rate,
                phq8_max=base.data.phq8_max,
                phq_loss_weight=base.model.phq_loss_weight,
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
                phq_loss_weight=base.model.phq_loss_weight,
                device=device,
                seed=base.seed,
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
            embeddings, subjects, views = extract_subject_embeddings(
                model,
                full,
                modality,
                num_views=atk.num_views,
                jitter=atk.jitter,
                seed=base.seed,
                device=device,
            )
            if release_epsilon is not None:
                embeddings = release_embeddings_dp(
                    embeddings,
                    target_epsilon=release_epsilon,
                    delta=atk.delta,
                    clip_norm=atk.embedding_clip_norm,
                    rng=np.random.default_rng(base.seed),
                )
            successes[modality] = _reid_success(
                split_enroll_probe(embeddings, subjects, views, atk.enroll_views), base.seed
            )
        return successes

    attacker = MembershipInferenceAttacker()

    def membership_result(model: MultimodalDepressionModel) -> dict[str, float]:
        """Run the membership-inference attack against one trained model."""
        return attacker.attack(
            _membership_scores(model, members, device),
            _membership_scores(model, nonmembers, device),
            rng=np.random.default_rng(base.seed),
        )

    # ── 1. Sweep: train a real DP-SGD model per ε, then attack *that* model ───
    reid: dict[str, list[dict[str, float]]] = {m: [] for m in MODALITIES}
    curve_rows: list[dict[str, float]] = []
    mia: list[dict[str, float]] = []
    for epsilon in atk.target_epsilons:
        model = train_at(epsilon)
        successes = attack_model(model, epsilon)
        row: dict[str, float] = {"target_epsilon": epsilon}
        for modality in MODALITIES:
            reid[modality].append({"target_epsilon": epsilon, "success_rate": successes[modality]})
            row[f"reid_{modality}"] = successes[modality]
        curve_rows.append(row)
        if atk.membership_inference.enabled:
            mia.append({"target_epsilon": epsilon, **membership_result(model)})
        print(
            f"eps={epsilon:5.2f}  reid: "
            + "  ".join(f"{m}={row[f'reid_{m}']:.3f}" for m in MODALITIES)
        )

    # ── 2. The ε = ∞ reference: same attacks on a non-private model ──────────
    non_private_model = train_at(None)
    non_private = attack_model(non_private_model, None)
    non_private_mia = (
        membership_result(non_private_model) if atk.membership_inference.enabled else {}
    )
    print(
        "eps=  inf  reid: "
        + "  ".join(f"{m}={non_private[m]:.3f}" for m in MODALITIES)
        + "  (no DP)"
    )

    # ── 3. Adaptive allocation headline (per-modality ε from H1) ─────────────
    # Each modality is attacked on a model trained at *its own* budget, which is
    # the claim under test: higher risk -> smaller ε -> better protected.
    adaptive_eps = allocate_target_epsilons(priv.allocation, priv.per_modality)
    adaptive: dict[str, dict[str, float]] = {}
    for modality in MODALITIES:
        epsilon = adaptive_eps[modality]
        successes = attack_model(train_at(epsilon), epsilon)
        adaptive[modality] = {
            "epsilon": epsilon,
            "reidentification_risk": priv.per_modality[modality].reidentification_risk,
            "success_rate": successes[modality],
        }

    report = {
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
        "embedding_clip_norm": atk.embedding_clip_norm,
        "num_subjects": num_subjects,
        "chance_accuracy": chance,
        "delta": priv.delta,
        "reidentification": reid,
        "non_private_reference": {
            "reidentification": non_private,
            "membership_inference": non_private_mia,
        },
        "adaptive_allocation": {"mode": priv.allocation.mode, "per_modality": adaptive},
        "membership_inference": mia,
    }
    (run_dir / "attack_success.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run_dir / "attack_curve.jsonl").open("w", encoding="utf-8") as handle:
        for row in curve_rows:
            handle.write(json.dumps(row) + "\n")
    _plot(curve_rows, chance, run_dir / "attack_success_vs_epsilon.png")

    print(f"\nchance re-identification accuracy = {chance:.3f}")
    print("Adaptive allocation (per-modality budget from H1; higher risk -> smaller eps):")
    for modality in MODALITIES:
        info = adaptive[modality]
        print(
            f"  {modality:5s}  risk={info['reidentification_risk']:.2f}  "
            f"eps={info['epsilon']:.2f}  reid_success={info['success_rate']:.3f}"
        )
    print(f"\nRun dir: {run_dir}")
    print("(On mock data the depression labels are noise; subject identity is real.)")


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
