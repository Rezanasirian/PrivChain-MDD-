"""CLI: privacy-attacker evaluation (Phase 6, objective H5).

Produces the Chapter-4 table of **attack success rate per modality and per
privacy budget** plus a membership-inference curve:

1. Trains the multimodal model briefly on the "member" split (so membership
   inference has something to find), then extracts several noisy embedding views
   per subject for each modality.
2. For a sweep of target ε values, calibrates the released-embedding noise σ(ε)
   with the RDP accountant and runs the re-identification attacker per modality.
3. Runs the same attackers at each modality's *adaptive* ε (the H1 allocation) —
   the headline that the highest-risk modality (audio, smallest ε) is the best
   protected.
4. Runs a loss-threshold membership-inference attack at each ε.

Outputs ``attack_success.json`` (the table), ``attack_curve.jsonl``, and
``attack_success_vs_epsilon.png`` under ``experiments/phase6/<run-id>/``. On mock
noise the depression labels are meaningless, but subject identity is a real
signal, so the re-identification curve is informative; see ADR-0007.

Usage:
    python scripts/run_attack_eval.py
    python scripts/run_attack_eval.py --rounds-epochs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
)
from privchain.eval.embeddings import extract_subject_embeddings, split_enroll_probe
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.accountant import get_noise_multiplier
from privchain.privacy.budget_allocator import allocate_target_epsilons
from privchain.seeding import seed_everything
from privchain.training.experiment import create_run_dir, save_config
from privchain.training.loaders import split_dataset
from privchain.training.objective import DepressionObjective, move_batch_to_device

MODALITIES = ("audio", "video", "text")


def _train_briefly(
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
    """Fit the model on the member split so membership inference has a signal."""
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
        loss = binary_cross_entropy_with_logits(
            logits, batch["label"].float(), reduction="none"
        )
        scores.append((-loss).cpu().numpy().astype(np.float64))
    return np.concatenate(scores)


def _sigma_for_epsilon(epsilon: float, *, sample_rate: float, steps: int, delta: float) -> float:
    """Map a target ε to a noise multiplier σ via the RDP accountant."""
    return get_noise_multiplier(epsilon, sample_rate, steps, delta)


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
    model = MultimodalDepressionModel(input_dims, base.model).to(device)
    _train_briefly(
        model,
        members,
        epochs=args.train_epochs,
        batch_size=base.train.batch_size,
        learning_rate=base.train.learning_rate,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        device=device,
    )

    num_subjects = len(full)
    chance = ReidentificationAttacker.chance_accuracy(num_subjects)
    noise_kwargs = {"sample_rate": atk.sample_rate, "steps": atk.steps, "delta": atk.delta}

    # Clean per-modality enrollment/probe embeddings (identity signal, no DP noise).
    # The released-embedding noise is scaled by each modality's own embedding RMS
    # (its sensitivity), so a unit of sigma is commensurate with the signal.
    enroll_probe: dict[str, tuple] = {}
    emb_rms: dict[str, float] = {}
    for modality in MODALITIES:
        embeddings, subjects, views = extract_subject_embeddings(
            model, full, modality, num_views=atk.num_views, jitter=atk.jitter,
            seed=base.seed, device=device,
        )
        emb_rms[modality] = float(np.sqrt(np.mean(embeddings**2))) or 1.0
        enroll_probe[modality] = split_enroll_probe(embeddings, subjects, views, atk.enroll_views)

    run_dir = create_run_dir(base.train.output_dir, "phase6", "phase6_attack_eval")
    save_config(
        run_dir,
        {"baseline": base.model_dump(), "privacy": priv.model_dump(), "attack": atk.model_dump()},
    )

    # ── 1. Re-identification success per modality across the ε sweep ──────────
    reid: dict[str, list[dict[str, float]]] = {m: [] for m in MODALITIES}
    curve_rows: list[dict[str, float]] = []
    for epsilon in atk.target_epsilons:
        sigma = _sigma_for_epsilon(epsilon, **noise_kwargs)
        row: dict[str, float] = {"target_epsilon": epsilon, "sigma": sigma}
        for modality in MODALITIES:
            noise_std = sigma * atk.noise_scale * emb_rms[modality]
            success = _reid_success(enroll_probe[modality], noise_std, base.seed)
            reid[modality].append(
                {"target_epsilon": epsilon, "sigma": sigma, "success_rate": success}
            )
            row[f"reid_{modality}"] = success
        curve_rows.append(row)
        print(
            f"eps={epsilon:5.2f}  reid: "
            + "  ".join(f"{m}={row[f'reid_{m}']:.3f}" for m in MODALITIES)
        )

    # ── 2. Adaptive allocation headline (per-modality ε from H1) ─────────────
    adaptive_eps = allocate_target_epsilons(priv.allocation, priv.per_modality)
    adaptive: dict[str, dict[str, float]] = {}
    for modality in MODALITIES:
        epsilon = adaptive_eps[modality]
        sigma = _sigma_for_epsilon(epsilon, **noise_kwargs)
        noise_std = sigma * atk.noise_scale * emb_rms[modality]
        success = _reid_success(enroll_probe[modality], noise_std, base.seed)
        adaptive[modality] = {
            "epsilon": epsilon,
            "reidentification_risk": priv.per_modality[modality].reidentification_risk,
            "sigma": sigma,
            "success_rate": success,
        }

    # ── 3. Membership inference across the ε sweep ────────────────────────────
    mia: list[dict[str, float]] = []
    if atk.membership_inference.enabled:
        member_scores = _membership_scores(model, members, device)
        nonmember_scores = _membership_scores(model, nonmembers, device)
        score_scale = float(np.std(np.concatenate([member_scores, nonmember_scores]))) or 1.0
        attacker = MembershipInferenceAttacker()
        for epsilon in atk.target_epsilons:
            sigma = _sigma_for_epsilon(epsilon, **noise_kwargs)
            rng = np.random.default_rng(base.seed)
            std = sigma * atk.noise_scale * score_scale
            noisy_member = member_scores + rng.standard_normal(member_scores.shape) * std
            noisy_nonmember = nonmember_scores + rng.standard_normal(nonmember_scores.shape) * std
            result = attacker.attack(noisy_member, noisy_nonmember)
            mia.append({"target_epsilon": epsilon, "sigma": sigma, **result})

    report = {
        "num_subjects": num_subjects,
        "chance_accuracy": chance,
        "noise_mapping": {"delta": atk.delta, "sample_rate": atk.sample_rate, "steps": atk.steps},
        "reidentification": reid,
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


def _reid_success(enroll_probe: tuple, noise_std: float, seed: int) -> float:
    """Run one re-identification attack at a given embedding-noise level."""
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
