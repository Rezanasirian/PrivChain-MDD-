"""CLI: final comparative evaluation for Chapter 4 (Phase 7, objective H5).

Runs one shared 10-fold cross-validation (+ a held-out test fold) over every
method variant and writes the Chapter-4 tables/plots:

* **Main comparison** — centralized (no FL/DP), plain FedAvg, a personalized
  variant, and the full proposed framework.
* **Ablations** — the proposed framework minus reputation, minus distillation.
* **DP privacy–utility** — per-modality *adaptive* allocation vs a *uniform*
  budget of the same total ε (centralized DP-SGD).
* **Inference latency** — forward-pass latency across batch sizes (compute
  budget).

Each named variant also maps to a representative baseline from the literature
(centralized ≈ Xu et al. 2023, uniform-DP ≈ De Chaudhury et al. 2024,
personalized ≈ Fan et al. 2025); these are **simplified stand-ins** on the same
data/model, not faithful reimplementations — see ADR-0008.

Outputs live under ``experiments/phase7/<run-id>/``: ``cv_results.json``,
``ablation.json``, ``dp_comparison.json``, ``latency.json``, and a combined
``chapter4_summary.md``. On mock data the depression label is random, so the
accuracy numbers are placeholders (as in earlier phases); the harness and table
shapes are what Phase 7 delivers, and DAIC-WOZ fills in the real numbers.

On real data (``--daic-config``) the folds are drawn from the **pooled official
train+dev split** and the AVEC2017 ``full_test_split`` is reserved for a single
read per method at the end, written to ``official_test.json`` (ADR-0023). Every
arm follows the same protocol as ``scripts/train_baseline.py``: a selection split
carved out of each fold's training data drives early stopping and the decision
threshold, so no fold reports a number it also tuned on (ADR-0015).

Usage:
    python scripts/run_final_evaluation.py
    python scripts/run_final_evaluation.py --k-folds 5 --rounds 3
    python scripts/run_final_evaluation.py --daic-config configs/daic_woz.yaml
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from privchain.config import (
    AggregationConfig,
    load_baseline_config,
    load_evaluation_config,
    load_federated_config,
    load_privacy_config,
    load_yaml,
    modality_input_dims,
    resolve_device,
)
from privchain.data.mock_daic_woz import MockDaicWozDataset, Sample, collate_fn
from privchain.eval.benchmark import (
    aggregate_metrics,
    measure_inference_latency,
    stratified_held_out_split,
    stratified_k_fold_indices,
)
from privchain.federated.partition import build_client_partitions
from privchain.federated.simulation import (
    build_federated_clients,
    run_capability_aware_simulation,
    run_simulation,
)
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.budget_allocator import (
    PerModalityBudgetAllocator,
    allocate_target_epsilons,
    scale_to_participant_epsilon,
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
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import carve_selection_split, labels_of
from privchain.training.trainer import CentralizedTrainer

MODALITIES = ("audio", "video", "text")
FoldRunner = Callable[[list[int], list[int], int], dict[str, float]]


def _loader(
    dataset: Any, indices: list[int], batch_size: int, *, shuffle: bool = False
) -> DataLoader:
    """Build a padded DataLoader over a subset of ``dataset``."""
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def _build_corpus(
    base: Any, daic_config: Path | None
) -> tuple[Any, list[int], dict[str, int], list[int]]:
    """Assemble the pool the folds are drawn from, plus the reserved test split.

    On real data the folds come from the **pooled official train+dev split**, and
    ``full_test_split`` is held back for a single final read (ADR-0023). Audio and
    video both use per-participant (``session``) normalization, so pooling
    introduces no cross-participant statistics leakage into the folds.

    The official test sessions are concatenated *after* the pool and returned as
    indices rather than a separate dataset, so the same index-based fold runners
    score them without a second code path.

    Args:
        base: Validated baseline config.
        daic_config: Real-data config path, or ``None`` for mock data.

    Returns:
        ``(dataset, pool_labels, input_dims, official_test_idx)``. The label list
        covers the pool only; ``official_test_idx`` is empty on the mock path,
        which has no official split to reserve.
    """
    if daic_config is None:
        mock = MockDaicWozDataset(base.data, seed=base.seed)
        return mock, labels_of(mock), modality_input_dims(base.data), []

    # Imported lazily so the mock path keeps no real-data dependencies.
    from privchain.data.daic_woz import build_daic_woz_dataset

    cfg = load_yaml(daic_config)
    train_ds = build_daic_woz_dataset(cfg, split="train")
    dev_ds = build_daic_woz_dataset(cfg, split="dev")
    test_ds = build_daic_woz_dataset(cfg, split="test")
    # Labels come from the split records, not by indexing: materializing 188
    # sessions of audio and video just to read a label costs minutes.
    pool_labels = labels_of(train_ds) + labels_of(dev_ds)
    official_idx = [len(pool_labels) + i for i in range(len(labels_of(test_ds)))]
    return (
        ConcatDataset([train_ds, dev_ds, test_ds]),
        pool_labels,
        train_ds.feature_dims,
        official_idx,
    )


def _carve_fold(
    full: Any, train_idx: list[int], labels_all: list[int], *, selection_fraction: float, seed: int
) -> tuple[Dataset[Sample], Dataset[Sample]]:
    """Split a fold's training indices into (train, selection).

    Keeping epoch and threshold selection off the scored split is what makes each
    fold's number an estimate rather than a best-of-N (ADR-0015).
    """
    return carve_selection_split(
        Subset(full, train_idx),
        [labels_all[i] for i in train_idx],
        selection_fraction=selection_fraction,
        seed=seed,
    )


def _eval_centralized(
    full: Any,
    train_idx: list[int],
    test_idx: list[int],
    *,
    base: Any,
    input_dims: dict[str, int],
    labels_all: list[int],
    device: torch.device,
    epochs: int,
    seed: int,
) -> dict[str, float]:
    """Train the centralized baseline on a fold and score the test split.

    Follows the same protocol as ``scripts/train_baseline.py``: early-stop and
    pick the decision threshold on a selection split carved out of the fold's
    training data, never on the split being reported.
    """
    seed_everything(seed)
    fold_train, selection = _carve_fold(
        full, train_idx, labels_all, selection_fraction=base.train.selection_fraction, seed=seed
    )
    batch_size = base.train.batch_size
    train_loader: DataLoader[Sample] = DataLoader(
        fold_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    selection_loader: DataLoader[Sample] = DataLoader(
        selection, batch_size=batch_size, collate_fn=collate_fn
    )
    # Measured per fold, not configured: each fold has its own class balance.
    pos_weight = (
        positive_class_weight(DataLoader(fold_train, batch_size=batch_size, collate_fn=collate_fn))
        if base.train.class_weighting
        else None
    )
    model = MultimodalDepressionModel(input_dims, base.model)
    trainer = CentralizedTrainer(
        model,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        device=str(device),
        pos_weight=pos_weight,
    )
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trainer.fit(
            train_loader,
            selection_loader,
            epochs=epochs,
            run_dir=run_dir,
            selection_metric=base.train.selection_metric,
            early_stopping_patience=base.train.early_stopping_patience,
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight).to(device)
    return evaluate_with_selected_threshold(
        model, selection_loader, _loader(full, test_idx, batch_size), objective, device
    )


def _eval_centralized_dp(
    full: Any,
    train_idx: list[int],
    test_idx: list[int],
    *,
    base: Any,
    priv: Any,
    input_dims: dict[str, int],
    device: torch.device,
    epochs: int,
    seed: int,
    mode: str,
) -> dict[str, float]:
    """Train centralized per-modality DP-SGD (adaptive or uniform ε) on a fold."""
    seed_everything(seed)
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight).to(device)

    risks = {m: priv.per_modality[m].reidentification_risk for m in MODALITIES}
    batch_size = base.train.batch_size
    n_train = len(train_idx)
    sample_rate = min(1.0, batch_size / n_train)
    expected_batch_size = sample_rate * n_train
    steps = steps_for_epochs(n_train, batch_size, epochs)

    # Both arms are scaled to the same *composed participant* epsilon. This used
    # to match them on the arithmetic mean of the per-modality budgets, which is
    # not the same privacy cost: composition is dominated by the loosest
    # mechanism, so the mean quietly gave the (uneven) adaptive arm more real
    # privacy than the uniform one -- biasing the comparison toward the
    # hypothesis under test. See ADR-0018.
    adaptive = allocate_target_epsilons(priv.allocation, priv.per_modality)
    shape = dict.fromkeys(MODALITIES, 1.0) if mode == "uniform" else adaptive
    targets = scale_to_participant_epsilon(
        shape,
        priv.allocation.total_participant_epsilon,
        delta=priv.delta,
        sample_rate=sample_rate,
        steps=steps,
    )
    allocator = PerModalityBudgetAllocator(
        targets, risks, delta=priv.delta, sample_rate=sample_rate, steps=steps
    )
    group_sigmas = resolve_group_sigmas(allocator.noise_multipliers())

    model = MultimodalDepressionModel(input_dims, base.model).to(device)
    dp_model = wrap_for_per_sample_grads(model)
    groups = map_parameter_groups(dp_model)
    optimizer = torch.optim.SGD(dp_model.parameters(), lr=base.train.learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed)
    train_subset = Subset(full, train_idx)
    dp_train_steps(
        dp_model,
        train_subset,
        poisson_batches(n_train, sample_rate, steps, generator),
        objective,
        groups=groups,
        group_sigmas=group_sigmas,
        max_grad_norm=priv.max_grad_norm,
        expected_batch_size=expected_batch_size,
        optimizer=optimizer,
        device=device,
        generator=generator,
    )
    return evaluate_model(model, _loader(full, test_idx, batch_size), objective, device)


def _eval_federated(
    full: Any,
    train_idx: list[int],
    test_idx: list[int],
    *,
    base: Any,
    fed: Any,
    input_dims: dict[str, int],
    labels_all: list[int],
    device: torch.device,
    rounds: int,
    seed: int,
    strategy: str,
    aggregation: AggregationConfig,
) -> dict[str, float]:
    """Train a federated variant on a fold and score the held-out test split.

    Mirrors ``scripts/run_federated_comparison.py``: clients partition the fold's
    *training* data only, per-round model selection happens on a selection split
    carved out of it, and the fold's scored split is touched once at the end.
    Passing the scored split in as the round validation loader would select on
    the number being reported (ADR-0015).
    """
    seed_everything(seed)
    fold_train, selection = _carve_fold(
        full, train_idx, labels_all, selection_fraction=base.train.selection_fraction, seed=seed
    )
    # A fold cannot support more clients than it has training sessions. This only
    # binds on the 32-session mock corpus (9 per fold after the selection split);
    # on DAIC-WOZ each fold carries ~100, so the configured count stands.
    num_clients = min(fed.federation.num_clients, len(fold_train))  # type: ignore[arg-type]
    federation = fed.federation.model_copy(
        update={
            "num_rounds": rounds,
            "num_clients": num_clients,
            "clients_per_round": num_clients,
        }
    )
    partitions = build_client_partitions(len(fold_train), federation, seed)  # type: ignore[arg-type]
    clients = build_federated_clients(
        fold_train,
        partitions,
        input_dims=input_dims,
        model_config=base.model,
        batch_size=base.train.batch_size,
        local_epochs=federation.local_epochs,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        phq8_max=base.data.phq8_max,
        phq_loss_weight=base.model.phq_loss_weight,
        seed=seed,
    )
    global_model = MultimodalDepressionModel(input_dims, base.model)
    batch_size = base.train.batch_size
    selection_loader: DataLoader[Sample] = DataLoader(
        selection, batch_size=batch_size, collate_fn=collate_fn
    )
    common = {
        "num_rounds": rounds,
        "clients_per_round": num_clients,
        "phq8_max": base.data.phq8_max,
        "phq_loss_weight": base.model.phq_loss_weight,
        "seed": seed,
        "device": str(device),
        "early_stopping_patience": federation.early_stopping_patience,
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        if strategy == "fedavg":
            run_simulation(global_model, clients, selection_loader, run_dir=run_dir, **common)
        else:
            run_capability_aware_simulation(
                global_model,
                clients,
                selection_loader,
                aggregation=aggregation,
                run_dir=run_dir,
                **common,
            )
    objective = DepressionObjective(base.data.phq8_max, base.model.phq_loss_weight).to(device)
    return evaluate_with_selected_threshold(
        global_model.to(device),
        selection_loader,
        _loader(full, test_idx, batch_size),
        objective,
        device,
    )


def _cross_validate(
    runner: FoldRunner,
    dev_idx: list[int],
    splits: list[tuple[list[int], list[int]]],
    held_out_idx: list[int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run a method over CV folds + one held-out fit, returning aggregate stats."""
    per_fold = [
        runner([dev_idx[i] for i in tr], [dev_idx[i] for i in te], seed + fold)
        for fold, (tr, te) in enumerate(splits)
    ]
    held_out = runner(dev_idx, held_out_idx, seed)
    return {
        "cv": aggregate_metrics(per_fold),
        "held_out": {k: held_out[k] for k in ("f1", "roc_auc", "accuracy")},
    }


def _fmt(agg: dict[str, float], key: str) -> str:
    """Format ``mean±std`` for a metric from an aggregate dict."""
    mean, std = agg.get(f"{key}_mean", float("nan")), agg.get(f"{key}_std", float("nan"))
    return f"{mean:.3f}±{std:.3f}"


def main() -> None:
    """Run the full Chapter-4 comparison and write its tables/plots."""
    parser = argparse.ArgumentParser(description="Final comparative evaluation (Phase 7).")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--federated-config", type=Path, default=Path("configs/federated.yaml"))
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.yaml"))
    parser.add_argument("--k-folds", type=int, default=None, help="Override k_folds.")
    parser.add_argument("--rounds", type=int, default=None, help="Override federated rounds.")
    parser.add_argument(
        "--daic-config",
        type=Path,
        default=None,
        help="Real DAIC-WOZ config; omit to run on mock data.",
    )
    parser.add_argument("--partition", choices=("iid", "dirichlet"), default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    args = parser.parse_args()

    base = load_baseline_config(args.config)
    fed = load_federated_config(args.federated_config)
    priv = load_privacy_config(args.privacy_config).privacy
    ev = load_evaluation_config(args.evaluation_config).evaluation
    seed_everything(base.seed)

    k_folds = args.k_folds if args.k_folds is not None else ev.k_folds
    # federated.yaml owns the round count unless explicitly overridden here.
    rounds = args.rounds or ev.rounds or fed.federation.num_rounds
    # Same override semantics as scripts/run_federated_comparison.py, so the
    # Phase 4 and Phase 7 federated arms can be compared run for run.
    federation = fed.federation
    if args.partition is not None:
        federation = federation.model_copy(
            update={"partition": federation.partition.model_copy(update={"mode": args.partition})}
        )
    if args.num_clients is not None:
        federation = federation.model_copy(
            update={
                "num_clients": args.num_clients,
                "clients_per_round": min(federation.clients_per_round, args.num_clients),
            }
        )
    fed = fed.model_copy(update={"federation": federation})
    device = torch.device(resolve_device(base.train.device))

    full, labels_all, input_dims, official_idx = _build_corpus(base, args.daic_config)
    n = len(labels_all)
    # Stratify on the binary depression label: on an imbalanced corpus an
    # unstratified 10-fold split yields single-class folds whose ROC-AUC is
    # undefined, quietly shrinking the fold count behind every reported mean.
    dev_idx, held_out_idx = stratified_held_out_split(labels_all, ev.held_out_fraction, base.seed)
    dev_labels = [labels_all[i] for i in dev_idx]
    splits = stratified_k_fold_indices(dev_labels, min(k_folds, len(dev_idx)), base.seed)
    positive_rate = sum(labels_all) / n
    dataset_name = "daic_woz" if args.daic_config is not None else "mock"
    print(
        f"dataset={dataset_name}  device={device}  dims={input_dims}\n"
        f"n={n}  dev={len(dev_idx)}  held_out={len(held_out_idx)}  folds={len(splits)}  "
        f"positive_rate={positive_rate:.3f} (stratified)"
    )

    def cap_config(*, reputation: bool, distillation: bool) -> AggregationConfig:
        return fed.aggregation.model_copy(
            update={
                "strategy": "capability_aware",
                "reputation_weighting": reputation,
                "federated_distillation": distillation,
            }
        )

    # Method runners (each a fold closure: train_idx, test_idx, seed -> metrics).
    def federated(strategy: str, aggregation: AggregationConfig) -> FoldRunner:
        """Bind one federated variant into a fold runner."""
        return lambda tr, te, s: _eval_federated(
            full,
            tr,
            te,
            base=base,
            fed=fed,
            input_dims=input_dims,
            labels_all=labels_all,
            device=device,
            rounds=rounds,
            seed=s,
            strategy=strategy,
            aggregation=aggregation,
        )

    methods: dict[str, FoldRunner] = {
        "centralized": lambda tr, te, s: _eval_centralized(
            full,
            tr,
            te,
            base=base,
            input_dims=input_dims,
            labels_all=labels_all,
            device=device,
            epochs=ev.epochs,
            seed=s,
        ),
        "fedavg": federated("fedavg", fed.aggregation),
        "personalized": federated(
            "capability_aware", cap_config(reputation=True, distillation=False)
        ),
        "proposed": federated("capability_aware", cap_config(reputation=True, distillation=True)),
        "proposed_no_reputation": federated(
            "capability_aware", cap_config(reputation=False, distillation=True)
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, runner in methods.items():
        print(f"  running {name} ...")
        results[name] = _cross_validate(runner, dev_idx, splits, held_out_idx, seed=base.seed)

    # The official AVEC2017 test split, read exactly once per method after every
    # CV decision is already fixed. This is the only number comparable to
    # published DAIC-WOZ work; across seeds because a single fit on ~47 sessions
    # is not a result (ADR-0015).
    official: dict[str, dict[str, Any]] = {}
    if official_idx:
        pool_idx = list(range(len(labels_all)))
        for name, runner in methods.items():
            print(f"  official test: {name} ...")
            official[name] = aggregate_metrics(
                [runner(pool_idx, official_idx, s) for s in base.train.seeds]
            )

    # DP privacy–utility: adaptive vs uniform allocation (centralized DP-SGD).
    dp_results: dict[str, dict[str, Any]] = {}
    for mode in ("adaptive", "uniform"):
        print(f"  running dp_{mode} ...")
        dp_results[mode] = _cross_validate(
            lambda tr, te, s, mode=mode: _eval_centralized_dp(
                full,
                tr,
                te,
                base=base,
                priv=priv,
                input_dims=input_dims,
                device=device,
                epochs=ev.dp_epochs,
                seed=s,
                mode=mode,
            ),
            dev_idx,
            splits,
            held_out_idx,
            seed=base.seed,
        )

    # Inference latency across batch sizes.
    seed_everything(base.seed)
    latency_model = MultimodalDepressionModel(input_dims, base.model)
    latency = measure_inference_latency(
        latency_model,
        full,
        batch_sizes=ev.latency_batch_sizes,
        repeats=ev.latency_repeats,
        device=device,
    )

    run_dir = create_run_dir(base.train.output_dir, "phase7", "phase7_final_evaluation")
    save_config(
        run_dir,
        {
            "baseline": base.model_dump(),
            "federated": fed.model_dump(),
            "privacy": priv.model_dump(),
            "evaluation": ev.model_dump(),
        },
    )
    (run_dir / "cv_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if official:
        (run_dir / "official_test.json").write_text(
            json.dumps(official, indent=2), encoding="utf-8"
        )
    ablation = {k: results[k] for k in ("proposed", "proposed_no_reputation", "personalized")}
    (run_dir / "ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    (run_dir / "dp_comparison.json").write_text(json.dumps(dp_results, indent=2), encoding="utf-8")
    (run_dir / "latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
    _write_summary(run_dir / "chapter4_summary.md", results, ablation, dp_results, latency)
    _plot_latency(latency, run_dir / "inference_latency.png")

    header = f"\n{'method':<24}{'CV F1':>16}{'CV ROC-AUC':>16}{'held-out F1':>14}"
    print(header + (f"{'test ROC-AUC':>18}" if official else ""))
    for name, res in results.items():
        row = (
            f"{name:<24}{_fmt(res['cv'], 'f1'):>16}{_fmt(res['cv'], 'roc_auc'):>16}"
            f"{res['held_out']['f1']:>14.3f}"
        )
        print(row + (f"{_fmt(official[name], 'roc_auc'):>18}" if official else ""))
    print(f"\nRun dir: {run_dir}")
    if args.daic_config is None:
        print(
            "(On mock data the labels are random, so accuracy is a placeholder; "
            "the tables are the deliverable.)"
        )


def _write_summary(
    path: Path,
    results: dict[str, dict[str, Any]],
    ablation: dict[str, dict[str, Any]],
    dp_results: dict[str, dict[str, Any]],
    latency: list[dict[str, Any]],
) -> None:
    """Write the combined Chapter-4 markdown tables."""
    lines = ["# Chapter 4 — final evaluation (mock data)\n"]
    lines.append(
        "> On mock data the depression label is random; these are "
        "placeholder numbers demonstrating the tables. DAIC-WOZ fills them in.\n"
    )

    lines.append("\n## Main comparison (10-fold CV, mean±std)\n")
    lines.append("| method | F1 | ROC-AUC | accuracy | held-out F1 |")
    lines.append("|---|---|---|---|---|")
    labels = {
        "centralized": "Centralized (no FL/DP) — cf. Xu et al. 2023",
        "fedavg": "Plain FedAvg",
        "personalized": "Personalized (reputation) — cf. Fan et al. 2025",
        "proposed": "Proposed (full framework)",
        "proposed_no_reputation": "Proposed − reputation",
    }
    for name, res in results.items():
        lines.append(
            f"| {labels.get(name, name)} | {_fmt(res['cv'], 'f1')} | "
            f"{_fmt(res['cv'], 'roc_auc')} | {_fmt(res['cv'], 'accuracy')} | "
            f"{res['held_out']['f1']:.3f} |"
        )

    lines.append("\n## Ablation (proposed vs. component removed)\n")
    lines.append("| variant | F1 | ROC-AUC |")
    lines.append("|---|---|---|")
    abl_labels = {
        "proposed": "Full framework",
        "proposed_no_reputation": "− reputation weighting",
        "personalized": "− federated distillation",
    }
    for name, res in ablation.items():
        lines.append(
            f"| {abl_labels.get(name, name)} | {_fmt(res['cv'], 'f1')} | "
            f"{_fmt(res['cv'], 'roc_auc')} |"
        )

    lines.append("\n## DP privacy–utility (centralized DP-SGD, same total ε)\n")
    lines.append("| allocation | F1 | ROC-AUC |")
    lines.append("|---|---|---|")
    for mode in ("adaptive", "uniform"):
        res = dp_results[mode]
        lines.append(
            f"| {mode} per-modality | {_fmt(res['cv'], 'f1')} | {_fmt(res['cv'], 'roc_auc')} |"
        )

    lines.append("\n## Inference latency (forward pass)\n")
    lines.append("| batch size | ms/batch | ms/sample |")
    lines.append("|---|---|---|")
    for row in latency:
        lines.append(
            f"| {row['batch_size']} | {row['ms_per_batch']:.3f} | {row['ms_per_sample']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_latency(latency: list[dict[str, Any]], path: Path) -> None:
    """Plot ms/sample vs batch size (no-op without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote latency.json only.")
        return

    sizes = [row["batch_size"] for row in latency]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, [row["ms_per_sample"] for row in latency], marker="o")
    ax.set_xlabel("Batch size (compute budget)")
    ax.set_ylabel("Latency (ms / sample)")
    ax.set_title("Inference latency vs. batch size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
