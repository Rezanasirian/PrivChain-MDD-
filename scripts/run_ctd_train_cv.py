"""Train-only nested CV for the CTD federated optimization repair.

Compares the old per-shard class weighting against the candidate repair
(``aggregate_counts`` + lr=0.1) without constructing or scoring official dev/test.
Every participant receives one out-of-fold score for paired bootstrap analysis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from privchain.data.ctd_reference import CTD_REFERENCE_DIM, CtdLinearClassifier
from privchain.data.mock_daic_woz import MODALITIES, Sample, collate_fn
from privchain.eval.benchmark import stratified_held_out_split, stratified_k_fold_indices
from privchain.eval.metrics import paired_bootstrap_auc_difference
from privchain.federated.client import FederatedClient
from privchain.federated.partition import partition_indices
from privchain.federated.simulation import run_simulation
from privchain.seeding import derive_seed, seed_everything
from privchain.training.objective import (
    DepressionObjective,
    collect_scores,
    evaluate_with_selected_threshold,
    positive_class_weight,
)


class _CtdArrays(Dataset[Sample]):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Sample:
        return Sample(
            audio=torch.from_numpy(self.x[index]).unsqueeze(0),
            video=torch.zeros((1, 1)),
            text=torch.zeros((1, 1)),
            presence={m: torch.tensor(int(m == "audio")) for m in MODALITIES},
            phq8_score=torch.tensor(0),
            label=torch.tensor(int(self.y[index])),
        )


@dataclass(frozen=True)
class _Arm:
    class_weight_mode: str
    learning_rate: float


ARMS = {
    "per_shard": _Arm("per_shard", 0.01),
    "aggregate_counts_lr01": _Arm("aggregate_counts", 0.1),
}


def _loader(data: Dataset[Sample], batch_size: int, *, shuffle: bool = False) -> DataLoader[Sample]:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def _clients(
    data: Dataset[Sample], labels: np.ndarray, arm: _Arm, *, seed: int,
    num_clients: int, batch_size: int, local_epochs: int,
) -> list[FederatedClient]:
    shards = partition_indices(len(data), num_clients, seed + 1)  # type: ignore[arg-type]
    total_pos = int(labels.sum())
    shared_weight = (len(labels) - total_pos) / total_pos if 0 < total_pos < len(labels) else None
    clients: list[FederatedClient] = []
    for client_id, indices in enumerate(shards):
        subset = torch.utils.data.Subset(data, indices)
        if arm.class_weight_mode == "per_shard":
            weight = positive_class_weight(_loader(subset, batch_size))
        else:
            weight = shared_weight
        clients.append(
            FederatedClient(
                client_id, (1, 0, 0), CtdLinearClassifier(),
                DataLoader(
                    subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                    generator=torch.Generator().manual_seed(seed + client_id),
                ),
                local_epochs=local_epochs, learning_rate=arm.learning_rate,
                weight_decay=0.0, phq8_max=24, phq_loss_weight=0.0,
                pos_weight=weight, optimizer_name="sgd",
            )
        )
    return clients


def _run_fold(
    x: np.ndarray, y: np.ndarray, outer_train: list[int], outer_val: list[int],
    *, arm: _Arm, rep_seed: int, fold: int, args: argparse.Namespace, run_dir: Path,
) -> tuple[dict[str, float], np.ndarray]:
    # ``rep_seed + fold`` would collide whenever two repetition seeds sit closer
    # together than the fold count, silently shrinking the spread the
    # confirmatory run exists to measure (see commit adc6689).
    seed = derive_seed(rep_seed, fold)
    inner_fit_rel, inner_selection_rel = stratified_held_out_split(
        y[outer_train].tolist(), args.selection_fraction, seed
    )
    fit_idx = [outer_train[i] for i in inner_fit_rel]
    selection_idx = [outer_train[i] for i in inner_selection_rel]
    mean = x[fit_idx].mean(axis=0)
    std = np.maximum(x[fit_idx].std(axis=0), 1e-6)
    normalized = (x - mean) / std
    fit = _CtdArrays(normalized[fit_idx], y[fit_idx])
    selection = _CtdArrays(normalized[selection_idx], y[selection_idx])
    report = _CtdArrays(normalized[outer_val], y[outer_val])

    seed_everything(seed)
    model = CtdLinearClassifier()
    fold_dir = run_dir / f"seed_{rep_seed}" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    run_simulation(
        model,
        _clients(
            fit, y[fit_idx], arm, seed=seed, num_clients=args.num_clients,
            batch_size=args.batch_size, local_epochs=args.local_epochs,
        ),
        _loader(selection, args.batch_size),
        num_rounds=args.rounds, clients_per_round=args.num_clients,
        phq8_max=24, phq_loss_weight=0.0, run_dir=fold_dir, seed=seed,
        early_stopping_patience=args.patience, selection_metric="roc_auc",
    )
    model.load_state_dict(
        torch.load(fold_dir / "best_global_model.pt", map_location="cpu", weights_only=True)
    )
    objective = DepressionObjective(24, 0.0)
    metrics = evaluate_with_selected_threshold(
        model, _loader(selection, args.batch_size), _loader(report, args.batch_size),
        objective, torch.device("cpu"),
    )
    scores, _ = collect_scores(model, _loader(report, args.batch_size), torch.device("cpu"))
    return metrics, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ctd_train_cv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--selection-fraction", type=float, default=0.2)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    with np.load(args.artifact_dir / "ctd_train.npz") as payload:
        session_ids = payload["session_id"].astype(np.int64)
        x = payload["emb"].astype(np.float32)
        y = payload["label"].astype(np.int64)
    if x.shape != (len(y), CTD_REFERENCE_DIM):
        raise ValueError(f"invalid CTD train shape {x.shape}")
    if len(session_ids) != len(y):
        raise ValueError("CTD train session IDs and labels have different lengths")
    folds = stratified_k_fold_indices(y.tolist(), args.folds, 42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof: dict[str, list[np.ndarray]] = {name: [] for name in ARMS}
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as results_sink, (
        args.output_dir / "oof_predictions.jsonl"
    ).open("w", encoding="utf-8") as oof_sink:
        for arm_name, arm in ARMS.items():
            for seed in args.seeds:
                scores = np.zeros(len(y), dtype=np.float64)
                for fold, (train_idx, val_idx) in enumerate(folds):
                    metrics, fold_scores = _run_fold(
                        x, y, train_idx, val_idx, arm=arm, rep_seed=seed,
                        fold=fold, args=args, run_dir=args.output_dir / arm_name,
                    )
                    scores[val_idx] = fold_scores
                    results_sink.write(
                        json.dumps({"arm": arm_name, "seed": seed, "fold": fold, **metrics})
                        + "\n"
                    )
                    results_sink.flush()
                oof[arm_name].append(scores)
                oof_sink.write(json.dumps({
                    "arm": arm_name, "seed": seed,
                    # Stable positional keys align arms for a participant-level
                    # paired test without exporting DAIC participant IDs.
                    "participant_index": list(range(len(y))), "labels": y.tolist(),
                    "scores": scores.tolist(),
                }) + "\n")
                oof_sink.flush()

    mean_scores = {name: np.stack(values).mean(axis=0) for name, values in oof.items()}
    paired = paired_bootstrap_auc_difference(
        y, mean_scores["aggregate_counts_lr01"], mean_scores["per_shard"],
        n_resamples=args.bootstrap_resamples, seed=42,
    )
    summary = {
        "protocol": "nested-inner-cv-official-train-only",
        "official_dev_read": False,
        "official_test_read": False,
        "folds": args.folds,
        "seeds": args.seeds,
        "arms": {name: arm.__dict__ for name, arm in ARMS.items()},
        "paired_auc_aggregate_counts_lr01_minus_per_shard": paired,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
