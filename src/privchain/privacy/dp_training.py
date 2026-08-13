"""One DP-SGD training arm, shared by every experiment that needs one (Phase 3).

The per-modality DP-SGD loop had been written out separately in each script that
wanted it — the ε sweep, the attacker evaluation, the final-evaluation harness —
which is exactly the drift ADR-0015 was written to stop: two harnesses given the
same hyperparameters produced different numbers because one passed a seeded
generator and the other did not.

It lives here once, carrying the details ADR-0013 paid for:

* **Adam, not plain SGD.** The "SGD" in DP-SGD names the noised, per-sample
  clipped *gradient*, not the optimizer consuming it. Plain SGD at the baseline's
  learning rate does not move this model at all, which showed up as F1 = 0 even
  at ε = ∞.
* **Poisson subsampling**, the sampling assumption the RDP accountant makes.
* **Epoch selection on the selection split**, matching the non-private baseline.
  Comparing the baseline's best epoch against DP's last epoch charges DP for a
  difference in protocol rather than in privacy. Evaluating a released model is
  post-processing, so it costs no budget.
* **Selection over trained epochs only.** Including a pre-training evaluation
  let untrained weights win, which silently reported identical metrics at every ε.
* **Early stopping on the baseline's patience.** Stopping early spends *fewer*
  mechanism applications than planned, so the reported ε stays an upper bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from privchain.config import BaselineConfig, ModelConfig, PrivacySettings
from privchain.data.mock_daic_woz import Sample
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.dp_sgd import (
    dp_train_steps,
    map_parameter_groups,
    poisson_batches,
    steps_for_epochs,
    wrap_for_per_sample_grads,
)
from privchain.seeding import seed_everything
from privchain.training.objective import (
    DepressionObjective,
    evaluate_model,
    evaluate_with_selected_threshold,
    positive_class_weight,
)
from privchain.training.protocol import Splits, make_loader


@dataclass(frozen=True)
class DpArmConfig:
    """Everything held fixed across the arms of a DP experiment.

    Building this once and varying only ``group_sigmas`` is what makes two arms
    comparable: same splits, same objective, same step budget, same selection
    rule. Anything that differs between arms belongs in the call, not here.
    """

    input_dims: dict[str, int]
    model: ModelConfig
    train_subset: Dataset[Sample]
    selection_loader: DataLoader[Sample]
    report_loader: DataLoader[Sample]
    objective: DepressionObjective
    sample_rate: float
    expected_batch_size: float
    planned_steps: int
    epochs: int
    max_grad_norm: float
    learning_rate: float
    weight_decay: float
    selection_metric: str
    early_stopping_patience: int | None
    device: torch.device
    n_train: int
    # Already folded into `objective`; carried separately so runs can report the
    # class weighting they actually used.
    pos_weight: float | None


def build_dp_arm_config(
    base: BaselineConfig,
    priv: PrivacySettings,
    splits: Splits,
    input_dims: dict[str, int],
    device: torch.device,
) -> DpArmConfig:
    """Assemble the fixed half of a DP experiment from validated config.

    The splits come from :func:`privchain.training.protocol.build_splits`, so the
    DP arms see exactly the data the non-private baseline sees (ADR-0015).

    Args:
        base: Validated baseline config (model + training).
        priv: Validated ``privacy`` section.
        splits: The three protocol splits.
        input_dims: Per-modality feature widths.
        device: Torch device.

    Returns:
        A :class:`DpArmConfig` ready to be run at any noise level.
    """
    batch_size = base.train.batch_size
    n_train = len(splits.train)  # type: ignore[arg-type]
    # Poisson subsampling: each sample enters a step with probability q, exactly
    # the assumption the RDP accountant makes (ADR-0004).
    sample_rate = min(1.0, batch_size / n_train)

    # Match the non-private baseline's objective, including class weighting, so
    # the privacy-utility comparison isolates the cost of DP. The weighting is
    # largely neutralized by per-sample clipping (ADR-0013), but it is kept
    # identical across arms rather than silently differing.
    pos_weight = None
    if base.train.class_weighting:
        pos_weight = positive_class_weight(
            make_loader(splits.train, batch_size=batch_size, shuffle=False)
        )

    return DpArmConfig(
        input_dims=input_dims,
        model=base.model,
        train_subset=splits.train,
        selection_loader=make_loader(splits.selection, batch_size=batch_size, shuffle=False),
        report_loader=make_loader(splits.report, batch_size=batch_size, shuffle=False),
        objective=DepressionObjective(
            base.data.phq8_max, base.model.phq_loss_weight, pos_weight
        ).to(device),
        sample_rate=sample_rate,
        expected_batch_size=sample_rate * n_train,
        planned_steps=steps_for_epochs(n_train, batch_size, priv.sweep.epochs),
        epochs=priv.sweep.epochs,
        max_grad_norm=priv.max_grad_norm,
        learning_rate=base.train.learning_rate,
        weight_decay=base.train.weight_decay,
        selection_metric=base.train.selection_metric,
        early_stopping_patience=base.train.early_stopping_patience,
        device=device,
        n_train=n_train,
        pos_weight=pos_weight,
    )


def train_dp_arm(
    config: DpArmConfig, group_sigmas: dict[str, float], seed: int
) -> dict[str, float]:
    """Train one DP-SGD arm and report on the untouched split.

    Args:
        config: The fixed half of the experiment.
        group_sigmas: Noise multiplier per parameter group. All-zero disables the
            perturbation while leaving clipping and Poisson sampling in place —
            the ε = ∞ reference that isolates the cost of the noise itself.
        seed: Seeds initialization, the Poisson draws, and the DP noise.

    Returns:
        Report-split metrics, with the decision threshold chosen on the selection
        split.

    Raises:
        RuntimeError: If the step budget produced no trained epoch.
    """
    seed_everything(seed)
    model = MultimodalDepressionModel(config.input_dims, config.model).to(config.device)
    dp_model = wrap_for_per_sample_grads(model)
    groups = map_parameter_groups(dp_model)
    optimizer = torch.optim.Adam(
        dp_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device=config.device).manual_seed(seed)
    batches = poisson_batches(
        len(config.train_subset),  # type: ignore[arg-type]
        config.sample_rate,
        config.planned_steps,
        generator,
    )

    steps_per_epoch = max(1, len(batches) // config.epochs)
    best_selector = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for start in range(0, len(batches), steps_per_epoch):
        dp_train_steps(
            dp_model,
            config.train_subset,
            batches[start : start + steps_per_epoch],
            config.objective,
            groups=groups,
            group_sigmas=group_sigmas,
            max_grad_norm=config.max_grad_norm,
            expected_batch_size=config.expected_batch_size,
            optimizer=optimizer,
            device=config.device,
            generator=generator,
        )
        # threshold=None here only decides which epoch wins; the reported
        # threshold is re-derived on the same split below.
        epoch_metrics = evaluate_model(
            model, config.selection_loader, config.objective, config.device, threshold=None
        )
        selector = epoch_metrics[config.selection_metric]
        if selector > best_selector:
            best_selector = selector
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            patience = config.early_stopping_patience
            if patience is not None and stale >= patience:
                break

    if best_state is None:  # only reachable if the step budget is empty
        raise RuntimeError("no training steps ran for this DP arm")
    model.load_state_dict(best_state)
    return evaluate_with_selected_threshold(
        model, config.selection_loader, config.report_loader, config.objective, config.device
    )
