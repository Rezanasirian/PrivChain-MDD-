"""Framework-agnostic federated client (Phase 2, objective H2).

Holds a local model copy and dataset shard; ``fit`` loads the server's global
parameters, runs a few local epochs, and returns the updated parameters with the
local sample count (for FedAvg weighting). This same client backs both the
in-house simulator (:mod:`privchain.federated.simulation`) and the Flower adapter
(:mod:`privchain.federated.flower_app`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from opacus import GradSampleModule
from torch.utils.data import DataLoader

from privchain.data.mock_daic_woz import Batch, Sample
from privchain.federated.distillation import capability_masked_anchor, distillation_loss
from privchain.fusion.baseline_model import MultimodalDepressionModel
from privchain.privacy.accountant import (
    Mechanism,
    compose_epsilon,
    get_epsilon,
    get_noise_multiplier,
)
from privchain.privacy.dp_sgd import (
    SHARED_GROUP,
    PerSampleBackend,
    dp_train_steps,
    map_parameter_groups,
    poisson_batches,
    steps_for_epochs,
    wrap_for_per_sample_grads,
)
from privchain.training.objective import DepressionObjective, evaluate_model, move_batch_to_device


@dataclass(frozen=True)
class ClientDPConfig:
    """Privacy configuration fixed before one client's training begins."""

    target_epsilons: dict[str, float]
    delta: float
    max_grad_norm: float
    batch_size: int
    num_rounds: int
    seed: int
    backend: PerSampleBackend = "grad_sample"


@dataclass(frozen=True)
class PrivacySpend:
    """Incremental and cumulative epsilon produced by an executed fit."""

    incremental: dict[str, float]
    cumulative: dict[str, float]


@dataclass(frozen=True)
class LocalWork:
    """Compute one client performed during one ``fit``.

    A shard smaller than ``batch_size`` yields a single batch, so
    ``local_epochs=1`` is one optimizer step per round however large the round
    budget looks. Recording the work makes that visible instead of inferable, and
    lets a sweep hold total steps constant while varying how they are grouped.

    Attributes:
        optimizer_steps: ``optimizer.step()`` calls, including anchor KD steps.
        num_batches: Local batches drawn (KD anchor steps excluded).
        examples_seen: Training examples consumed, counting repeats across epochs.
    """

    optimizer_steps: int
    num_batches: int
    examples_seen: int


class FederatedClient:
    """A single simulated federated client.

    Args:
        client_id: Identifier for logging.
        capability: ``[audio, video, text]`` availability flags (metadata).
        model: The client's local model (architecture matches the server).
        train_loader: DataLoader over the client's shard.
        local_epochs: Local SGD epochs per round.
        learning_rate: Local optimizer learning rate.
        weight_decay: Local optimizer weight decay.
        phq8_max: Max PHQ-8 score (for the objective).
        phq_loss_weight: Weight on the PHQ-8 regression term.
        device: Torch device string.
        pos_weight: Positive-class weight for the BCE term, measured from this
            client's own shard. Left unweighted when ``None``. The centralized
            baseline weights its loss whenever ``train.class_weighting`` is set;
            clients that did not do the same were trained under a different loss
            than the arm they are compared against (ADR-0026).
    """

    def __init__(
        self,
        client_id: int,
        capability: tuple[int, int, int],
        model: MultimodalDepressionModel,
        train_loader: DataLoader[Sample],
        *,
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        phq8_max: int,
        phq_loss_weight: float,
        device: str = "cpu",
        dp: ClientDPConfig | None = None,
        pos_weight: float | None = None,
    ) -> None:
        self.client_id = client_id
        self.capability = capability
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.objective = DepressionObjective(phq8_max, phq_loss_weight, pos_weight).to(self.device)
        self.pos_weight = pos_weight
        self.last_work = LocalWork(0, 0, 0)
        self.dp = dp
        self._dp_steps = 0
        self._steps_per_round = 0
        self._sample_rate = 0.0
        self._group_sigmas: dict[str, float] = {}
        if dp is not None:
            self._initialize_dp(dp)

    def _initialize_dp(self, dp: ClientDPConfig) -> None:
        """Calibrate this client's mechanisms against its maximum step budget."""
        if dp.num_rounds <= 0 or dp.batch_size <= 0:
            raise ValueError("DP num_rounds and batch_size must be positive")
        self._sample_rate = min(1.0, dp.batch_size / self.num_samples)
        self._steps_per_round = steps_for_epochs(self.num_samples, dp.batch_size, self.local_epochs)
        max_steps = self._steps_per_round * dp.num_rounds
        present = [
            modality
            for modality, flag in zip(("audio", "video", "text"), self.capability, strict=True)
            if flag == 1
        ]
        missing_targets = [name for name in present if name not in dp.target_epsilons]
        if missing_targets:
            raise ValueError(f"missing target epsilon for {missing_targets}")
        modality_sigmas = {
            name: get_noise_multiplier(
                dp.target_epsilons[name], self._sample_rate, max_steps, dp.delta
            )
            for name in present
        }
        self._group_sigmas = dict(modality_sigmas)
        self._group_sigmas[SHARED_GROUP] = max(modality_sigmas.values())

    def _privacy_at(self, steps: int) -> dict[str, float]:
        """Return per-group and participant-composed epsilon at ``steps``."""
        assert self.dp is not None
        mechanisms = [
            Mechanism(sigma, self._sample_rate, steps, name)
            for name, sigma in self._group_sigmas.items()
        ]
        values = {
            mechanism.name: get_epsilon(
                mechanism.noise_multiplier,
                mechanism.sample_rate,
                mechanism.steps,
                self.dp.delta,
            )
            for mechanism in mechanisms
        }
        values["composed"] = compose_epsilon(mechanisms, self.dp.delta)
        return values

    @property
    def num_samples(self) -> int:
        """Number of local training samples (FedAvg weight)."""
        return len(self.train_loader.dataset)  # type: ignore[arg-type]

    def set_parameters(self, state: OrderedDict[str, torch.Tensor]) -> None:
        """Load a (server) ``state_dict`` into the local model.

        Args:
            state: The global model parameters to adopt.
        """
        self.model.load_state_dict(state)

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return a CPU copy of the local model ``state_dict``.

        Returns:
            The local parameters.
        """
        return OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items()
        )

    def fit(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        *,
        teacher: MultimodalDepressionModel | None = None,
        distill_weight: float = 0.0,
        distill_temperature: float = 1.0,
        anchor: Batch | None = None,
        distill_steps: int = 1,
    ) -> tuple[OrderedDict[str, torch.Tensor], int, PrivacySpend | None]:
        """Adopt global params, train locally, and return updated params.

        When a ``teacher`` is supplied with a positive ``distill_weight``, each
        local step adds a federated-distillation term (Phase 4, H2) matching the
        teacher's soft predictions on the same batch — transferring cross-modal
        knowledge to this client (see
        :mod:`privchain.federated.distillation`).

        Args:
            global_state: The server's current global parameters.
            teacher: Optional frozen teacher model (the round's global model).
            distill_weight: Weight on the distillation term (0 disables it).
            distill_temperature: Softening temperature for distillation.
            anchor: Optional data-free full-modality anchor batch.
            distill_steps: Fixed post-training KD optimizer steps.

        Returns:
            ``(updated_state, num_samples, privacy_spend)``. The spend is
            ``None`` only when this client has no DP mechanism.
        """
        self.set_parameters(global_state)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.dp is not None:
            if teacher is not None and distill_weight > 0.0 and anchor is None:
                raise ValueError("private training and distillation must run as separate steps")
            before = self._privacy_at(self._dp_steps)
            generator = torch.Generator(device=self.device).manual_seed(
                self.dp.seed + 1_000_003 * self.client_id + self._dp_steps
            )
            batches = poisson_batches(
                self.num_samples, self._sample_rate, self._steps_per_round, generator
            )
            dp_model = (
                wrap_for_per_sample_grads(self.model)
                if self.dp.backend == "grad_sample"
                else self.model
            )
            groups = map_parameter_groups(dp_model, self.capability)
            dp_train_steps(
                dp_model,
                self.train_loader.dataset,
                batches,
                self.objective,
                groups=groups,
                group_sigmas=self._group_sigmas,
                max_grad_norm=self.dp.max_grad_norm,
                expected_batch_size=self._sample_rate * self.num_samples,
                optimizer=optimizer,
                device=self.device,
                generator=generator,
                backend=self.dp.backend,
            )
            if isinstance(dp_model, GradSampleModule):
                dp_model.to_standard_module()
            self._dp_steps += self._steps_per_round
            cumulative = self._privacy_at(self._dp_steps)
            incremental = {name: cumulative[name] - before[name] for name in cumulative}
            anchor_steps = self._distill_anchor(
                teacher, anchor, optimizer, distill_weight, distill_temperature, distill_steps
            )
            self.last_work = LocalWork(
                optimizer_steps=len(batches) + anchor_steps,
                num_batches=len(batches),
                examples_seen=sum(len(b) for b in batches),
            )
            return self.get_parameters(), self.num_samples, PrivacySpend(incremental, cumulative)

        distilling = teacher is not None and distill_weight > 0.0 and anchor is None
        self.model.train()
        local_batches = 0
        examples = 0
        for _ in range(self.local_epochs):
            for raw_batch in self.train_loader:
                batch = move_batch_to_device(raw_batch, self.device)
                local_batches += 1
                examples += int(batch["label"].numel())
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.objective(outputs, batch)
                if distilling:
                    assert teacher is not None  # narrows type for mypy
                    with torch.no_grad():
                        teacher_logit = teacher(batch)["logit"]
                    loss = loss + distill_weight * distillation_loss(
                        outputs["logit"], teacher_logit, distill_temperature
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
        anchor_steps = self._distill_anchor(
            teacher, anchor, optimizer, distill_weight, distill_temperature, distill_steps
        )
        self.last_work = LocalWork(
            optimizer_steps=local_batches + anchor_steps,
            num_batches=local_batches,
            examples_seen=examples,
        )
        return self.get_parameters(), self.num_samples, None

    def _distill_anchor(
        self,
        teacher: MultimodalDepressionModel | None,
        anchor: Batch | None,
        optimizer: torch.optim.Optimizer,
        weight: float,
        temperature: float,
        steps: int,
    ) -> int:
        """Run post-training KD; this method never advances a DP accountant.

        Returns:
            The number of optimizer steps taken, so the caller can account for
            them in :class:`LocalWork` (zero when KD is not configured).
        """
        if teacher is None or anchor is None or weight <= 0.0:
            return 0
        student_anchor = capability_masked_anchor(anchor, self.capability)
        teacher.eval()
        with torch.no_grad():
            teacher_logit = teacher(anchor)["logit"]
        self.model.train()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            student_logit = self.model(student_anchor)["logit"]
            loss = weight * distillation_loss(student_logit, teacher_logit, temperature)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        return steps

    def evaluate(
        self, global_state: OrderedDict[str, torch.Tensor], loader: DataLoader[Sample]
    ) -> dict[str, float]:
        """Evaluate the given global params on a loader.

        Args:
            global_state: Parameters to evaluate.
            loader: Evaluation DataLoader.

        Returns:
            Classification metrics including ``f1``, ``roc_auc``, ``loss``.
        """
        self.set_parameters(global_state)
        return evaluate_model(self.model, loader, self.objective, self.device)
