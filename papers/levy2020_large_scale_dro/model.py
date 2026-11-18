r"""Training of the linear head of Levy et al. (2020), Appendix F.

Every run trains `torch.nn.Linear(d, 10)` with stochastic gradient steps on
the plug-in robust loss of a batch (Section 3). The robust losses come from
Optora only. The rest follows the authors' `train.py` (fast-dro dc75246).

- SGD with Nesterov momentum 0.9, a constant step, and weight decay on the
  weight only, which adds $\tfrac{\mu}{2} \lVert x \rVert_2^2$ to the
  objective.
- The training set is reshuffled every epoch and split into consecutive
  batches, the last one smaller when $n$ does not divide the set, as a
  `DataLoader` with `shuffle=True` does.
- The averaged iterate of Shamir and Zhang (2013) with parameter
  $\eta = 3$, $\bar x_t = (1 - w_t) \bar x_{t-1} + w_t x_t$ with
  $w_t = (\eta + 1)/(t + \eta)$ and $t$ starting at 2, as the authors'
  `utils.average_step`. The full-batch run is not averaged.

With `cuda_graphs`, the forward and backward pass of each batch size is
captured once and replayed. The gradients are persistent tensors that the
captured pass zeroes and refills, so the optimizer update, which runs
eagerly, reads the same tensors after any replay.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import optora
import torch
from torch.nn import functional

NUM_CLASSES = 10
NUM_GROUPS = 20
OBJECTIVES = ("cvar", "chi_square", "penalty")
ERM = "erm"
FULL_BATCH = 0
ALPHA = 0.02
RADIUS = 2.0
PENALTY = 0.025
WEIGHT_DECAY = 1e-2
MOMENTUM = 0.9
AVERAGING = 3.0
WARMUP_STEPS = 3

STEP_SIZES: dict[tuple[str, int], float] = {
    ("cvar", 50): 1e-4,
    ("cvar", 500): 1e-3,
    ("cvar", 5000): 5e-3,
    ("cvar", FULL_BATCH): 1e-2,
    ("chi_square", 50): 2e-4,
    ("chi_square", 500): 2e-3,
    ("chi_square", 5000): 2e-2,
    ("chi_square", FULL_BATCH): 2e-2,
    ("penalty", 50): 1e-4,
    ("penalty", 500): 1e-3,
    ("penalty", 5000): 1e-2,
    ("penalty", FULL_BATCH): 1e-2,
}
"""Step sizes of `hyperparameters/mnist_final_*.yaml` and `mnist_full_batch.yaml`."""

ERM_STEP_SIZES: dict[float, float] = {1e-4: 5e-2, 1e-3: 2e-2}
"""Step size of each ERM weight decay in `hyperparameters/mnist_erm.yaml`."""

RobustLoss = Callable[[torch.Tensor], torch.Tensor]


def robust_loss(objective: str, size: int, device: torch.device) -> RobustLoss:
    """Return the plug-in robust loss of `objective` on `size` samples.

    Args:
        objective: One of `OBJECTIVES`, or `ERM` for the mean.
        size: Number of samples, the support of the uniform nominal.
        device: Device of the nominal distribution.

    Returns:
        A map from per-sample losses of shape `(..., size)` to the robust
        loss of shape `(...)`.

    Raises:
        ValueError: If `objective` is unknown.
    """
    if objective == ERM:
        return lambda loss: loss.mean(dim=-1)
    nominal = torch.full((size,), 1.0 / size, device=device)
    if objective == "cvar":
        loss = optora.CVaRAmbiguitySet(nominal, ALPHA).worst_case_expectation
    elif objective == "chi_square":
        loss = optora.ChiSquareAmbiguitySet(nominal, RADIUS).worst_case_expectation
    elif objective == "penalty":
        loss = optora.PenalizedChiSquareDRO(nominal, PENALTY)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    return cast(RobustLoss, loss)


@dataclass(frozen=True)
class Setting:
    r"""Hyperparameters of one training run.

    Attributes:
        objective: One of `OBJECTIVES` or `ERM`.
        batch_size: Minibatch size, or `FULL_BATCH` for the whole set.
        step_size: Constant SGD step.
        weight_decay: Weight decay $\mu$ on the weight.
        epochs: Passes over the training set.
        eval_interval: Epochs between evaluations of the training objective.
        seed: Seed of the initialization and of the shuffling.
    """

    objective: str
    batch_size: int
    step_size: float
    weight_decay: float
    epochs: int
    eval_interval: int
    seed: int

    @property
    def averaged(self) -> bool:
        """Whether the run reports the averaged iterate (every minibatch run)."""
        return self.batch_size != FULL_BATCH


@dataclass(frozen=True)
class Run:
    """Outcome of one training run, on CPU.

    Attributes:
        setting: Hyperparameters of the run.
        epochs: Epochs after which the training objective was evaluated.
        objective: Training objective at those epochs, robust loss of the
            whole training set plus the regularizer.
        group_loss: Mean test log loss of each of the 20 subgroups after the
            last epoch, shape `(20,)`, `nan` for a subgroup without test
            samples.
        test_loss: Mean test log loss after the last epoch.
        test_accuracy: Test accuracy after the last epoch.
        seconds_per_epoch: Wall-clock training time per epoch, evaluation
            excluded.
    """

    setting: Setting
    epochs: torch.Tensor
    objective: torch.Tensor
    group_loss: torch.Tensor
    test_loss: float
    test_accuracy: float
    seconds_per_epoch: float


class _Step:
    """Forward and backward pass on one batch size, eager or graph-replayed."""

    def __init__(
        self,
        model: torch.nn.Linear,
        features: torch.Tensor,
        targets: torch.Tensor,
        loss: RobustLoss,
        size: int,
        graphed: bool,
    ) -> None:
        self.model = model
        self.features = features
        self.targets = targets
        self.loss = loss
        self.index = torch.arange(size, device=features.device)
        self.whole = size == len(features)
        self.graph: torch.cuda.CUDAGraph | None = None
        if graphed:
            self.graph = self._capture()

    def _forward_backward(self) -> None:
        for parameter in self.model.parameters():
            assert parameter.grad is not None
            parameter.grad.zero_()
        # A batch of the whole set needs no gather, since every robust loss
        # here is invariant to the order of the samples.
        if self.whole:
            batch_features, batch_targets = self.features, self.targets
        else:
            batch_features = self.features[self.index]
            batch_targets = self.targets[self.index]
        per_sample = functional.cross_entropy(
            self.model(batch_features), batch_targets, reduction="none"
        )
        torch.autograd.backward(self.loss(per_sample))

    def _capture(self) -> torch.cuda.CUDAGraph:
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(WARMUP_STEPS):
                self._forward_backward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._forward_backward()
        return graph

    def __call__(self, index: torch.Tensor) -> None:
        if not self.whole:
            self.index.copy_(index)
        if self.graph is None:
            self._forward_backward()
        else:
            self.graph.replay()


def averaging_weight(iteration: int) -> float:
    r"""Return the weight $w_t = (\eta + 1)/(t + \eta)$ of the new iterate.

    The authors' `train.py` calls `utils.average_step` with $t = 2$ after the
    first step and increments it by one per step.

    Args:
        iteration: The index $t \ge 2$.

    Returns:
        The weight with $\eta = 3$.
    """
    return (AVERAGING + 1.0) / (iteration + AVERAGING)


def _objective_value(
    weight: torch.Tensor,
    bias: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    loss: RobustLoss,
    weight_decay: float,
) -> torch.Tensor:
    """Return the training objective at `(weight, bias)` without a host sync."""
    per_sample = functional.cross_entropy(
        functional.linear(features, weight, bias), targets, reduction="none"
    )
    return loss(per_sample) + 0.5 * weight_decay * torch.sum(weight**2)


def _test_metrics(
    weight: torch.Tensor,
    bias: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Return the per-group mean test log loss, the mean loss and the accuracy."""
    logits = functional.linear(features, weight, bias)
    per_sample = functional.cross_entropy(logits, targets, reduction="none")
    totals = torch.zeros(NUM_GROUPS, device=features.device).index_add_(
        0, groups, per_sample
    )
    counts = torch.bincount(groups, minlength=NUM_GROUPS).to(totals.dtype)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return (totals / counts).cpu(), float(per_sample.mean()), float(accuracy)


def train(
    setting: Setting,
    train_split: tuple[torch.Tensor, torch.Tensor],
    test_split: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cuda_graphs: bool,
) -> Run:
    """Train one linear head and record its training objective and test loss.

    Args:
        setting: Hyperparameters of the run.
        train_split: Training features `(N, d)` and targets `(N,)` on the
            run's device.
        test_split: Test features, targets and subgroups on the same device.
        cuda_graphs: Whether to replay the forward and backward passes from
            CUDA graphs.

    Returns:
        The evaluated training objective, the test metrics of the reported
        iterate, and the training time per epoch.
    """
    features, targets = train_split
    device = features.device
    size, dimension = features.shape
    torch.manual_seed(setting.seed)
    model = torch.nn.Linear(dimension, NUM_CLASSES).to(device)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight], "weight_decay": setting.weight_decay},
            {"params": [model.bias], "weight_decay": 0.0},
        ],
        lr=setting.step_size,
        momentum=MOMENTUM,
        nesterov=True,
    )
    averaged = [parameter.detach().clone() for parameter in model.parameters()]
    batch = size if setting.batch_size == FULL_BATCH else min(setting.batch_size, size)
    steps: dict[int, _Step] = {}
    for length in {batch, size % batch} - {0}:
        steps[length] = _Step(
            model,
            features,
            targets,
            robust_loss(setting.objective, length, device),
            length,
            cuda_graphs,
        )
    full_loss = robust_loss(setting.objective, size, device)
    generator = torch.Generator(device=device).manual_seed(setting.seed)
    evaluations = setting.epochs // setting.eval_interval
    history = torch.empty(evaluations, device=device)
    elapsed = 0.0
    iteration = 1
    for epoch in range(1, setting.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        order = torch.randperm(size, device=device, generator=generator)
        for index in torch.split(order, batch):
            steps[len(index)](index)
            optimizer.step()
            iteration += 1
            if setting.averaged:
                share = averaging_weight(iteration)
                with torch.no_grad():
                    for mean, parameter in zip(
                        averaged, model.parameters(), strict=True
                    ):
                        mean.lerp_(parameter, share)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        if epoch % setting.eval_interval == 0:
            weight, bias = averaged if setting.averaged else model.parameters()
            with torch.no_grad():
                history[epoch // setting.eval_interval - 1] = _objective_value(
                    weight, bias, features, targets, full_loss, setting.weight_decay
                )
    weight, bias = averaged if setting.averaged else model.parameters()
    with torch.no_grad():
        group_loss, test_loss, test_accuracy = _test_metrics(
            weight.detach(), bias.detach(), *test_split
        )
    return Run(
        setting=setting,
        epochs=torch.arange(1, evaluations + 1) * setting.eval_interval,
        objective=history.cpu(),
        group_loss=group_loss,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        seconds_per_epoch=elapsed / setting.epochs,
    )


def epochs_to_reach(run: Run, target: float) -> int:
    """Return the first evaluated epoch whose objective is at most `target`.

    Args:
        run: A training run.
        target: Objective value to reach.

    Returns:
        The epoch, or the budget plus one if the run never reaches `target`.
    """
    reached = torch.nonzero(run.objective <= target)
    if reached.numel() == 0:
        return run.setting.epochs + 1
    return int(run.epochs[reached[0, 0]])
