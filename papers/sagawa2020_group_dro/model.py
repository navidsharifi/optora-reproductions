"""ResNet-50 training with ERM and online group DRO, Sagawa et al. (2020).

Section 3 fine-tunes an ImageNet-pretrained ResNet-50 with SGD, momentum
0.9, batch size 128, a constant learning rate and no data augmentation
(Appendix C.2). ERM draws each epoch as a random permutation of the training
set. Group DRO follows the authors' `--reweight_groups --robust` runs. An
epoch is `n` draws with replacement, each group equally likely, and every
step plays one exponentiated-gradient ascent step of size 0.01 on the group
weights before the descent step on the weighted loss (Algorithm 1). The
group losses and the ascent come from `optora.dro.group_losses` and
`optora.OnlineGroupDRO`.

A batch of 128 does not fit in 6 GB, so a step accumulates the gradients of
micro-batches. For ERM this changes only the batch normalization statistics.
For group DRO the ascent step needs the group losses of the whole batch
before any backward pass. `_Trainer.step` therefore evaluates the batch once
without gradients, with the running statistics of batch normalization
frozen, takes the ascent step, and then backpropagates the weighted loss of
each micro-batch, whose group sums are divided by the group counts of the
whole batch. The accumulated gradient is the gradient of the weighted loss
of the whole batch.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import optora
import torch
import torch.nn.functional as F
from optora.dro import group_losses
from torch import nn

from .data import NUM_CLASSES, NUM_GROUPS, Split, Waterbirds, group_counts

ERM = "erm"
GROUP_DRO = "group_dro"
METHODS = (ERM, GROUP_DRO)
MOMENTUM = 0.9
ROBUST_STEP_SIZE = 0.01
EVAL_BATCH_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Regime:
    r"""Regularization regime of Section 3 with the values of Appendix C.2.

    Attributes:
        learning_rate: Constant SGD learning rate.
        weight_decay: Coefficient $\lambda$ of the penalty
            $\frac{\lambda}{2}\lVert\theta\rVert_2^2$, applied by SGD to
            every parameter as in the authors' code.
        epochs: Training epochs, or `None` for the budget of the
            configuration.
    """

    learning_rate: float
    weight_decay: float
    epochs: int | None


STANDARD = "standard"
STRONG_L2 = "strong_l2"
EARLY_STOPPING = "early_stopping"
REGIMES = {
    STANDARD: Regime(learning_rate=1e-3, weight_decay=1e-4, epochs=None),
    STRONG_L2: Regime(learning_rate=1e-5, weight_decay=1.0, epochs=None),
    EARLY_STOPPING: Regime(learning_rate=1e-3, weight_decay=1e-4, epochs=1),
}


@dataclass(frozen=True)
class Setting:
    """One training run.

    Attributes:
        regime: Key of `REGIMES`.
        method: `ERM` or `GROUP_DRO`.
        seed: Seed of the classifier initialization and of the sampling.
        epochs: Training epochs.
    """

    regime: str
    method: str
    seed: int
    epochs: int


@dataclass(frozen=True)
class Evaluation:
    """Group accuracies of one model on the three splits.

    Attributes:
        train: Accuracy of each group of the training set, shape `(4,)`.
        validation: Accuracy of each group of the validation set.
        test: Accuracy of each group of the test set.
    """

    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


@dataclass(frozen=True)
class Run:
    """Result of one training run.

    Attributes:
        setting: The run.
        epochs: Epochs after which the model was evaluated, the last one
            included.
        evaluations: Evaluation after each entry of `epochs`.
        train_proportions: Group proportions of the training set, shape
            `(4,)`, the weights of the average accuracy.
        seconds_per_epoch: Wall-clock training time per epoch, evaluation
            excluded.
    """

    setting: Setting
    epochs: tuple[int, ...]
    evaluations: tuple[Evaluation, ...]
    train_proportions: torch.Tensor
    seconds_per_epoch: float

    @property
    def final(self) -> Evaluation:
        """Evaluation after the last epoch."""
        return self.evaluations[-1]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form of the run."""
        return {
            "regime": self.setting.regime,
            "method": self.setting.method,
            "seed": self.setting.seed,
            "num_epochs": self.setting.epochs,
            "epochs": list(self.epochs),
            "train": [e.train.tolist() for e in self.evaluations],
            "validation": [e.validation.tolist() for e in self.evaluations],
            "test": [e.test.tolist() for e in self.evaluations],
            "train_proportions": self.train_proportions.tolist(),
            "seconds_per_epoch": self.seconds_per_epoch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        """Rebuild a run from the output of `to_dict`."""
        setting = Setting(
            data["regime"], data["method"], data["seed"], data["num_epochs"]
        )
        evaluations = tuple(
            Evaluation(
                torch.tensor(train), torch.tensor(validation), torch.tensor(test)
            )
            for train, validation, test in zip(
                data["train"], data["validation"], data["test"], strict=True
            )
        )
        return cls(
            setting=setting,
            epochs=tuple(data["epochs"]),
            evaluations=evaluations,
            train_proportions=torch.tensor(data["train_proportions"]),
            seconds_per_epoch=data["seconds_per_epoch"],
        )


def worst_group_accuracy(group_accuracy: torch.Tensor) -> float:
    """Return the smallest group accuracy."""
    return float(group_accuracy.min())


def average_accuracy(group_accuracy: torch.Tensor, proportions: torch.Tensor) -> float:
    """Return the group accuracies weighted by the training proportions.

    On the training set this is the plain accuracy. On the balanced test set
    it is the average test accuracy of Appendix C.1.
    """
    return float((group_accuracy * proportions).sum())


def build_model(pretrained: bool) -> nn.Module:
    """Return a ResNet-50 with a new two-class linear head.

    Args:
        pretrained: Whether to load torchvision's `IMAGENET1K_V1` weights,
            the weights of `pretrained=True` that the authors' code loads.
            torchvision checks their hash on download.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model: nn.Module = resnet50(weights=weights)
    head = cast(nn.Linear, model.fc)
    model.fc = nn.Linear(head.in_features, NUM_CLASSES)
    return model


class Inputs:
    """Input normalization, autocast type and memory format on one device.

    `"bfloat16"` runs the model under bfloat16 autocast in channels-last
    format, where cuDNN is fastest on the reproduction's GPU, and
    `"float32"` in the default contiguous format.

    Args:
        precision: `"bfloat16"` or `"float32"`.
        device: Device of the model.
    """

    def __init__(self, precision: str, device: torch.device) -> None:
        self.device = device
        bfloat16 = precision == "bfloat16"
        self.autocast = torch.bfloat16 if bfloat16 else None
        self.memory_format = (
            torch.channels_last if bfloat16 else torch.contiguous_format
        )
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Scale `uint8` images to `[0, 1]` and standardize them."""
        scaled = (images.float() / 255.0 - self.mean) / self.std
        return scaled.contiguous(memory_format=self.memory_format)

    def context(self) -> contextlib.AbstractContextManager[Any]:
        """Return the autocast context of a forward pass."""
        if self.autocast is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast)


@contextlib.contextmanager
def frozen_running_statistics(model: nn.Module) -> Iterator[None]:
    """Keep batch normalization in training mode without updating its statistics.

    A momentum of zero leaves the running mean and variance unchanged, while
    the forward pass still normalizes with the batch statistics. The batch
    counter `num_batches_tracked` is restored on exit, so every buffer is
    left as it was.
    """
    layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    saved = [(layer.momentum, layer.num_batches_tracked) for layer in layers]
    for layer in layers:
        layer.momentum = 0.0
        if layer.num_batches_tracked is not None:
            layer.num_batches_tracked = layer.num_batches_tracked.clone()
    try:
        yield
    finally:
        for layer, (momentum, tracked) in zip(layers, saved, strict=True):
            layer.momentum = momentum
            layer.num_batches_tracked = tracked


def epoch_order(
    method: str, groups: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Return the training indices of one epoch, drawn on the CPU.

    ERM uses a random permutation. Group DRO draws `n` indices with
    replacement, with probability inversely proportional to the size of
    each example's group, as `WeightedRandomSampler` in the authors' code.

    Args:
        method: `ERM` or `GROUP_DRO`.
        groups: Group of every training example, on the CPU.
        generator: CPU generator of the draws.
    """
    if method == ERM:
        return torch.randperm(len(groups), generator=generator)
    weights = (1.0 / group_counts(groups).double())[groups]
    return torch.multinomial(
        weights, len(groups), replacement=True, generator=generator
    )


class _Trainer:
    """Optimizer, group weights and the training step of one run."""

    def __init__(
        self,
        model: nn.Module,
        setting: Setting,
        train: Split,
        micro_batch_size: int,
        inputs: Inputs,
    ) -> None:
        regime = REGIMES[setting.regime]
        self.model = model
        self.method = setting.method
        self.train = train
        self.micro_batch_size = micro_batch_size
        self.inputs = inputs
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=regime.learning_rate,
            momentum=MOMENTUM,
            weight_decay=regime.weight_decay,
        )
        uniform = torch.full((NUM_GROUPS,), 1.0 / NUM_GROUPS, device=inputs.device)
        self.adversary = optora.OnlineGroupDRO(uniform, step_size=ROBUST_STEP_SIZE)

    def _losses(self, index: torch.Tensor) -> torch.Tensor:
        """Per-example losses of the training examples `index`."""
        images = self.inputs(self.train.images[index])
        with self.inputs.context():
            logits = self.model(images)
        return F.cross_entropy(
            logits.float(), self.train.targets[index], reduction="none"
        )

    def step(self, index: torch.Tensor) -> None:
        """Take one optimizer step on the training examples `index`."""
        micro = torch.split(index, self.micro_batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        if self.method == ERM:
            for part in micro:
                torch.autograd.backward(self._losses(part).sum() / len(index))
        else:
            with torch.no_grad(), frozen_running_statistics(self.model):
                losses = torch.cat([self._losses(part) for part in micro])
            groups = self.train.groups[index]
            self.adversary.step(group_losses(losses, groups, NUM_GROUPS))
            counts = group_counts(groups).clamp(min=1)
            for part in micro:
                part_groups = self.train.groups[part]
                partial = (
                    group_losses(self._losses(part), part_groups, NUM_GROUPS)
                    * group_counts(part_groups)
                    / counts
                )
                torch.autograd.backward(self.adversary(partial))
        self.optimizer.step()


def _group_accuracy(model: nn.Module, split: Split, inputs: Inputs) -> torch.Tensor:
    """Accuracy of each group of `split`."""
    device = inputs.device
    correct = torch.zeros(NUM_GROUPS, device=device)
    for start in range(0, len(split.targets), EVAL_BATCH_SIZE):
        stop = start + EVAL_BATCH_SIZE
        images = split.images[start:stop].to(device)
        targets = split.targets[start:stop].to(device)
        groups = split.groups[start:stop].to(device)
        with torch.no_grad(), inputs.context():
            predictions = model(inputs(images)).argmax(dim=1)
        correct.scatter_add_(0, groups, (predictions == targets).float())
    counts = group_counts(split.groups.to(device)).clamp(min=1)
    return (correct / counts).cpu()


def evaluate(model: nn.Module, data: Waterbirds, inputs: Inputs) -> Evaluation:
    """Return the group accuracies of `model` on every split, in evaluation mode."""
    model.eval()
    try:
        return Evaluation(
            *(
                _group_accuracy(model, split, inputs)
                for split in (data.train, data.validation, data.test)
            )
        )
    finally:
        model.train()


def train(
    setting: Setting,
    data: Waterbirds,
    generator: torch.Generator,
    *,
    batch_size: int,
    micro_batch_size: int,
    eval_interval: int,
    precision: str,
    pretrained: bool,
    device: torch.device,
    model_factory: Callable[[bool], nn.Module] = build_model,
) -> Run:
    """Train one model and evaluate it along the way and after the last epoch.

    Args:
        setting: The run.
        data: Training, validation and test splits on the CPU.
        generator: CPU generator of the sampling, seeded by the caller.
        batch_size: Examples per optimizer step.
        micro_batch_size: Examples per forward and backward pass.
        eval_interval: Epochs between evaluations.
        precision: `"bfloat16"` or `"float32"`.
        pretrained: Whether to start from the ImageNet weights.
        device: Device of the model and of the training set.
        model_factory: Builds the model from `pretrained`.

    Returns:
        The evaluations and the timing of the run.
    """
    inputs = Inputs(precision, device)
    model = model_factory(pretrained).to(device).to(memory_format=inputs.memory_format)
    model.train()
    train_split = Split(
        data.train.images.to(device),
        data.train.targets.to(device),
        data.train.groups.to(device),
    )
    trainer = _Trainer(model, setting, train_split, micro_batch_size, inputs)
    proportions = group_counts(data.train.groups).float() / len(data.train.groups)
    epochs: list[int] = []
    evaluations: list[Evaluation] = []
    elapsed = 0.0
    for epoch in range(1, setting.epochs + 1):
        order = epoch_order(setting.method, data.train.groups, generator).to(device)
        _synchronize(device)
        start = time.perf_counter()
        for index in torch.split(order, batch_size):
            trainer.step(index)
        _synchronize(device)
        elapsed += time.perf_counter() - start
        if epoch % eval_interval == 0 or epoch == setting.epochs:
            epochs.append(epoch)
            evaluations.append(evaluate(model, data, inputs))
    return Run(
        setting=setting,
        epochs=tuple(epochs),
        evaluations=tuple(evaluations),
        train_proportions=proportions,
        seconds_per_epoch=elapsed / setting.epochs,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
