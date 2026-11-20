r"""Networks, losses and training of the MNIST and synthetic experiments.

The MNIST settings follow the authors' `train_mnist_models.py`. The network
has three convolutions with ELU activations and a linear output, Adam runs
with learning rate $10^{-3}$ and batch size 256 for 25 epochs, and the
training labels are smoothed by clipping the one-hot vectors to
$[0.1 / 9, 0.9]$. Every method minimizes the loss at its perturbed inputs
only. WRM uses `eps` $= 1.3$, which is $1 / (2\gamma)$ for
$\gamma = 1 / 2.6 \approx 0.04\, C_2$, where $C_2 = \mathbb E \lVert X
\rVert_2 = 9.21$ (Section 4.2), and FGM, IFGM and PGM use the $ell_2$
radius $\epsilon = \sqrt{\hat\rho_n(\theta_{\mathrm{WRM}})}$ (Eq. 19).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn

from prototypes.continuous_wasserstein import transport_map

from .attacks import fgm, ifgm, pgm
from .data import NUM_CLASSES, Split

ERM = "erm"
FGM = "fgm"
IFGM = "ifgm"
PGM = "pgm"
WRM = "wrm"
METHODS = (ERM, FGM, IFGM, PGM, WRM)
PENALTY = 1.0 / 2.6
WRM_STEPS = 16
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1
SYNTHETIC_PENALTY = 2.0
SYNTHETIC_STEPS = 15
SYNTHETIC_LEARNING_RATE = 1e-2

Adversary = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _glorot(module: nn.Module) -> None:
    """Initialize like Keras, Glorot-uniform weights and zero biases."""
    if isinstance(module, nn.Conv2d | nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def mnist_network() -> nn.Sequential:
    """Return the authors' `cnn_model(activation="elu")` with 64 filters.

    The first convolution uses TensorFlow's `"same"` padding, which for an
    8 by 8 kernel with stride 2 on 28 by 28 inputs pads 3 pixels on each
    side.
    """
    network = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3),
        nn.ELU(),
        nn.Conv2d(64, 128, kernel_size=6, stride=2),
        nn.ELU(),
        nn.Conv2d(128, 128, kernel_size=5, stride=1),
        nn.ELU(),
        nn.Flatten(),
        nn.Linear(128, NUM_CLASSES),
    )
    network.apply(_glorot)
    return network


def synthetic_network() -> nn.Sequential:
    """Return the network of Section 4.1, hidden layers of 4 and 2 ELU units."""
    network = nn.Sequential(
        nn.Linear(2, 4), nn.ELU(), nn.Linear(4, 2), nn.ELU(), nn.Linear(2, 2)
    )
    network.apply(_glorot)
    return network


def one_hot(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Return the one-hot target distribution of every label."""
    return nn.functional.one_hot(labels, num_classes).float()


def smoothed(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Return the authors' smoothed targets, one-hot clipped to `[0.1/9, 0.9]`."""
    low = LABEL_SMOOTHING / (num_classes - 1)
    return one_hot(labels, num_classes).clamp(low, 1.0 - LABEL_SMOOTHING)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the cross-entropy of every example against a target distribution."""
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)


def batch_starts(size: int, batch_size: int) -> list[int]:
    """Return the first index of every batch, as the authors' `batch_indices`.

    The last batch is shifted back so that it is full, so it repeats a few
    examples of the batch before it.

    Raises:
        ValueError: If `batch_size` is not in `[1, size]`.
    """
    if not 1 <= batch_size <= size:
        raise ValueError(f"batch_size must be in [1, {size}], got {batch_size}")
    count = -(-size // batch_size)
    return [min(index * batch_size, size - batch_size) for index in range(count)]


def adversary(method: str, model: nn.Module, radius: float) -> Adversary:
    r"""Return the training adversary of a method.

    Args:
        method: One of `METHODS`.
        model: Network whose loss the adversary increases.
        radius: $\ell_2$ radius of FGM, IFGM and PGM.

    Returns:
        A function of a batch of inputs and their targets that returns the
        perturbed inputs.

    Raises:
        ValueError: If `method` is unknown.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    def perturb(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        def loss(points: torch.Tensor) -> torch.Tensor:
            return cross_entropy(model(points), targets)

        if method == WRM:
            return transport_map(loss, inputs, PENALTY, WRM_STEPS)
        if method == FGM:
            return fgm(loss, inputs, radius, 2.0)
        if method == IFGM:
            return ifgm(loss, inputs, radius, 2.0)
        if method == PGM:
            return pgm(loss, inputs, radius, 2.0)
        return inputs

    return perturb


def train_mnist(
    model: nn.Module,
    method: str,
    train: Split,
    order: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    radius: float = 0.0,
) -> float:
    r"""Train a network in place with Adam on the perturbed inputs of a method.

    Args:
        model: Network to train.
        method: One of `METHODS`.
        train: Training split on the device of `model`.
        order: Permutation of the training set, drawn once per seed and
            kept for every epoch, as the authors shuffle MNIST once.
        epochs: Passes over the training set.
        batch_size: Examples per step.
        radius: $\ell_2$ radius of FGM, IFGM and PGM.

    Returns:
        Seconds per epoch.
    """
    inputs = train.inputs[order]
    targets = smoothed(train.labels[order])
    perturb = adversary(method, model, radius)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    starts = batch_starts(len(inputs), batch_size)
    clock = time.perf_counter()
    for _ in range(epochs):
        for start in starts:
            batch = slice(start, start + batch_size)
            points = perturb(inputs[batch], targets[batch])
            loss = cross_entropy(model(points), targets[batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
    if inputs.is_cuda:
        torch.cuda.synchronize(inputs.device)
    return (time.perf_counter() - clock) / epochs


def train_synthetic(
    model: nn.Module,
    train: Split,
    generator: torch.Generator,
    *,
    epochs: int,
    batch_size: int,
) -> None:
    r"""Train the synthetic network in place with WRM at $\gamma = 2$.

    The paper gives neither the optimizer nor the epochs of Section 4.1.
    Adam with learning rate $10^{-2}$, 25 epochs and batches of 256 drawn
    from a new permutation every epoch were fixed before the full run.

    Args:
        model: Network to train.
        train: Training split on the device of `model`.
        generator: CPU generator of the permutations.
        epochs: Passes over the training set.
        batch_size: Examples per step.
    """
    targets = one_hot(train.labels, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=SYNTHETIC_LEARNING_RATE)
    for _ in range(epochs):
        order = torch.randperm(len(targets), generator=generator).to(targets.device)
        for batch in torch.split(order, batch_size):
            inputs = train.inputs[batch]

            def loss(points: torch.Tensor, batch: torch.Tensor = batch) -> torch.Tensor:
                return cross_entropy(model(points), targets[batch])

            points = transport_map(loss, inputs, SYNTHETIC_PENALTY, SYNTHETIC_STEPS)
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss(points).mean())
            optimizer.step()
