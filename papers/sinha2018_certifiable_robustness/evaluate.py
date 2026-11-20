r"""Accuracy, error under attack, and the Lagrangian worst case of a model.

For a penalty $\gamma$ the worst case of a model on a sample is found by
moving every example to its transport map $T_\gamma(\theta; Z)$. The mean
transport cost is the robustness level

$$
\hat\rho(\gamma) = \mathbb E_{\hat P}\big[\lVert T_\gamma(\theta; Z) - Z \rVert_2^2\big]
$$

(Eqs. 10 and 20), and by the duality of Proposition 1 the mean loss at the
moved examples is the worst-case loss over the Wasserstein ball of that
radius,

$$
\mathbb E_{\hat P}[\phi_\gamma(\theta; Z)] + \gamma \hat\rho(\gamma)
    = \mathbb E_{\hat P}[\ell(\theta; T_\gamma(\theta; Z))]
    = \sup_{P : W_c(P, \hat P) \le \hat\rho(\gamma)} \mathbb E_P[\ell(\theta; Z)]
$$

(Eq. 14). Sweeping $\gamma$ traces the worst-case loss as a function of the
radius, the red curves of Figure 2 and the curves of Figure 4(a).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from prototypes.continuous_wasserstein import squared_distance, transport_map

from .attacks import pgm
from .model import cross_entropy


@dataclass(frozen=True)
class WorstCase:
    r"""Robustness level and worst-case loss of a model for one penalty.

    Attributes:
        penalty: The penalty $\gamma$.
        distance: Mean transport cost $\hat\rho(\gamma)$.
        loss: Mean loss at the transported examples.
    """

    penalty: float
    distance: float
    loss: float


def _batches(size: int, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    return list(torch.split(torch.arange(size, device=device), batch_size))


def accuracy(
    model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> float:
    """Return the fraction of examples the model classifies correctly."""
    correct = torch.zeros((), device=inputs.device)
    with torch.no_grad():
        for batch in _batches(len(inputs), batch_size, inputs.device):
            predictions = model(inputs[batch]).argmax(dim=1)
            correct += (predictions == labels[batch]).sum()
    return float(correct) / len(inputs)


def pgm_error(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    radius: float,
    norm: float,
    batch_size: int,
) -> float:
    r"""Return the test error under the PGM attack of Figure 3.

    Args:
        model: Network under attack.
        inputs: Test inputs.
        targets: One-hot target distribution of every test example.
        radius: Radius $\epsilon_{\mathrm{adv}}$ of the attack.
        norm: `2.0` or `float("inf")`.
        batch_size: Examples per attacked batch.

    Returns:
        The fraction of perturbed examples that are misclassified.
    """
    labels = targets.argmax(dim=1)
    wrong = torch.zeros((), device=inputs.device)
    for batch in _batches(len(inputs), batch_size, inputs.device):

        def loss(points: torch.Tensor, batch: torch.Tensor = batch) -> torch.Tensor:
            return cross_entropy(model(points), targets[batch])

        points = (
            inputs[batch] if radius == 0 else pgm(loss, inputs[batch], radius, norm)
        )
        with torch.no_grad():
            wrong += (model(points).argmax(dim=1) != labels[batch]).sum()
    return float(wrong) / len(inputs)


def worst_case(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    penalty: float,
    num_steps: int,
    batch_size: int,
) -> WorstCase:
    r"""Return the robustness level and worst-case loss for one penalty.

    Args:
        model: Network whose loss is perturbed.
        inputs: Examples of the sample.
        targets: Target distribution of every example.
        penalty: The penalty $\gamma$.
        num_steps: Ascent steps of the transport map.
        batch_size: Examples per batch.

    Returns:
        The mean transport cost and the mean loss at the transported examples.
    """
    distance = torch.zeros((), device=inputs.device)
    total = torch.zeros((), device=inputs.device)
    for batch in _batches(len(inputs), batch_size, inputs.device):
        origin = inputs[batch]

        def loss(points: torch.Tensor, batch: torch.Tensor = batch) -> torch.Tensor:
            return cross_entropy(model(points), targets[batch])

        points = transport_map(loss, origin, penalty, num_steps)
        with torch.no_grad():
            distance += squared_distance(points, origin).sum()
            total += loss(points).sum()
    return WorstCase(penalty, float(distance) / len(inputs), float(total) / len(inputs))


def worst_case_curve(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    penalties: Sequence[float],
    num_steps: int,
    batch_size: int,
) -> list[WorstCase]:
    """Return the worst case of a model for every penalty."""
    return [
        worst_case(model, inputs, targets, penalty, num_steps, batch_size)
        for penalty in penalties
    ]


def loss_at(curve: Sequence[WorstCase], distance: float) -> float:
    r"""Interpolate the worst-case loss of a curve linearly at a robustness level.

    Args:
        curve: Worst cases for several penalties.
        distance: Robustness level $\rho$ to read the loss at.

    Returns:
        The worst-case loss at `distance`.

    Raises:
        ValueError: If `distance` lies outside the levels of the curve, where
            the interpolation would silently extrapolate.
    """
    points = sorted((point.distance, point.loss) for point in curve)
    levels = np.array([level for level, _ in points])
    if not levels[0] <= distance <= levels[-1]:
        raise ValueError(
            f"distance {distance} lies outside [{levels[0]}, {levels[-1]}]"
        )
    return float(np.interp(distance, levels, [loss for _, loss in points]))
