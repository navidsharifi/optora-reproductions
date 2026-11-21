r"""Norm-ball adversaries that the paper compares with, FGM, IFGM and PGM.

Each adversary perturbs every example inside the ball
$\{x : \lVert x - x_0 \rVert_p \le \epsilon\}$ along the steepest ascent
direction of the loss for the norm $p \in \{2, \infty\}$,

$$
\eta(x) = \arg\max_{\lVert \eta \rVert_p \le 1} \nabla_x \ell(\theta; x)^\top \eta,
$$

which is $\nabla_x \ell / \lVert \nabla_x \ell \rVert_2$ for $p = 2$ and
$\mathrm{sign}(\nabla_x \ell)$ for $p = \infty$ (Eq. 3).

- FGM takes one step of length $\epsilon$ (Goodfellow et al. 2015).
- IFGM takes $T$ steps of length $\epsilon / T$, the basic iterative method
  of Kurakin et al. (2017) with a step small enough to stay in the ball.
- PGM takes $T$ steps of length $\alpha = 2.5 \epsilon / T$, each followed
  by the projection onto the ball (Eq. 18), with the step rule of Madry et
  al. (2018) and no random start.

The paper sets $T = 15$ and leaves the step sizes unstated, so these two
step rules were fixed before the full run. No adversary clips the pixels to
$[0, 1]$, as the authors' WRM adversary does not either.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

NUM_STEPS = 15
PGM_STEP = 2.5
NORMS = (2.0, float("inf"))

Loss = Callable[[torch.Tensor], torch.Tensor]


def _check(norm: float) -> None:
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm}")


def _gradient(loss: Loss, points: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        points = points.detach().requires_grad_(True)
        (gradient,) = torch.autograd.grad(loss(points).sum(), points)
    return gradient


def steepest_direction(gradient: torch.Tensor, norm: float) -> torch.Tensor:
    """Return the unit vector of the norm that maximizes the inner product.

    Args:
        gradient: Gradient of each example, with the examples first.
        norm: `2.0` or `float("inf")`.

    Returns:
        The direction of every example, zero where the gradient vanishes.
    """
    _check(norm)
    if norm == float("inf"):
        return gradient.sign()
    shape = (-1,) + (1,) * (gradient.dim() - 1)
    length = torch.linalg.vector_norm(gradient.flatten(1), dim=1).reshape(shape)
    direction: torch.Tensor = gradient / length.clamp_min(
        torch.finfo(gradient.dtype).tiny
    )
    return direction


def project(
    points: torch.Tensor, origin: torch.Tensor, radius: float, norm: float
) -> torch.Tensor:
    """Project every example onto the ball of `radius` around its origin."""
    _check(norm)
    offset = points - origin
    if norm == float("inf"):
        return origin + offset.clamp(-radius, radius)
    shape = (-1,) + (1,) * (offset.dim() - 1)
    length = torch.linalg.vector_norm(offset.flatten(1), dim=1).reshape(shape)
    scale = (radius / length.clamp_min(torch.finfo(offset.dtype).tiny)).clamp(max=1.0)
    projected: torch.Tensor = origin + offset * scale
    return projected


def fgm(loss: Loss, inputs: torch.Tensor, radius: float, norm: float) -> torch.Tensor:
    """Return the fast-gradient perturbation of every example (Eq. 3)."""
    step = steepest_direction(_gradient(loss, inputs), norm)
    return (inputs + radius * step).detach()


def ifgm(
    loss: Loss,
    inputs: torch.Tensor,
    radius: float,
    norm: float,
    num_steps: int = NUM_STEPS,
) -> torch.Tensor:
    """Return the iterated fast-gradient perturbation, steps of `radius / num_steps`."""
    points = inputs.detach()
    for _ in range(num_steps):
        step = steepest_direction(_gradient(loss, points), norm)
        points = points + (radius / num_steps) * step
    return points.detach()


def pgm(
    loss: Loss,
    inputs: torch.Tensor,
    radius: float,
    norm: float,
    num_steps: int = NUM_STEPS,
) -> torch.Tensor:
    """Return the projected-gradient perturbation of Eq. (18)."""
    origin = inputs.detach()
    size = PGM_STEP * radius / num_steps
    points = origin
    for _ in range(num_steps):
        step = steepest_direction(_gradient(loss, points), norm)
        points = project(points + size * step, origin, radius, norm)
    return points.detach()
