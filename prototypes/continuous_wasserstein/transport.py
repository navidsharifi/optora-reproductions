r"""Gradient ascent for the transport map of the Lagrangian Wasserstein adversary."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


def squared_distance(points: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    r"""Return the transport cost $\lVert z - z_0 \rVert_2^2$ of every example.

    Args:
        points: Transported points $z$, with the examples along the first
            dimension.
        inputs: Original points $z_0$, of the same shape as `points`.

    Returns:
        Tensor of shape `(n,)` holding the squared Euclidean distance of each
        example, summed over every dimension but the first.
    """
    return (points - inputs).flatten(1).square().sum(1)


def transport_map(
    loss: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    penalty: float | torch.Tensor,
    num_steps: int = 16,
) -> torch.Tensor:
    r"""Approximate the transport map $T_\gamma(\theta; z_0)$ by gradient ascent.

    The map is the maximizer of $\ell(\theta; z) - \gamma \lVert z - z_0
    \rVert_2^2$ (Eq. 9 of Sinha et al. 2018). The ascent runs on the same
    objective divided by $2\gamma$,

    $$
    f(z) = \frac{\ell(\theta; z)}{2\gamma} - \tfrac12 \lVert z - z_0 \rVert_2^2,
    $$

    from $z = z_0$ with the step size $1 / \sqrt{t}$ at step
    $t = 1, \dots, T$. This is the schedule of the authors' `wrm` function
    with `eps` $= 1 / (2\gamma)$, which takes a first step of size one and
    then `steps` $= T - 1$ steps of size $1 / \sqrt{t + 2}$. Dividing by
    $2\gamma$ makes the penalty term 1-strongly concave, so the first step
    from $z_0$, where its gradient vanishes, already moves by
    $\nabla_z \ell / (2\gamma)$, the exact maximizer when the loss is linear
    in $z$.

    The examples are solved jointly by differentiating the sum of their
    losses, so the loss of each example must depend on its own point only,
    which rules out batch normalization in training mode. The parameters of
    the model receive no gradient, and the result is detached, since by
    Danskin's theorem the gradient of the robust surrogate with respect to
    $\theta$ is $\nabla_\theta \ell(\theta; T_\gamma(\theta; z_0))$ (Lemma 1).

    Args:
        loss: Maps a batch of points of the shape of `inputs` to the loss of
            each example, a tensor of shape `(n,)`.
        inputs: Original points $z_0$, with the examples along the first
            dimension.
        penalty: The penalty $\gamma > 0$, a float or a tensor that
            broadcasts to `(n,)`, for example one value per example.
        num_steps: Number of ascent steps $T$.

    Returns:
        The transported points, detached and of the shape of `inputs`.

    Raises:
        ValueError: If `num_steps` is not positive, or if `penalty` is a float
            that is not positive.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")
    if isinstance(penalty, float | int) and not penalty > 0:
        raise ValueError(f"penalty must be positive, got {penalty}")
    origin = inputs.detach()
    shape = (-1,) + (1,) * (origin.dim() - 1)
    scale = torch.as_tensor(penalty, dtype=origin.dtype, device=origin.device)
    scale = (0.5 / scale).reshape(shape) if scale.dim() else 0.5 / scale
    points = origin
    with torch.enable_grad():
        for step in range(1, num_steps + 1):
            points = points.detach().requires_grad_(True)
            (gradient,) = torch.autograd.grad(loss(points).sum(), points)
            ascent = scale * gradient - (points - origin)
            points = points + ascent / math.sqrt(step)
    return points.detach()
