r"""Loss, estimators and solver of the Section 5.2 simulation.

The loss is $\ell(\theta; x) = \tfrac12\|\theta - v\|_2^2 + x^\top(\theta - v)$
on $\Theta = \{\theta : \|\theta\|_2 \le r\}$. Every function works on a
batch of independent replications along the leading dimension.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import optora
import torch

Objective = Callable[[torch.Tensor], torch.Tensor]


def radius_rho(
    num_samples: int,
    noise_level: float,
    dimension: int,
    domain_radius: float,
    delta: float,
) -> float:
    r"""Robustness parameter $\rho$ of Eq. (17).

    $\rho = \log(2/\delta) + d \log(2 n D L)$ with $D = 2r$ the diameter of
    $\Theta$ and $L = 3r + \sqrt d B$ the Lipschitz constant of the loss
    stated in Section 5.2.

    Args:
        num_samples: Sample size $n$.
        noise_level: Noise level $B$.
        dimension: Dimension $d$.
        domain_radius: Radius $r$ of $\Theta$.
        delta: Confidence level $\delta$.

    Returns:
        The value of $\rho$.
    """
    diameter = 2.0 * domain_radius
    lipschitz = 3.0 * domain_radius + math.sqrt(dimension) * noise_level
    return math.log(2.0 / delta) + dimension * math.log(
        2.0 * num_samples * diameter * lipschitz
    )


def chi_square_radius(rho: float, num_samples: int) -> float:
    r"""Optora chi-square radius of the paper's ball $D_\phi \le \rho/n$.

    The paper uses $\phi(t) = \tfrac12(t-1)^2$, while Optora's chi-square
    divergence is $\sum_i (q_i - p_i)^2 / p_i = 2 D_\phi$, so the same ball
    has Optora radius $2\rho/n$.

    Args:
        rho: Robustness parameter $\rho$.
        num_samples: Sample size $n$.

    Returns:
        The radius to pass to `optora.ChiSquareAmbiguitySet`.
    """
    return 2.0 * rho / num_samples


def optimum(
    dimension: int, domain_radius: float, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    r"""The population minimizer $v = \tfrac{r}{2\sqrt d}\mathbf 1$.

    Args:
        dimension: Dimension $d$.
        domain_radius: Radius $r$ of $\Theta$.
        dtype: Floating-point type.
        device: Device.

    Returns:
        Tensor of shape `(dimension,)`.
    """
    value = domain_radius / (2.0 * math.sqrt(dimension))
    return torch.full((dimension,), value, dtype=dtype, device=device)


def project(theta: torch.Tensor, domain_radius: float) -> torch.Tensor:
    """Euclidean projection onto the ball of radius `domain_radius`.

    Args:
        theta: Points of shape `(..., d)`.
        domain_radius: Radius of the ball.

    Returns:
        The projected points, same shape as `theta`.
    """
    norm = torch.linalg.vector_norm(theta, dim=-1, keepdim=True)
    scale = torch.clamp(domain_radius / torch.clamp(norm, min=1e-300), max=1.0)
    return theta * scale


def population_risk(theta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r"""Population risk $R(\theta) = \tfrac12\|\theta - v\|_2^2$.

    Args:
        theta: Points of shape `(..., d)`.
        v: Population minimizer of shape `(d,)`.

    Returns:
        Tensor of shape `(...)`.
    """
    return 0.5 * torch.sum((theta - v) ** 2, dim=-1)


def sample_losses(
    theta: torch.Tensor, features: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Per-sample losses of every replication.

    Args:
        theta: Parameters of shape `(b, d)`.
        features: Samples of shape `(b, n, d)`.
        v: Population minimizer of shape `(d,)`.

    Returns:
        Tensor of shape `(b, n)`.
    """
    offset = theta - v
    linear = torch.bmm(features, offset.unsqueeze(-1)).squeeze(-1)
    return 0.5 * torch.sum(offset**2, dim=-1, keepdim=True) + linear


def erm_solution(
    features: torch.Tensor, v: torch.Tensor, domain_radius: float
) -> torch.Tensor:
    r"""Exact empirical risk minimizer over $\Theta$.

    The empirical risk is $\tfrac12\|\theta - v\|_2^2 + \bar X_n^\top(\theta - v)$,
    a shifted isotropic quadratic, so its minimizer over the ball is the
    projection of $v - \bar X_n$.

    Args:
        features: Samples of shape `(b, n, d)`.
        v: Population minimizer of shape `(d,)`.
        domain_radius: Radius $r$ of $\Theta$.

    Returns:
        Tensor of shape `(b, d)`.
    """
    return project(v - features.mean(dim=-2), domain_radius)


def robust_objective(
    features: torch.Tensor, v: torch.Tensor, radius: float
) -> Objective:
    r"""Robustly regularized empirical risk $R_n(\cdot, \mathcal P_n)$.

    Args:
        features: Samples of shape `(b, n, d)`.
        v: Population minimizer of shape `(d,)`.
        radius: Optora chi-square radius, see `chi_square_radius`.

    Returns:
        A function mapping parameters of shape `(b, d)` to the worst-case
        expected loss of each replication, shape `(b,)`.
    """
    num_samples = features.shape[-2]
    nominal = torch.full(
        (num_samples,), 1.0 / num_samples, dtype=features.dtype, device=features.device
    )
    ambiguity_set = optora.ChiSquareAmbiguitySet(nominal, radius=radius)

    def objective(theta: torch.Tensor) -> torch.Tensor:
        losses = sample_losses(theta, features, v)
        value: torch.Tensor = ambiguity_set.worst_case_expectation(losses)
        return value

    return objective


def minimize_projected(
    objective: Objective,
    initial: torch.Tensor,
    domain_radius: float,
    max_iterations: int,
    max_backtracks: int = 60,
    min_step: float = 1e-16,
) -> torch.Tensor:
    r"""Projected gradient descent with a backtracking line search.

    Each replication keeps its own step size. A step is accepted when the
    objective at the projected point lies below the quadratic upper model
    $f(\theta) + g^\top(\theta^+ - \theta) + \|\theta^+ - \theta\|^2 / (2t)$
    (Beck and Teboulle 2009); otherwise the step is halved. A replication
    whose step falls below `min_step` without an accepted move has reached
    the solver's numerical floor and stays fixed. Accepted steps double for
    the next iteration.

    Args:
        objective: Batched objective mapping `(b, d)` to `(b,)`.
        initial: Starting points of shape `(b, d)`, inside the ball.
        domain_radius: Radius of the feasible ball.
        max_iterations: Iteration cap.
        max_backtracks: Cap on step halvings per iteration.
        min_step: Step size below which a replication is considered done.

    Returns:
        The final points, shape `(b, d)`.
    """
    theta = initial.detach().clone()
    step = torch.ones(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
    for _ in range(max_iterations):
        active = step > min_step
        if not bool(active.any()):
            break
        point = theta.detach().requires_grad_(True)
        value = objective(point)
        (gradient,) = torch.autograd.grad(value.sum(), point)
        value = value.detach()

        accepted = torch.zeros_like(active)
        candidate = theta
        for _ in range(max_backtracks):
            with torch.no_grad():
                trial = project(theta - step.unsqueeze(-1) * gradient, domain_radius)
                move = trial - theta
                bound = (
                    value
                    + torch.sum(gradient * move, dim=-1)
                    + torch.sum(move**2, dim=-1) / (2.0 * step)
                )
                success = (objective(trial) <= bound) & active & ~accepted
            candidate = torch.where(success.unsqueeze(-1), trial, candidate)
            accepted = accepted | success
            pending = active & ~accepted
            if not bool(pending.any()):
                break
            step = torch.where(pending, 0.5 * step, step)
            active = active & (step > min_step)
        theta = candidate
        step = torch.where(accepted, 2.0 * step, step)
    return theta


def minimize_subgradient(
    objective: Objective,
    initial: torch.Tensor,
    domain_radius: float,
    iterations: int,
    strong_convexity: float = 1.0,
) -> torch.Tensor:
    r"""Projected subgradient method for a strongly convex objective.

    Uses the step $2 / (\mu (k + 1))$ and returns the average of the iterates
    weighted by $k$, which converges at rate $O(1/k)$ for a
    $\mu$-strongly convex Lipschitz objective (Lacoste-Julien, Schmidt and
    Bach 2012). The autograd gradient of Optora's worst-case expectation is
    the gradient at the computed maximizer, a valid subgradient by Danskin's
    theorem, so the method needs no smoothness.

    Args:
        objective: Batched objective mapping `(b, d)` to `(b,)`.
        initial: Starting points of shape `(b, d)`, inside the ball.
        domain_radius: Radius of the feasible ball.
        iterations: Number of subgradient steps.
        strong_convexity: Modulus $\mu$; the robust risk of Section 5.2 is
            1-strongly convex because of its $\tfrac12\|\theta - v\|_2^2$ term.

    Returns:
        The weighted average of the iterates, shape `(b, d)`.
    """
    theta = initial.detach().clone()
    average = torch.zeros_like(theta)
    for k in range(1, iterations + 1):
        point = theta.requires_grad_(True)
        (subgradient,) = torch.autograd.grad(objective(point).sum(), point)
        with torch.no_grad():
            step = 2.0 / (strong_convexity * (k + 1))
            theta = project(theta - step * subgradient, domain_radius)
            average.add_(theta, alpha=2.0 * k / (iterations * (iterations + 1)))
    return average


def minimize_robust(
    objective: Objective,
    initial: torch.Tensor,
    domain_radius: float,
    descent_iterations: int,
    subgradient_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize the robust risk, keeping the descent phase on its own.

    The descent phase is the solver of Section 5.1. It can stall where the
    robust risk is not differentiable, which happens when the chi-square
    ball is large enough to hold every distribution on a set of tied top
    losses. The subgradient phase then continues from the descent point,
    and each replication keeps whichever of the two points has the lower
    objective, so a descent point that already sits at the minimizer is
    never replaced by the slower averaged iterate.

    Args:
        objective: Batched objective mapping `(b, d)` to `(b,)`.
        initial: Starting points of shape `(b, d)`, inside the ball.
        domain_radius: Radius of the feasible ball.
        descent_iterations: Iteration cap of the descent phase.
        subgradient_iterations: Steps of the subgradient phase; 0 skips it.

    Returns:
        The descent-phase points and the final points, both `(b, d)`.
    """
    descent = minimize_projected(objective, initial, domain_radius, descent_iterations)
    if subgradient_iterations == 0:
        return descent, descent
    return descent, refine(objective, descent, domain_radius, subgradient_iterations)


def refine(
    objective: Objective,
    start: torch.Tensor,
    domain_radius: float,
    iterations: int,
) -> torch.Tensor:
    """Run the subgradient phase from `start` and keep the better point.

    Args:
        objective: Batched objective mapping `(b, d)` to `(b,)`.
        start: Points of shape `(b, d)`, usually the descent-phase output.
        domain_radius: Radius of the feasible ball.
        iterations: Steps of the subgradient phase.

    Returns:
        For each replication, whichever of `start` and the averaged iterate
        has the lower objective, shape `(b, d)`.
    """
    averaged = minimize_subgradient(objective, start, domain_radius, iterations)
    with torch.no_grad():
        better = objective(averaged) < objective(start)
    return torch.where(better.unsqueeze(-1), averaged, start)
