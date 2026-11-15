r"""Independent check against a conic solver (workflow step 6).

For fixed parameters the inner supremum over the paper's ball
$\{p \in \Delta : \|np - \mathbf 1\|_2 \le \sqrt{2\rho}\}$ has the conic dual
$\bar a + \tfrac1n \min_{\eta, \mu \ge 0} \sqrt{2\rho}\,\|a - \eta\mathbf 1 + \mu\|_2
+ \mathbf 1^\top\mu$, so the whole robust problem is one second-order cone
program that CVXPY solves without any Optora code.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from ..data import sample_signs
from ..model import (
    chi_square_radius,
    erm_solution,
    minimize_robust,
    optimum,
    population_risk,
    radius_rho,
    robust_objective,
)

cp = pytest.importorskip("cvxpy")

D, R, B = 50, 10.0, 1.0


def solve_with_cvxpy(
    features: np.ndarray, v: np.ndarray, rho: float
) -> tuple[Any, float]:
    num_samples = features.shape[0]
    theta = cp.Variable(features.shape[1])
    eta = cp.Variable()
    mu = cp.Variable(num_samples, nonneg=True)
    linear = features @ (theta - v)
    objective = (
        0.5 * cp.sum_squares(theta - v)
        + cp.sum(linear) / num_samples
        + (math.sqrt(2 * rho) * cp.norm(linear - eta + mu, 2) + cp.sum(mu))
        / num_samples
    )
    problem = cp.Problem(cp.Minimize(objective), [cp.norm(theta, 2) <= R])
    problem.solve(solver="CLARABEL")
    return theta.value, float(problem.value)


@pytest.fixture(scope="module", params=[100, 500])
def instance(request: pytest.FixtureRequest) -> dict[str, Any]:
    num_samples = int(request.param)
    rho = radius_rho(num_samples, B, D, R, 0.05)
    features = sample_signs(11, num_samples, 6, num_samples, D).double() * B
    v = optimum(D, R, torch.float64, torch.device("cpu"))
    reference = [solve_with_cvxpy(x.numpy(), v.numpy(), rho) for x in features]
    return {
        "features": features,
        "v": v,
        "objective": robust_objective(features, v, chi_square_radius(rho, num_samples)),
        "theta": torch.tensor(np.stack([theta for theta, _ in reference])),
        "value": torch.tensor([value for _, value in reference], dtype=torch.float64),
    }


def test_optora_value_at_the_cvxpy_optimum_equals_the_cvxpy_value(
    instance: dict[str, Any],
) -> None:
    value = instance["objective"](instance["theta"])
    assert torch.allclose(value, instance["value"], atol=1e-6)


def test_robust_solver_reaches_the_cvxpy_optimum(instance: dict[str, Any]) -> None:
    features, v = instance["features"], instance["v"]
    start = erm_solution(features, v, R)

    _, theta = minimize_robust(instance["objective"], start, R, 300, 2000)

    # The averaged subgradient phase converges at rate O(1/k); after 2,000
    # steps its objective gap is about 1e-3 (the full run uses 5,000).
    gap = instance["objective"](theta) - instance["value"]
    assert torch.all(gap <= 5e-3)
    risk = population_risk(theta, v).mean()
    reference_risk = population_risk(instance["theta"], v).mean()
    assert risk.item() == pytest.approx(reference_risk.item(), abs=1e-3)
