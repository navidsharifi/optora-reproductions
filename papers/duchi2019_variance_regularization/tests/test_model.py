from __future__ import annotations

import math

import optora
import pytest
import torch

from ..model import (
    Objective,
    chi_square_radius,
    erm_solution,
    minimize_projected,
    minimize_robust,
    minimize_subgradient,
    optimum,
    population_risk,
    project,
    radius_rho,
    robust_objective,
    sample_losses,
)

F64 = torch.float64
CPU = torch.device("cpu")


def test_radius_rho_matches_eq_17() -> None:
    lipschitz = 30.0 + math.sqrt(50)
    expected = math.log(40.0) + 50 * math.log(2 * 100 * 20.0 * lipschitz)
    assert radius_rho(100, 1.0, 50, 10.0, 0.05) == pytest.approx(expected, rel=1e-15)
    assert radius_rho(100, 1.0, 50, 10.0, 0.05) == pytest.approx(599.03, abs=0.01)


def test_radius_rho_grows_with_n_and_noise() -> None:
    assert radius_rho(1000, 1.0, 50, 10.0, 0.05) > radius_rho(100, 1.0, 50, 10.0, 0.05)
    assert radius_rho(100, 10.0, 50, 10.0, 0.05) > radius_rho(100, 1.0, 50, 10.0, 0.05)


def test_chi_square_radius_matches_paper_variance_expansion() -> None:
    # Paper Eq. (5) with epsilon_n = 0: R_n = mean + sqrt(2 rho Var / n), exact
    # while the maximizer keeps positive mass everywhere (small radius here).
    generator = torch.Generator().manual_seed(0)
    num_samples, rho = 400, 0.5
    losses = torch.rand(3, num_samples, generator=generator, dtype=F64)
    nominal = torch.full((num_samples,), 1.0 / num_samples, dtype=F64)
    ambiguity_set = optora.ChiSquareAmbiguitySet(
        nominal, radius=chi_square_radius(rho, num_samples)
    )

    value = ambiguity_set.worst_case_expectation(losses)

    variance = losses.var(dim=-1, correction=0)
    expected = losses.mean(dim=-1) + torch.sqrt(2.0 * rho * variance / num_samples)
    assert torch.allclose(value, expected, atol=1e-12)


def test_optimum_has_half_the_domain_radius() -> None:
    v = optimum(50, 10.0, F64, CPU)
    assert v.shape == (50,)
    assert torch.linalg.vector_norm(v).item() == pytest.approx(5.0, rel=1e-14)


def test_project_keeps_inside_points_and_scales_outside_points() -> None:
    inside = torch.tensor([[0.3, 0.4]], dtype=F64)
    outside = torch.tensor([[3.0, 4.0]], dtype=F64)
    zero = torch.zeros(1, 2, dtype=F64)

    assert torch.equal(project(inside, 1.0), inside)
    assert torch.allclose(project(outside, 1.0), torch.tensor([[0.6, 0.8]], dtype=F64))
    assert torch.equal(project(zero, 1.0), zero)


def test_population_risk() -> None:
    v = torch.tensor([1.0, 2.0], dtype=F64)
    theta = torch.tensor([[1.0, 2.0], [2.0, 0.0]], dtype=F64)
    assert torch.allclose(
        population_risk(theta, v), torch.tensor([0.0, 2.5], dtype=F64)
    )


def test_sample_losses_match_the_definition() -> None:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(2, 5, 3, generator=generator, dtype=F64)
    theta = torch.randn(2, 3, generator=generator, dtype=F64)
    v = torch.randn(3, generator=generator, dtype=F64)

    losses = sample_losses(theta, features, v)

    offset = theta - v
    expected = 0.5 * (offset**2).sum(-1, keepdim=True) + torch.einsum(
        "bnd,bd->bn", features, offset
    )
    assert losses.shape == (2, 5)
    assert torch.allclose(losses, expected)


def test_erm_solution_zeroes_the_empirical_gradient_inside_the_ball() -> None:
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(4, 30, 3, generator=generator, dtype=F64) * 0.1
    v = torch.tensor([0.5, 0.0, 0.0], dtype=F64)

    theta = erm_solution(features, v, 10.0)

    gradient = (theta - v) + features.mean(dim=-2)
    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-14)


def test_erm_solution_projects_when_unconstrained_minimizer_is_outside() -> None:
    features = torch.full((1, 4, 2), -100.0, dtype=F64)
    v = torch.zeros(2, dtype=F64)

    theta = erm_solution(features, v, 1.0)

    assert torch.linalg.vector_norm(theta).item() == pytest.approx(1.0)


def quadratic(center: torch.Tensor) -> Objective:
    def objective(theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((theta - center) ** 2).sum(-1)

    return objective


def test_projected_descent_solves_a_constrained_quadratic() -> None:
    center = torch.tensor([[3.0, 4.0], [0.1, -0.2]], dtype=F64)

    theta = minimize_projected(
        quadratic(center), torch.zeros(2, 2, dtype=F64), 1.0, 100
    )

    expected = project(center, 1.0)
    assert torch.allclose(theta, expected, atol=1e-10)


def test_projected_descent_never_increases_the_objective() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(6, 50, 4, generator=generator, dtype=F64)
    v = torch.zeros(4, dtype=F64)
    objective = robust_objective(features, v, radius=2.0)
    start = erm_solution(features, v, 10.0)

    theta = minimize_projected(objective, start, 10.0, 50)

    assert torch.all(objective(theta) <= objective(start) + 1e-12)


def lasso(center: torch.Tensor, weight: float) -> Objective:
    def objective(theta: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((theta - center) ** 2).sum(-1) + weight * theta.abs().sum(-1)

    return objective


def test_subgradient_method_solves_a_nonsmooth_strongly_convex_problem() -> None:
    center = torch.tensor([[2.0, 0.5, -3.0]], dtype=F64)
    soft_threshold = torch.sign(center) * torch.clamp(center.abs() - 1.0, min=0.0)

    theta = minimize_subgradient(
        lasso(center, 1.0), torch.zeros_like(center), 10.0, 4000
    )

    assert torch.allclose(theta, soft_threshold, atol=2e-2)


def test_zero_subgradient_iterations_return_the_descent_point() -> None:
    center = torch.tensor([[0.5, 0.5]], dtype=F64)
    descent, final = minimize_robust(
        quadratic(center), torch.zeros_like(center), 1.0, 50, 0
    )
    assert torch.equal(descent, final)


def test_robust_minimizer_keeps_an_exact_descent_point() -> None:
    center = torch.tensor([[0.3, -0.1]], dtype=F64)

    descent, final = minimize_robust(
        quadratic(center), torch.zeros_like(center), 1.0, 200, 200
    )

    assert torch.allclose(descent, center, atol=1e-12)
    assert torch.equal(final, descent)


def test_robust_minimizer_improves_on_a_stalled_descent() -> None:
    center = torch.tensor([[2.0, 0.5, -3.0]], dtype=F64)
    objective = lasso(center, 1.0)

    descent, final = minimize_robust(objective, torch.zeros_like(center), 10.0, 1, 2000)

    assert torch.all(objective(final) <= objective(descent))


def test_batched_solve_equals_per_replication_solves() -> None:
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(3, 40, 4, generator=generator, dtype=F64)
    v = torch.full((4,), 0.2, dtype=F64)
    start = erm_solution(features, v, 10.0)

    batched = minimize_robust(robust_objective(features, v, 3.0), start, 10.0, 30, 30)[
        1
    ]
    single = torch.cat(
        [
            minimize_robust(
                robust_objective(features[i : i + 1], v, 3.0),
                start[i : i + 1],
                10.0,
                30,
                30,
            )[1]
            for i in range(3)
        ]
    )

    assert torch.allclose(batched, single, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_robust_objective_on_cuda_matches_cpu() -> None:
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(2, 30, 3, generator=generator, dtype=F64)
    v = torch.zeros(3, dtype=F64)
    theta = torch.randn(2, 3, generator=generator, dtype=F64)
    cuda = torch.device("cuda")

    cpu_value = robust_objective(features, v, 2.0)(theta)
    cuda_value = robust_objective(features.to(cuda), v.to(cuda), 2.0)(theta.to(cuda))

    assert torch.allclose(cuda_value.cpu(), cpu_value, atol=1e-12)
