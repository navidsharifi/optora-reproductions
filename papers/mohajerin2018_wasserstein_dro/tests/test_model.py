from __future__ import annotations

import optora
import pytest
import torch

from ..data import NUM_ASSETS, return_moments, sample_returns
from ..model import (
    ALPHA,
    RHO,
    closed_form_worst_case,
    finite_support,
    lp_portfolio,
    out_of_sample,
    portfolio_loss,
    robust_portfolios,
    simplex_basis,
)

F64 = torch.float64
EQUAL = torch.full((NUM_ASSETS,), 1.0 / NUM_ASSETS, dtype=F64)


def random_portfolios(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weights = torch.rand(count, NUM_ASSETS, generator=generator, dtype=F64)
    return weights / weights.sum(dim=-1, keepdim=True)


def test_loss_is_the_mean_cvar_integrand() -> None:
    samples = sample_returns([0], 50)[0]
    x, tau = random_portfolios(1, 0)[0], torch.tensor(0.05, dtype=F64)
    returns = samples @ x

    expected = -returns + RHO * tau + RHO / ALPHA * torch.clamp(-returns - tau, min=0)

    assert torch.allclose(portfolio_loss(samples, x, tau), expected, atol=1e-14)


def test_loss_broadcasts_portfolios_against_points() -> None:
    points = sample_returns([0, 1], 6).unsqueeze(1)
    x = random_portfolios(6, 1).reshape(2, 3, NUM_ASSETS)
    tau = torch.zeros(2, 3, dtype=F64)
    assert portfolio_loss(points, x, tau).shape == (2, 3, 6)


def test_finite_support_layout() -> None:
    samples = sample_returns([0, 1], 7)

    points, nominal, cost = finite_support(samples, 100.0)

    support = 7 + 2 * NUM_ASSETS
    assert points.shape == (2, support, NUM_ASSETS)
    assert torch.equal(points[:, :7], samples)
    assert torch.equal(
        points[0, 7:], torch.cat([100 * torch.eye(10), -100 * torch.eye(10)]).double()
    )
    assert nominal.sum().item() == pytest.approx(1.0, abs=1e-15)
    assert torch.all(nominal[7:] == 0)
    assert cost.shape == (2, support, support)
    assert torch.allclose(cost, cost.transpose(-1, -2))
    assert torch.all(cost.diagonal(dim1=-2, dim2=-1) == 0)
    assert torch.allclose(cost[1, 0, 3], (samples[1, 0] - samples[1, 3]).abs().sum())


@pytest.mark.parametrize("radius", [1e-4, 1e-2, 0.3, 1.0])
def test_finite_support_reproduces_the_continuous_worst_case(radius: float) -> None:
    samples = sample_returns([0, 1, 2], 30)
    x = random_portfolios(3, 2)
    tau = torch.tensor([-0.1, 0.0, 0.2], dtype=F64)
    points, nominal, cost = finite_support(samples, 1e6)

    value = optora.WassersteinAmbiguitySet(
        nominal, cost, radius
    ).worst_case_expectation(portfolio_loss(points, x, tau))

    expected = closed_form_worst_case(samples, x, tau, radius)
    assert torch.all(value <= expected + 1e-9)
    assert torch.allclose(value, expected, atol=1e-6)


def test_closer_far_points_give_a_lower_worst_case() -> None:
    samples = sample_returns([0], 20)
    x, tau = random_portfolios(1, 3), torch.zeros(1, dtype=F64)
    values = []
    for far in (10.0, 1e3, 1e6):
        points, nominal, cost = finite_support(samples, far)
        ambiguity_set = optora.WassersteinAmbiguitySet(nominal, cost, 0.1)
        values.append(
            ambiguity_set.worst_case_expectation(portfolio_loss(points, x, tau))
        )
    assert values[0] < values[1] < values[2]


def test_simplex_basis_is_orthonormal_and_sums_to_zero() -> None:
    basis = simplex_basis(NUM_ASSETS, F64)

    assert basis.shape == (NUM_ASSETS, NUM_ASSETS - 1)
    assert torch.allclose(
        basis.T @ basis, torch.eye(NUM_ASSETS - 1, dtype=F64), atol=1e-14
    )
    assert torch.allclose(
        basis.sum(dim=0), torch.zeros(NUM_ASSETS - 1, dtype=F64), atol=1e-14
    )


def test_out_of_sample_matches_monte_carlo() -> None:
    mean, covariance = return_moments()
    x = random_portfolios(1, 4)[0]
    losses = -(sample_returns([9], 400_000)[0] @ x)
    tail = torch.sort(losses, descending=True).values[: int(ALPHA * losses.numel())]

    estimate = losses.mean() + RHO * tail.mean()

    assert out_of_sample(x, mean, covariance).item() == pytest.approx(
        estimate.item(), abs=5e-3
    )


def test_out_of_sample_of_equal_weights_is_on_the_scale_of_figure_5() -> None:
    mean, covariance = return_moments()
    assert out_of_sample(EQUAL, mean, covariance).item() == pytest.approx(
        -1.07, abs=0.01
    )


def test_out_of_sample_is_batched() -> None:
    mean, covariance = return_moments()
    x = random_portfolios(6, 5).reshape(2, 3, NUM_ASSETS)
    values = out_of_sample(x, mean, covariance)
    assert values.shape == (2, 3)
    assert torch.allclose(values[1, 2], out_of_sample(x[1, 2], mean, covariance))


@pytest.mark.parametrize("radius", [1e-4, 1e-2, 1.0])
def test_lp_value_equals_the_closed_form_at_its_solution(radius: float) -> None:
    samples = sample_returns([3], 40)[0]

    x, tau, value = lp_portfolio(samples.numpy(), radius)

    assert x.min() >= -1e-12
    assert x.sum() == pytest.approx(1.0, abs=1e-10)
    closed = closed_form_worst_case(
        samples, torch.from_numpy(x), torch.tensor(tau, dtype=F64), radius
    )
    assert closed.item() == pytest.approx(value, abs=1e-8)


def test_lp_gives_equal_weights_at_a_large_radius() -> None:
    x, _, _ = lp_portfolio(sample_returns([0], 300)[0].numpy(), 1.0)
    assert torch.allclose(torch.from_numpy(x), EQUAL, atol=1e-9)


def test_ellipsoid_method_reaches_the_lp_optimum() -> None:
    samples = sample_returns([0, 1], 10)
    radii = [1e-3, 0.05, 1.0]

    x, tau, certificate = robust_portfolios(
        samples, torch.tensor(radii, dtype=F64), 1e6, 1500
    )

    assert x.shape == (2, 3, NUM_ASSETS) and tau.shape == certificate.shape == (2, 3)
    assert torch.all(x >= 0)
    assert torch.allclose(x.sum(dim=-1), torch.ones(2, 3, dtype=F64), atol=1e-12)
    reference = torch.tensor(
        [[lp_portfolio(d.numpy(), r)[2] for r in radii] for d in samples], dtype=F64
    )
    assert torch.all(certificate >= reference - 1e-6)
    assert torch.allclose(certificate, reference, atol=1e-3)


def test_ellipsoid_certificate_is_optoras_value_at_the_solution() -> None:
    samples = sample_returns([2], 8)
    radius = torch.tensor([0.1], dtype=F64)

    x, tau, certificate = robust_portfolios(samples, radius, 1e6, 100)

    points, nominal, cost = finite_support(samples, 1e6)
    value = optora.WassersteinAmbiguitySet(nominal, cost, 0.1).worst_case_expectation(
        portfolio_loss(points, x[:, 0], tau[:, 0])
    )
    assert torch.allclose(certificate[:, 0], value, atol=1e-12)


def test_one_step_returns_the_starting_portfolio() -> None:
    x, tau, _ = robust_portfolios(
        sample_returns([0], 5), torch.tensor([0.1], dtype=F64), 1e6, 1
    )
    assert torch.allclose(x[0, 0], torch.eye(NUM_ASSETS, dtype=F64)[-1], atol=1e-15)
    assert tau.item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu() -> None:
    samples = sample_returns([0, 1], 10)
    radii = torch.tensor([1e-2, 1.0], dtype=F64)
    cpu = robust_portfolios(samples, radii, 1e6, 100)
    cuda = robust_portfolios(samples.cuda(), radii.cuda(), 1e6, 100)
    for a, b in zip(cpu, cuda, strict=True):
        assert torch.allclose(a, b.cpu(), atol=1e-8)
