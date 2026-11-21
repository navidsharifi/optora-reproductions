from __future__ import annotations

import pytest
import torch

from ..attacks import NUM_STEPS, fgm, ifgm, pgm, project, steepest_direction

INF = float("inf")
SLOPE = torch.tensor([[3.0, -4.0], [0.0, 2.0]], dtype=torch.float64)
INPUTS = torch.tensor([[1.0, 1.0], [-1.0, 0.5]], dtype=torch.float64)


def linear(points: torch.Tensor) -> torch.Tensor:
    return (points * SLOPE).sum(1)


def test_steepest_direction_has_unit_norm() -> None:
    gradient = torch.tensor([[3.0, 4.0], [0.0, 0.0], [-1.0, 2.0]])

    l2 = steepest_direction(gradient, 2.0)
    linf = steepest_direction(gradient, INF)

    torch.testing.assert_close(l2[0], torch.tensor([0.6, 0.8]))
    assert torch.equal(l2[1], torch.zeros(2))
    assert linf.tolist() == [[1.0, 1.0], [0.0, 0.0], [-1.0, 1.0]]


def test_project_keeps_inner_points_and_pulls_outer_points_to_the_sphere() -> None:
    origin = torch.zeros(2, 2)
    points = torch.tensor([[0.3, 0.4], [3.0, 4.0]])

    l2 = project(points, origin, 1.0, 2.0)
    linf = project(points, origin, 1.0, INF)

    torch.testing.assert_close(l2, torch.tensor([[0.3, 0.4], [0.6, 0.8]]))
    torch.testing.assert_close(linf, torch.tensor([[0.3, 0.4], [1.0, 1.0]]))


@pytest.mark.parametrize("norm", [2.0, INF])
def test_every_adversary_solves_a_linear_loss_exactly(norm: float) -> None:
    exact = INPUTS + 0.5 * steepest_direction(SLOPE, norm)

    for attack in (fgm, ifgm, pgm):
        torch.testing.assert_close(attack(linear, INPUTS, 0.5, norm), exact)


def test_pgm_stays_in_the_ball_on_a_curved_loss() -> None:
    torch.manual_seed(0)
    network = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ELU())
    inputs = torch.randn(8, 4)

    def loss(points: torch.Tensor) -> torch.Tensor:
        outputs: torch.Tensor = network(points)
        return outputs.square().sum(1)

    for norm, radius in ((2.0, 0.7), (INF, 0.2)):
        points = pgm(loss, inputs, radius, norm)
        lengths = torch.linalg.vector_norm(points - inputs, ord=norm, dim=1)
        assert bool((lengths <= radius + 1e-6).all())
        with torch.no_grad():
            assert float(loss(points).mean()) > float(loss(inputs).mean())


def test_ifgm_moves_by_exactly_the_radius_along_a_constant_direction() -> None:
    points = ifgm(linear, INPUTS, 0.9, 2.0, num_steps=NUM_STEPS)
    lengths = torch.linalg.vector_norm(points - INPUTS, dim=1)
    torch.testing.assert_close(lengths, torch.full((2,), 0.9, dtype=torch.float64))


def test_adversaries_return_detached_points() -> None:
    inputs = INPUTS.clone().requires_grad_(True)
    assert not fgm(linear, inputs, 0.1, 2.0).requires_grad
    assert not pgm(linear, inputs, 0.1, INF).requires_grad


@pytest.mark.parametrize("norm", [1.0, 3.0])
def test_rejects_other_norms(norm: float) -> None:
    with pytest.raises(ValueError, match="norm must be one of"):
        fgm(linear, INPUTS, 0.1, norm)
    with pytest.raises(ValueError, match="norm must be one of"):
        project(INPUTS, INPUTS, 0.1, norm)
