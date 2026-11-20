from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch
from optora.core import MinimizationProblem
from optora.solvers import GradientDescent
from torch import nn

from prototypes.continuous_wasserstein import squared_distance, transport_map


def _network(seed: int = 0, dtype: torch.dtype = torch.float64) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 8), nn.ELU(), nn.Linear(8, 2)).to(dtype)


Loss = Callable[[torch.Tensor], torch.Tensor]


def _loss(network: nn.Module, labels: torch.Tensor) -> Loss:
    def loss(points: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(network(points), labels, reduction="none")

    return loss


def _authors_wrm(
    loss: Loss, inputs: torch.Tensor, eps: float, steps: int
) -> torch.Tensor:
    """The loop of `attacks_tf.wrm` in the authors' code, written in PyTorch."""
    x = inputs.detach().requires_grad_(True)
    (grad,) = torch.autograd.grad((eps * loss(x)).sum(), x)
    x_adv = (inputs + grad).detach()
    for t in range(steps):
        x_adv.requires_grad_(True)
        (grad,) = torch.autograd.grad((eps * loss(x_adv)).sum(), x_adv)
        grad2 = x_adv - inputs
        x_adv = (x_adv + (grad - grad2) / math.sqrt(t + 2)).detach()
    return x_adv


def test_squared_distance_sums_every_dimension_but_the_first() -> None:
    points = torch.arange(12.0).reshape(2, 3, 2)
    inputs = torch.zeros(2, 3, 2)

    assert squared_distance(points, inputs).tolist() == [55.0, 451.0]


@pytest.mark.parametrize("penalty", [0.1, 1.0, 7.5])
def test_a_linear_loss_is_solved_exactly_by_the_first_step(penalty: float) -> None:
    slope = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, 1.0]], dtype=torch.float64)
    inputs = torch.randn(2, 3, dtype=torch.float64)

    points = transport_map(lambda z: (z * slope).sum(1), inputs, penalty, num_steps=5)

    torch.testing.assert_close(points, inputs + slope / (2 * penalty))


def test_a_quadratic_loss_converges_to_its_maximizer() -> None:
    generator = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
    curvature = basis[0] @ torch.diag(torch.tensor([1.5, 0.5, -1.0, 0.0]).double())
    curvature = curvature @ basis[0].T
    inputs = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    penalty = 2.0

    points = transport_map(
        lambda z: 0.5 * ((z @ curvature) * z).sum(1), inputs, penalty, num_steps=400
    )

    system = 2 * penalty * torch.eye(4, dtype=torch.float64) - curvature
    exact = torch.linalg.solve(system, 2 * penalty * inputs.T).T
    torch.testing.assert_close(points, exact, atol=1e-8, rtol=0)


@pytest.mark.parametrize("num_steps", [1, 2, 16])
def test_matches_the_authors_wrm_loop(num_steps: int) -> None:
    network = _network()
    inputs = torch.randn(5, 3, dtype=torch.float64)
    loss = _loss(network, torch.tensor([0, 1, 1, 0, 1]))

    ours = transport_map(loss, inputs, penalty=1 / 2.6, num_steps=num_steps)

    torch.testing.assert_close(ours, _authors_wrm(loss, inputs, 1.3, num_steps - 1))


def test_reaches_the_maximizer_found_by_optora_gradient_descent() -> None:
    network = _network(seed=1)
    inputs = torch.randn(4, 3, dtype=torch.float64)
    loss = _loss(network, torch.tensor([1, 0, 1, 1]))
    penalty = 3.0

    def objective(points: torch.Tensor) -> torch.Tensor:
        return -(loss(points) - penalty * squared_distance(points, inputs)).sum()

    solver = GradientDescent(step_size=0.05, max_iter=20_000, tol=1e-12)
    oracle = solver.solve(MinimizationProblem(objective, inputs.clone()))

    assert oracle.converged
    torch.testing.assert_close(
        transport_map(loss, inputs, penalty, num_steps=200),
        oracle.point,
        atol=1e-9,
        rtol=0,
    )
    torch.testing.assert_close(
        transport_map(loss, inputs, penalty, num_steps=16),
        oracle.point,
        atol=1e-3,
        rtol=0,
    )


def test_a_penalty_per_example_equals_separate_solves() -> None:
    network = _network(seed=2)
    inputs = torch.randn(3, 3, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0])
    penalties = torch.tensor([0.5, 2.0, 10.0], dtype=torch.float64)

    joint = transport_map(_loss(network, labels), inputs, penalties)

    for index in range(3):
        alone = transport_map(
            _loss(network, labels[index : index + 1]),
            inputs[index : index + 1],
            float(penalties[index]),
        )
        torch.testing.assert_close(joint[index : index + 1], alone)


def test_images_keep_their_shape_and_dtype() -> None:
    torch.manual_seed(0)
    network = nn.Sequential(
        nn.Conv2d(1, 2, 3), nn.ELU(), nn.Flatten(), nn.LazyLinear(3)
    )
    inputs = torch.rand(4, 1, 6, 6)
    loss = _loss(network, torch.tensor([0, 1, 2, 0]))

    points = transport_map(loss, inputs, 0.5)

    assert points.shape == inputs.shape and points.dtype == torch.float32
    assert not points.requires_grad


def test_leaves_the_parameters_without_gradients_and_works_under_no_grad() -> None:
    network = _network()
    inputs = torch.randn(2, 3, dtype=torch.float64)
    loss = _loss(network, torch.tensor([0, 1]))

    with torch.no_grad():
        points = transport_map(loss, inputs, 1.0)

    assert all(parameter.grad is None for parameter in network.parameters())
    assert not torch.equal(points, inputs)


@pytest.mark.parametrize(
    ("penalty", "num_steps", "message"),
    [
        (0.0, 16, "penalty must be positive"),
        (-1.0, 16, "penalty"),
        (1.0, 0, "num_steps"),
    ],
)
def test_rejects_invalid_arguments(
    penalty: float, num_steps: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        transport_map(lambda z: z.sum(1), torch.zeros(1, 2), penalty, num_steps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda_like_on_the_cpu() -> None:
    network = _network()
    inputs = torch.randn(5, 3, dtype=torch.float64)
    labels = torch.tensor([0, 1, 1, 0, 1])

    cpu = transport_map(_loss(network, labels), inputs, 0.7)
    network.cuda()
    cuda = transport_map(_loss(network, labels.cuda()), inputs.cuda(), 0.7)

    assert cuda.is_cuda
    torch.testing.assert_close(cuda.cpu(), cpu)
