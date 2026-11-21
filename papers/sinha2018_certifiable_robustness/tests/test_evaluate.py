from __future__ import annotations

import pytest
import torch
from torch import nn

from prototypes.continuous_wasserstein import squared_distance, transport_map

from ..evaluate import WorstCase, accuracy, loss_at, pgm_error, worst_case_curve
from ..model import cross_entropy, one_hot


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 6), nn.ELU(), nn.Linear(6, 4))


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(11, 3, generator=generator)
    return inputs, torch.randint(0, 4, (11,), generator=generator)


def test_accuracy_counts_every_batch() -> None:
    model = _model()
    inputs, labels = _data()
    predictions = model(inputs).argmax(1)

    expected = float((predictions == labels).float().mean())
    assert accuracy(model, inputs, labels, batch_size=4) == pytest.approx(expected)


def test_pgm_error_at_radius_zero_is_the_clean_error() -> None:
    model = _model()
    inputs, labels = _data()

    error = pgm_error(model, inputs, one_hot(labels, 4), 0.0, 2.0, batch_size=5)

    assert error == pytest.approx(1 - accuracy(model, inputs, labels, 5))


def test_pgm_error_grows_with_the_radius() -> None:
    model = _model()
    inputs, labels = _data()
    targets = one_hot(labels, 4)

    errors = [
        pgm_error(model, inputs, targets, radius, float("inf"), batch_size=3)
        for radius in (0.0, 0.5, 5.0)
    ]

    assert errors == sorted(errors) and errors[-1] == 1.0


def test_worst_case_is_the_mean_cost_and_loss_at_the_transport_map() -> None:
    model = _model()
    inputs, labels = _data()
    targets = one_hot(labels, 4)

    (point,) = worst_case_curve(model, inputs, targets, [0.8], 16, batch_size=4)

    def loss(z: torch.Tensor) -> torch.Tensor:
        return cross_entropy(model(z), targets)

    moved = transport_map(loss, inputs, 0.8, 16)
    assert point.penalty == 0.8
    assert point.distance == pytest.approx(
        float(squared_distance(moved, inputs).mean())
    )
    assert point.loss == pytest.approx(float(loss(moved).mean()), rel=1e-5)


def test_a_smaller_penalty_moves_further_and_raises_the_loss() -> None:
    model = _model()
    inputs, labels = _data()

    curve = worst_case_curve(
        model, inputs, one_hot(labels, 4), [10.0, 1.0, 0.3], 16, batch_size=11
    )

    assert [p.distance for p in curve] == sorted(p.distance for p in curve)
    assert [p.loss for p in curve] == sorted(p.loss for p in curve)


def test_loss_at_interpolates_between_the_levels_of_a_curve() -> None:
    curve = [
        WorstCase(2.0, 0.1, 1.0),
        WorstCase(0.5, 0.5, 3.0),
        WorstCase(1.0, 0.3, 2.0),
    ]

    assert loss_at(curve, 0.2) == pytest.approx(1.5)
    assert loss_at(curve, 0.5) == pytest.approx(3.0)


@pytest.mark.parametrize("distance", [0.05, 0.6])
def test_loss_at_rejects_a_level_outside_the_curve(distance: float) -> None:
    curve = [WorstCase(2.0, 0.1, 1.0), WorstCase(0.5, 0.5, 3.0)]
    with pytest.raises(ValueError, match="lies outside"):
        loss_at(curve, distance)
