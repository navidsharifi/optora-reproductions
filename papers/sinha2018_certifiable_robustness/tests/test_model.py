from __future__ import annotations

import pytest
import torch
from torch import nn

from prototypes.continuous_wasserstein import transport_map

from ..data import random_images, two_dimensional
from ..model import (
    ERM,
    METHODS,
    PENALTY,
    WRM,
    WRM_STEPS,
    adversary,
    batch_starts,
    cross_entropy,
    mnist_network,
    one_hot,
    smoothed,
    synthetic_network,
    train_mnist,
    train_synthetic,
)


def test_mnist_network_has_the_authors_shapes() -> None:
    network = mnist_network()
    outputs = network(torch.rand(3, 1, 28, 28))

    assert outputs.shape == (3, 10)
    assert sum(parameter.numel() for parameter in network.parameters()) == 710_218
    assert [
        layer.out_channels for layer in network if isinstance(layer, nn.Conv2d)
    ] == [
        64,
        128,
        128,
    ]


def test_networks_start_with_zero_biases() -> None:
    for network in (mnist_network(), synthetic_network()):
        biases = [m.bias for m in network if isinstance(m, nn.Conv2d | nn.Linear)]
        assert all(bias is not None and not bool(bias.any()) for bias in biases)


def test_synthetic_network_has_hidden_layers_of_four_and_two() -> None:
    network = synthetic_network()
    widths = [layer.out_features for layer in network if isinstance(layer, nn.Linear)]
    assert widths == [4, 2, 2]


def test_smoothed_targets_follow_the_authors_clipping() -> None:
    targets = smoothed(torch.tensor([3, 0]))

    torch.testing.assert_close(targets.sum(1), torch.ones(2))
    assert float(targets[0, 3]) == pytest.approx(0.9)
    assert float(targets[0, 0]) == pytest.approx(0.1 / 9)


def test_cross_entropy_with_one_hot_targets_is_the_usual_loss() -> None:
    logits = torch.randn(5, 10)
    labels = torch.tensor([0, 3, 9, 3, 1])

    torch.testing.assert_close(
        cross_entropy(logits, one_hot(labels)),
        nn.functional.cross_entropy(logits, labels, reduction="none"),
    )


def test_batch_starts_shift_the_last_batch_back() -> None:
    assert batch_starts(10, 4) == [0, 4, 6]
    assert batch_starts(8, 4) == [0, 4]
    assert batch_starts(60_000, 256)[-1] == 60_000 - 256
    assert len(batch_starts(60_000, 256)) == 235


@pytest.mark.parametrize("batch_size", [0, 11])
def test_batch_starts_reject_a_batch_outside_the_data(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be in"):
        batch_starts(10, batch_size)


def test_erm_keeps_the_inputs_and_wrm_uses_the_transport_map() -> None:
    torch.manual_seed(0)
    network = mnist_network()
    inputs = torch.rand(2, 1, 28, 28)
    targets = smoothed(torch.tensor([1, 2]))

    assert adversary(ERM, network, 0.0)(inputs, targets) is inputs
    expected = transport_map(
        lambda points: cross_entropy(network(points), targets),
        inputs,
        PENALTY,
        WRM_STEPS,
    )
    torch.testing.assert_close(adversary(WRM, network, 0.0)(inputs, targets), expected)


def test_adversary_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        adversary("trades", mnist_network(), 0.1)


@pytest.mark.parametrize("method", METHODS)
def test_training_is_reproducible_and_moves_the_weights(method: str) -> None:
    train = random_images(train_size=24, test_size=4).train
    order = torch.randperm(24, generator=torch.Generator().manual_seed(0))

    def trained() -> nn.Module:
        torch.manual_seed(0)
        network = mnist_network()
        seconds = train_mnist(
            network, method, train, order, epochs=1, batch_size=8, radius=0.5
        )
        assert seconds > 0
        return network

    first, second = trained(), trained()
    torch.manual_seed(0)
    start = mnist_network()
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert not torch.equal(next(first.parameters()), next(start.parameters()))


def test_synthetic_training_lowers_the_robust_loss() -> None:
    generator = torch.Generator().manual_seed(0)
    train = two_dimensional(400, generator)
    torch.manual_seed(0)
    network = synthetic_network()
    targets = one_hot(train.labels, 2)

    def robust_loss() -> float:
        points = transport_map(
            lambda z: cross_entropy(network(z), targets), train.inputs, 2.0, 15
        )
        return float(cross_entropy(network(points), targets).mean())

    before = robust_loss()
    train_synthetic(network, train, generator, epochs=5, batch_size=50)

    assert robust_loss() < before
