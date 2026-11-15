from __future__ import annotations

import pytest
import torch

from ..data import (
    HOLDING_COST,
    PURCHASE_COST,
    SALE_PRICE,
    demand_distribution,
    inventory_instances,
    random_instance,
)

F64 = torch.float64


def test_random_instance_is_a_distribution_and_values_in_unit_interval() -> None:
    nominal, z = random_instance(0, 50)

    assert nominal.shape == z.shape == (50,)
    assert torch.all(nominal > 0)
    assert nominal.sum().item() == pytest.approx(1.0, abs=1e-14)
    assert torch.all((z >= 0) & (z < 1))


def test_random_instance_is_deterministic_and_varies_with_seed_and_size() -> None:
    first = random_instance(3, 40)
    assert torch.equal(first[0], random_instance(3, 40)[0])
    assert not torch.equal(first[0], random_instance(4, 40)[0])
    assert not torch.equal(first[1][:30], random_instance(3, 30)[1])


def test_demand_is_a_distribution_centred_on_half_the_states() -> None:
    demand = demand_distribution(200, F64)
    mean = (demand * torch.arange(200, dtype=F64)).sum().item()

    assert demand.sum().item() == pytest.approx(1.0, abs=1e-14)
    assert torch.all(demand >= 0)
    assert mean == pytest.approx(100.0, abs=1.0)


def test_inventory_rows_are_distributions_supported_up_to_the_order() -> None:
    nominal, z = inventory_instances(30, 0.9)

    assert nominal.shape == z.shape == (30, 30)
    assert torch.allclose(nominal.sum(-1), torch.ones(30, dtype=F64), atol=1e-14)
    for order in range(30):
        assert torch.all(nominal[order, order + 1 :] == 0)


def test_inventory_order_zero_stays_empty_and_earns_nothing() -> None:
    nominal, z = inventory_instances(20, 0.9)

    assert nominal[0, 0].item() == pytest.approx(1.0)
    assert z[0, 0].item() == pytest.approx(0.0, abs=1e-14)


def test_inventory_reward_follows_the_cost_model() -> None:
    num_states, order = 10, 4
    nominal, z = inventory_instances(num_states, 0.5)
    demand = demand_distribution(num_states, F64)
    levels = torch.arange(num_states, dtype=F64)

    reward = (
        SALE_PRICE * (demand * torch.clamp(levels, max=order)).sum()
        - PURCHASE_COST * order
        - HOLDING_COST * (demand * torch.clamp(order - levels, min=0)).sum()
    )

    assert torch.allclose(z[order], reward + 0.5 * levels)
