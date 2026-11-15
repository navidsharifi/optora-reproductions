from __future__ import annotations

import pytest
import torch

from ..data import inventory_instances, random_instance
from ..model import lp_q, median_seconds, optora_q, optora_single_q

F64 = torch.float64
BUDGETS = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.5, 2.0], dtype=F64)


@pytest.mark.parametrize("num_states", [2, 7, 60])
def test_optora_matches_the_linear_program_on_random_instances(num_states: int) -> None:
    nominal, z = random_instance(1, num_states)

    values = optora_q(nominal.unsqueeze(0), z.unsqueeze(0), BUDGETS)[0]

    reference = [lp_q(nominal.numpy(), z.numpy(), float(b)) for b in BUDGETS]
    assert torch.allclose(values, torch.tensor(reference, dtype=F64), atol=1e-12)


def test_optora_matches_the_linear_program_with_zero_nominal_mass() -> None:
    nominal, z = inventory_instances(15, 0.9)

    values = optora_q(nominal, z, BUDGETS)

    reference = torch.tensor(
        [
            [lp_q(p.numpy(), v.numpy(), float(b)) for b in BUDGETS]
            for p, v in zip(nominal, z, strict=True)
        ],
        dtype=F64,
    )
    assert torch.allclose(values, reference, atol=1e-12)


def test_zero_budget_gives_the_nominal_expectation() -> None:
    nominal, z = random_instance(2, 30)
    value = optora_q(nominal.unsqueeze(0), z.unsqueeze(0), BUDGETS[:1])
    assert value.item() == pytest.approx(float(nominal @ z), abs=1e-14)


def test_full_budget_reaches_the_smallest_value() -> None:
    nominal, z = random_instance(2, 30)
    value = optora_q(nominal.unsqueeze(0), z.unsqueeze(0), BUDGETS[-1:])
    assert value.item() == pytest.approx(z.min().item(), abs=1e-14)


def test_value_is_nonincreasing_in_the_budget() -> None:
    nominal, z = random_instance(5, 40)
    values = optora_q(nominal.unsqueeze(0), z.unsqueeze(0), BUDGETS)[0]
    assert torch.all(values[1:] <= values[:-1] + 1e-15)


def test_single_call_equals_the_batched_call() -> None:
    nominal, z = random_instance(6, 25)
    batched = optora_q(nominal.unsqueeze(0), z.unsqueeze(0), BUDGETS)[0]
    single = [optora_single_q(nominal, z, float(b)) for b in BUDGETS]
    assert batched.tolist() == pytest.approx(single, abs=1e-15)


def test_batched_call_returns_one_value_per_instance_and_budget() -> None:
    nominal, z = inventory_instances(12, 0.9)
    assert optora_q(nominal, z, BUDGETS).shape == (12, BUDGETS.numel())


def test_median_seconds_calls_once_more_than_repeats() -> None:
    calls: list[int] = []
    seconds = median_seconds(lambda: calls.append(1), 5)
    assert len(calls) == 6
    assert seconds >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu() -> None:
    nominal, z = inventory_instances(20, 0.9)
    cpu = optora_q(nominal, z, BUDGETS)
    cuda = optora_q(nominal.cuda(), z.cuda(), BUDGETS.cuda()).cpu()
    assert torch.allclose(cuda, cpu, atol=1e-14)
