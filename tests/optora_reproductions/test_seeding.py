from __future__ import annotations

import os
import random

import pytest
import torch

from optora_reproductions.seeding import seed_everything


def draws() -> tuple[float, torch.Tensor, torch.Tensor]:
    generator = seed_everything(7)
    return random.random(), torch.rand(3), torch.rand(3, generator=generator)


def test_same_seed_gives_same_draws() -> None:
    first, second = draws(), draws()

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_different_seeds_give_different_draws() -> None:
    assert not torch.equal(
        torch.rand(4, generator=seed_everything(0)),
        torch.rand(4, generator=seed_everything(1)),
    )


def test_returns_cpu_generator_with_the_seed() -> None:
    generator = seed_everything(123)

    assert generator.device == torch.device("cpu")
    assert generator.initial_seed() == 123


def test_returned_generator_is_independent_of_global_state() -> None:
    generator = seed_everything(5)
    torch.rand(100)
    after_global_use = torch.rand(3, generator=generator)

    assert torch.equal(after_global_use, torch.rand(3, generator=seed_everything(5)))


def test_deterministic_mode_sets_flags() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    seed_everything(0)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.benchmark


def test_non_deterministic_mode_leaves_flags_unchanged() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    seed_everything(0, deterministic=False)

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark


def test_sets_cublas_workspace_only_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seed_everything(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_accepts_boundary_seeds(seed: int) -> None:
    assert seed_everything(seed).initial_seed() == seed


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_rejects_out_of_range_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(seed)
