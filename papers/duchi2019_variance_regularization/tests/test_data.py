from __future__ import annotations

import torch

from ..data import sample_signs


def test_signs_have_the_requested_shape_and_values() -> None:
    signs = sample_signs(0, 0, 3, 7, 5)

    assert signs.shape == (3, 7, 5)
    assert signs.dtype == torch.int8
    assert bool(((signs == 1) | (signs == -1)).all())
    assert bool((signs == 1).any()) and bool((signs == -1).any())


def test_signs_are_deterministic_per_seed_and_cell() -> None:
    assert torch.equal(sample_signs(4, 2, 3, 20, 5), sample_signs(4, 2, 3, 20, 5))


def test_cells_and_seeds_draw_different_signs() -> None:
    base = sample_signs(0, 0, 2, 50, 10)
    assert not torch.equal(base, sample_signs(0, 1, 2, 50, 10))
    assert not torch.equal(base, sample_signs(1, 0, 2, 50, 10))


def test_signs_are_balanced() -> None:
    mean = sample_signs(0, 0, 20, 500, 50).double().mean().item()
    assert abs(mean) < 0.01
