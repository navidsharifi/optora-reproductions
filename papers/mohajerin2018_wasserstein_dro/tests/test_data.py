from __future__ import annotations

import torch

from ..data import NUM_ASSETS, return_moments, sample_returns

F64 = torch.float64


def test_moments_follow_the_factor_model() -> None:
    mean, covariance = return_moments()
    index = torch.arange(1, NUM_ASSETS + 1, dtype=F64)

    assert torch.allclose(mean, 0.03 * index)
    assert torch.allclose(covariance.diagonal(), 0.02**2 + (0.025 * index) ** 2)
    off_diagonal = covariance[~torch.eye(NUM_ASSETS, dtype=torch.bool)]
    assert torch.allclose(off_diagonal, torch.full_like(off_diagonal, 0.02**2))
    assert torch.linalg.eigvalsh(covariance).min() > 0


def test_samples_have_one_dataset_per_seed() -> None:
    samples = sample_returns([0, 1, 2], 7)
    assert samples.shape == (3, 7, NUM_ASSETS)
    assert samples.dtype == F64


def test_samples_are_deterministic_and_independent_across_seeds_and_sizes() -> None:
    first = sample_returns([4], 30)
    assert torch.equal(first, sample_returns([4], 30))
    assert not torch.equal(first, sample_returns([5], 30))
    assert not torch.equal(first[:, :10], sample_returns([4], 10))


def test_seed_order_does_not_change_a_dataset() -> None:
    forward = sample_returns([0, 1], 5)
    backward = sample_returns([1, 0], 5)
    assert torch.equal(forward[0], backward[1])


def test_sample_moments_match_the_model() -> None:
    samples = sample_returns([0], 200_000)[0]
    mean, covariance = return_moments()

    assert torch.allclose(samples.mean(dim=0), mean, atol=2e-3)
    assert torch.allclose(torch.cov(samples.T), covariance, atol=2e-3)


def test_float32_samples() -> None:
    assert sample_returns([0], 4, torch.float32).dtype == torch.float32
