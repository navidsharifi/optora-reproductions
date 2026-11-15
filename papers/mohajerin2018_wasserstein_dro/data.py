r"""Asset returns of Section 7.2.

The return of asset $i \in \{1, \dots, m\}$ is $\xi_i = \psi + \zeta_i$ with a
systematic factor $\psi \sim N(0, 0.02^2)$ and independent idiosyncratic
factors $\zeta_i \sim N(0.03\, i, (0.025\, i)^2)$. The paper writes the
moments as $N(0, 2\%)$ and $N(i \times 3\%, i \times 2.5\%)$, and its
out-of-sample values (Fig. 5) fit only the reading of the second argument as
a standard deviation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

NUM_ASSETS = 10
SYSTEMATIC_DEVIATION = 0.02
MEAN_STEP = 0.03
DEVIATION_STEP = 0.025


def return_moments(
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Mean and covariance of the return vector $\xi$.

    Args:
        dtype: Floating-point type.

    Returns:
        Mean of shape `(m,)` and covariance
        $0.02^2\, \mathbf 1 \mathbf 1^\top + \mathrm{diag}((0.025\, i)^2)$ of
        shape `(m, m)`, both on CPU.
    """
    index = torch.arange(1, NUM_ASSETS + 1, dtype=dtype)
    covariance = torch.full(
        (NUM_ASSETS, NUM_ASSETS), SYSTEMATIC_DEVIATION**2, dtype=dtype
    )
    covariance += torch.diag((DEVIATION_STEP * index) ** 2)
    return MEAN_STEP * index, covariance


def sample_returns(
    seeds: Sequence[int], num_samples: int, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Draw one training dataset per simulation.

    Each simulation has its own generator, seeded from the simulation seed
    and the sample size, so datasets of different sizes are independent.

    Args:
        seeds: Seeds of the simulations.
        num_samples: Sample size $N$ of every dataset.
        dtype: Floating-point type.

    Returns:
        Tensor of shape `(len(seeds), num_samples, m)` on CPU.
    """
    index = torch.arange(1, NUM_ASSETS + 1, dtype=dtype)
    datasets = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed * 100_003 + num_samples)
        noise = torch.randn(
            num_samples, NUM_ASSETS + 1, generator=generator, dtype=dtype
        )
        systematic = SYSTEMATIC_DEVIATION * noise[:, :1]
        idiosyncratic = MEAN_STEP * index + DEVIATION_STEP * index * noise[:, 1:]
        datasets.append(systematic + idiosyncratic)
    return torch.stack(datasets)
