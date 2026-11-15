r"""Data of the simulation, $X \sim \mathrm{Uni}(\{-B, B\}^d)$."""

from __future__ import annotations

import torch

_CELL_STRIDE = 1_000_003


def sample_signs(
    seed: int, cell: int, replications: int, num_samples: int, dimension: int
) -> torch.Tensor:
    """Draw the Rademacher signs of every replication of one grid cell.

    The features are `B` times these signs. Each `(seed, cell)` pair has its
    own generator, so a cell's draws do not depend on the grid around it or
    on how replications are later split into batches.

    Args:
        seed: Seed of the run.
        cell: Index of the grid cell.
        replications: Number of replications.
        num_samples: Sample size $n$.
        dimension: Dimension $d$.

    Returns:
        CPU tensor of shape `(replications, num_samples, dimension)` with
        entries in `{-1, 1}` and dtype `int8`.
    """
    generator = torch.Generator().manual_seed(seed * _CELL_STRIDE + cell)
    bits = torch.randint(
        0,
        2,
        (replications, num_samples, dimension),
        generator=generator,
        dtype=torch.int8,
    )
    return bits * 2 - 1
