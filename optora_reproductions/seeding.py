"""Seeding and deterministic execution for reproducible runs."""

from __future__ import annotations

import os
import random

import torch

_MAX_SEED = 2**63 - 1


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python and PyTorch, and optionally request deterministic kernels.

    With `deterministic=True` the function calls
    `torch.use_deterministic_algorithms(True, warn_only=True)`, so an operation
    without a deterministic implementation warns instead of failing, and turns
    off cuDNN autotuning. It also sets `CUBLAS_WORKSPACE_CONFIG` when that
    variable is unset. cuBLAS reads it only once, so it takes effect only if
    this function runs before the first CUDA matrix product of the process.

    Args:
        seed: Seed in `[0, 2**63 - 1]`.
        deterministic: Whether to request deterministic kernels.

    Returns:
        A CPU `torch.Generator` seeded with `seed`. Sample data with it, then
        move the tensors to the target device, so the draws do not depend on
        the device or on other uses of the global generator.

    Raises:
        ValueError: If `seed` is outside `[0, 2**63 - 1]`.
    """
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, 2**63 - 1], got {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
