"""Fixtures shared by `tests/` and every paper's smoke tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch

_CUBLAS_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"


@pytest.fixture(autouse=True)
def _restore_global_torch_state() -> Iterator[None]:
    """Undo the global effects of `seed_everything` after each test."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cublas = os.environ.get(_CUBLAS_VARIABLE)
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if cublas is None:
        os.environ.pop(_CUBLAS_VARIABLE, None)
    else:
        os.environ[_CUBLAS_VARIABLE] = cublas
