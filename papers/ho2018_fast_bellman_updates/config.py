"""Settings of the experiment, loaded from `configs/*.toml`."""

from __future__ import annotations

from dataclasses import dataclass

_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class Config:
    r"""Settings of the Section 5.1 experiment.

    Attributes:
        seeds: Seeds of the runs; the paper averages over five runs.
        state_counts: Numbers of states $S$ of the grid.
        budgets: Ambiguity budgets $\xi$ of the $L_1$ ball.
        discount: Discount factor of the inventory domain.
        timing_states: Number of states of the timed random instance (c2).
        timing_repeats: Timed calls per budget; their median is used.
        dtype: Floating-point type, `"float32"` or `"float64"`.
        device: PyTorch device of the exactness check. Timing runs on CPU.
    """

    seeds: tuple[int, ...]
    state_counts: tuple[int, ...]
    budgets: tuple[float, ...]
    discount: float = 0.9
    timing_states: int = 400
    timing_repeats: int = 20
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.state_counts or min(self.state_counts) < 2:
            raise ValueError("state_counts must be nonempty and at least 2")
        if not self.budgets or min(self.budgets) < 0.0 or max(self.budgets) > 2.0:
            raise ValueError("budgets must be nonempty and in [0, 2]")
        if not 0.0 <= self.discount < 1.0:
            raise ValueError(f"discount must be in [0, 1), got {self.discount}")
        if self.timing_states < 2 or self.timing_repeats < 1:
            raise ValueError("timing_states must be at least 2, timing_repeats >= 1")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
