"""Settings of the simulation, loaded from `configs/*.toml`."""

from __future__ import annotations

from dataclasses import dataclass

_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class Config:
    r"""Settings of the Section 5.2 simulation.

    Attributes:
        seeds: Seeds of the independent runs. Each seed draws its own
            replications for every cell of the grid.
        replications_per_seed: Replications per seed and grid cell.
        sample_sizes: Sample sizes $n$ of the grid.
        noise_levels: Noise levels $B$ of the grid.
        dimension: Dimension $d$ of the parameter.
        domain_radius: Radius $r$ of the Euclidean ball $\Theta$.
        delta: Confidence level $\delta$ in the radius formula (Eq. 17).
        descent_iterations: Iteration cap of the descent phase (the
            projected gradient method of Section 5.1).
        subgradient_iterations: Steps of the averaged subgradient phase
            that follows it; 0 runs the descent phase alone.
        chunk_megabytes: Memory budget for the features of one batch of
            replications; larger cells are split into several batches.
        dtype: Floating-point type, `"float32"` or `"float64"`.
        device: PyTorch device, for example `"cpu"` or `"cuda"`.
    """

    seeds: tuple[int, ...]
    replications_per_seed: int
    sample_sizes: tuple[int, ...]
    noise_levels: tuple[float, ...]
    dimension: int = 50
    domain_radius: float = 10.0
    delta: float = 0.05
    descent_iterations: int = 500
    subgradient_iterations: int = 5000
    chunk_megabytes: int = 512
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.replications_per_seed < 1:
            raise ValueError("replications_per_seed must be positive")
        if not self.sample_sizes or min(self.sample_sizes) < 2:
            raise ValueError("sample_sizes must be nonempty and at least 2")
        if not self.noise_levels or min(self.noise_levels) <= 0.0:
            raise ValueError("noise_levels must be nonempty and positive")
        if self.dimension < 1 or self.domain_radius <= 0.0:
            raise ValueError("dimension and domain_radius must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}")
        if self.descent_iterations < 1 or self.chunk_megabytes < 1:
            raise ValueError("descent_iterations and chunk_megabytes must be positive")
        if self.subgradient_iterations < 0:
            raise ValueError("subgradient_iterations must be nonnegative")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
