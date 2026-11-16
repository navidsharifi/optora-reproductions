"""Settings of the experiment, loaded from `configs/*.toml`."""

from __future__ import annotations

from dataclasses import dataclass

_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class Config:
    r"""Settings of the Section 7.2 experiment.

    Attributes:
        seeds: Seeds of the simulations, one training dataset each.
        certificate_samples: Sample size $N$ of claims c1 and c2.
        certificate_radius: Radius $\varepsilon$ of claims c1 and c2.
        sweep_samples: Sample size $N$ of the radius sweep of claim c3.
        radii: Radii of the sweep. Must contain `small_radius`.
        small_radius: Smallest radius of claim c3.
        far: Distance $D$ of the support points $\pm D e_j$.
        iterations: Ellipsoid steps per problem.
        dtype: Floating-point type, `"float32"` or `"float64"`.
        device: PyTorch device, for example `"cpu"` or `"cuda"`.
    """

    seeds: tuple[int, ...]
    radii: tuple[float, ...]
    certificate_samples: int = 300
    certificate_radius: float = 1.0
    sweep_samples: int = 30
    small_radius: float = 1e-4
    far: float = 1e6
    iterations: int = 3000
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.radii or min(self.radii) <= 0.0:
            raise ValueError("radii must be nonempty and positive")
        if self.small_radius not in self.radii:
            raise ValueError(f"radii must contain small_radius {self.small_radius}")
        if min(self.certificate_samples, self.sweep_samples) < 1:
            raise ValueError("sample sizes must be positive")
        if self.certificate_radius <= 0.0:
            raise ValueError("certificate_radius must be positive")
        if self.far <= 0.0 or self.iterations < 1:
            raise ValueError("far must be positive and iterations at least 1")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
