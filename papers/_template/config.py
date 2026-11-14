"""Settings of the experiment, loaded from `configs/*.toml`."""

from __future__ import annotations

from dataclasses import dataclass

_DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class Config:
    """Settings of the template experiment.

    Attributes:
        seeds: Seeds of the independent runs, one problem instance each.
        num_samples: Support size of the empirical nominal distribution.
        radius_fraction: Radius as a fraction of the largest radius for
            which the identity holds, in `(0, 1]`.
        dtype: Floating-point type, `"float32"` or `"float64"`.
        device: PyTorch device, for example `"cpu"` or `"cuda"`.
    """

    seeds: tuple[int, ...]
    num_samples: int
    radius_fraction: float
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.num_samples < 2:
            raise ValueError(f"num_samples must be at least 2, got {self.num_samples}")
        if not 0.0 < self.radius_fraction <= 1.0:
            raise ValueError(
                f"radius_fraction must be in (0, 1], got {self.radius_fraction}"
            )
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")
