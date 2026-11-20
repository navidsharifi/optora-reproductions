"""Settings of the experiment, loaded from `configs/*.toml`.

The penalties, step counts, learning rates and label smoothing of the
methods are part of the paper's method and live in `model.py` and
`attacks.py`. A configuration chooses the data, the seeds, the training
budget, the resolution of the evaluated curves and the device.
"""

from __future__ import annotations

from dataclasses import dataclass

_DATA = ("mnist", "random")


@dataclass(frozen=True)
class Config:
    r"""Settings of the MNIST and synthetic experiments.

    Attributes:
        data: `"mnist"` for the paper's dataset or `"random"` for the random
            images of the smoke run.
        seeds: Seeds of the independent runs. Each seed trains the five
            MNIST models and the synthetic model.
        epochs: Epochs of every MNIST model.
        batch_size: Examples per MNIST training step.
        synthetic_size: Training and test examples of the synthetic data,
            each.
        synthetic_epochs: Epochs of the synthetic model.
        synthetic_batch_size: Examples per synthetic training step.
        num_penalties: Penalties of each worst-case curve.
        num_l2_radii: Radii of each $\ell_2$ error curve.
        num_linf_radii: Radii of each $\ell_\infty$ error curve.
        eval_batch_size: Examples per evaluated batch.
        device: PyTorch device, for example `"cpu"` or `"cuda"`.
    """

    data: str
    seeds: tuple[int, ...]
    epochs: int = 25
    batch_size: int = 256
    synthetic_size: int = 10_000
    synthetic_epochs: int = 25
    synthetic_batch_size: int = 256
    num_penalties: int = 24
    num_l2_radii: int = 14
    num_linf_radii: int = 11
    eval_batch_size: int = 500
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.data not in _DATA:
            raise ValueError(f"data must be one of {_DATA}, got {self.data!r}")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) < len(self.seeds):
            raise ValueError("seeds must not repeat")
        for name in (
            "epochs",
            "batch_size",
            "synthetic_size",
            "synthetic_epochs",
            "synthetic_batch_size",
            "eval_batch_size",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        for name in ("num_penalties", "num_l2_radii", "num_linf_radii"):
            value = getattr(self, name)
            if value < 2:
                raise ValueError(f"{name} must be at least 2, got {value}")
