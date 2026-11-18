"""Settings of the experiment, loaded from `configs/*.toml`.

The step sizes are part of the paper's method, not of a run, so they live in
`model.STEP_SIZES` with the authors' values. A configuration chooses the
data, the seeds, the budgets and the device.
"""

from __future__ import annotations

from dataclasses import dataclass

_DATA = ("digits", "synthetic")
_PAPER_BATCH_SIZES = (50, 500, 5000)


@dataclass(frozen=True)
class Config:
    """Settings of the Digits experiment.

    Attributes:
        data: `"digits"` for the authors' features or `"synthetic"` for the
            small stand-in of the smoke run.
        seeds: Seeds of the minibatch and ERM runs.
        batch_sizes: Minibatch sizes, a subset of the paper's 50, 500 and
            5000.
        epochs: Passes over the training set of every minibatch and ERM run.
        full_batch_epochs: Steps of the full-batch run.
        full_batch_eval_interval: Epochs between evaluations of the
            full-batch run.
        full_batch_seed: Seed of the single full-batch run per objective.
        erm_batch_size: Minibatch size of the ERM runs.
        comparison_batch_size: Minibatch size whose DRO runs claim c4
            compares with ERM. It must be one of `batch_sizes`.
        typed_train: Typed digits in the training set (`"digits"` only).
        device: PyTorch device, for example `"cpu"` or `"cuda"`.
        cuda_graphs: Whether to replay the forward and backward pass of every
            step from a CUDA graph. It needs a CUDA device.
    """

    data: str
    seeds: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    epochs: int
    full_batch_epochs: int
    full_batch_eval_interval: int
    full_batch_seed: int = 1
    erm_batch_size: int = 500
    comparison_batch_size: int = 500
    typed_train: int = 600
    device: str = "cpu"
    cuda_graphs: bool = False

    def __post_init__(self) -> None:
        if self.data not in _DATA:
            raise ValueError(f"data must be one of {_DATA}, got {self.data!r}")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.batch_sizes or not set(self.batch_sizes) <= set(_PAPER_BATCH_SIZES):
            raise ValueError(
                f"batch_sizes must be a nonempty subset of {_PAPER_BATCH_SIZES}, "
                f"got {self.batch_sizes}"
            )
        if self.comparison_batch_size not in self.batch_sizes:
            raise ValueError(
                f"comparison_batch_size {self.comparison_batch_size} is not in "
                f"batch_sizes {self.batch_sizes}"
            )
        for name in ("epochs", "full_batch_epochs", "full_batch_eval_interval"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.full_batch_epochs % self.full_batch_eval_interval:
            raise ValueError(
                "full_batch_epochs must be a multiple of full_batch_eval_interval"
            )
        if self.erm_batch_size < 1 or self.typed_train < 1:
            raise ValueError("erm_batch_size and typed_train must be positive")
        if self.cuda_graphs and not self.device.startswith("cuda"):
            raise ValueError("cuda_graphs needs a CUDA device")
