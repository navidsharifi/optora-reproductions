"""Settings of the experiment, loaded from `configs/*.toml`.

The learning rates, penalties and epochs of each regularization regime are
part of the paper's method, so they live in `model.REGIMES` with the values
of Appendix C.2. A configuration chooses the data, the seeds, the budget of
the 300-epoch runs, the precision and the device.
"""

from __future__ import annotations

from dataclasses import dataclass

_DATA = ("waterbirds", "synthetic")
_PRECISIONS = ("bfloat16", "float32")


@dataclass(frozen=True)
class Config:
    """Settings of the Waterbirds experiment.

    Attributes:
        data: `"waterbirds"` for the paper's dataset or `"synthetic"` for the
            small stand-in of the smoke run.
        seeds: Seeds of the standard and strong-penalty runs.
        early_stopping_seeds: Seeds of the early-stopping runs. They must
            include every entry of `seeds`, since claim c5 pairs the two
            regimes seed by seed.
        epochs: Epochs of the standard and strong-penalty runs.
        batch_size: Examples per optimizer step.
        micro_batch_size: Examples per forward and backward pass. A step
            accumulates the gradients of its micro-batches.
        eval_interval: Epochs between evaluations of the training curves.
        precision: `"bfloat16"` for bfloat16 autocast in channels-last
            memory format, or `"float32"`.
        pretrained: Whether to start from the ImageNet weights of
            torchvision, as the paper does. The smoke run starts from random
            weights so that it never downloads them.
        device: PyTorch device, for example `"cpu"` or `"cuda"`.
    """

    data: str
    seeds: tuple[int, ...]
    early_stopping_seeds: tuple[int, ...]
    epochs: int = 300
    batch_size: int = 128
    micro_batch_size: int = 64
    eval_interval: int = 10
    precision: str = "bfloat16"
    pretrained: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.data not in _DATA:
            raise ValueError(f"data must be one of {_DATA}, got {self.data!r}")
        if not self.seeds or not self.early_stopping_seeds:
            raise ValueError("at least one seed per regime is required")
        if not set(self.seeds) <= set(self.early_stopping_seeds):
            raise ValueError("early_stopping_seeds must include every entry of seeds")
        if len(set(self.seeds)) < len(self.seeds) or len(
            set(self.early_stopping_seeds)
        ) < len(self.early_stopping_seeds):
            raise ValueError("seeds must not repeat")
        for name in ("epochs", "batch_size", "micro_batch_size", "eval_interval"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if self.micro_batch_size > self.batch_size:
            raise ValueError("micro_batch_size must not exceed batch_size")
        if self.precision not in _PRECISIONS:
            raise ValueError(
                f"precision must be one of {_PRECISIONS}, got {self.precision!r}"
            )
