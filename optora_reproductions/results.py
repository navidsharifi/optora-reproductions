"""Result records for reproduced claims.

A record holds one claim of one paper, the values measured for it over
several seeds, and the software and hardware they were measured on. Records
are stored as JSON files in `results/` and are the only source of the
numbers shown in the repository README.

A paper also keeps one curve record in `results/curves/`. It holds the
series behind the figures of its report, such as a metric at every epoch,
which a claim record has no place for.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import json
import math
import platform
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CURVES_FOLDER = "curves"
"""Subfolder of a results folder that holds the curve records."""

_IDENTIFIER = re.compile(r"[a-z0-9_]+")


def _check_identifier(kind: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{kind} must match [a-z0-9_]+, got {value!r}")


@dataclass(frozen=True)
class Claim:
    """A quantitative claim of a paper, with a tolerance fixed before the run.

    A qualitative claim, for example that one method beats another, is
    written as a scalar metric such as the difference of the two accuracies.

    Attributes:
        claim_id: Identifier unique within the paper, for example `"c1"`.
        location: Where the claim appears, for example `"Table 2, row 3"`.
        metric: Name of the measured quantity.
        paper_value: Value reported in the paper.
        tolerance: Largest accepted absolute difference between the mean
            over seeds and `paper_value`.
    """

    claim_id: str
    location: str
    metric: str
    paper_value: float
    tolerance: float

    def __post_init__(self) -> None:
        _check_identifier("claim_id", self.claim_id)
        if not math.isfinite(self.paper_value):
            raise ValueError(f"paper_value must be finite, got {self.paper_value}")
        if not self.tolerance >= 0.0:
            raise ValueError(f"tolerance must be nonnegative, got {self.tolerance}")


@dataclass(frozen=True)
class Environment:
    """Software and hardware a result was measured on.

    Attributes:
        python: Python version.
        torch: PyTorch version, including the CUDA tag if any.
        optora: Installed `optora` version.
        platform: Operating system.
        device: Name of the device the experiment ran on.
    """

    python: str
    torch: str
    optora: str
    platform: str
    device: str


def capture_environment(device: torch.device | str) -> Environment:
    """Describe the current interpreter, libraries and device.

    Args:
        device: Device the experiment ran on.

    Returns:
        The environment record.
    """
    device = torch.device(device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine() or device.type
    return Environment(
        python=platform.python_version(),
        torch=torch.__version__,
        optora=importlib.metadata.version("optora"),
        platform=platform.platform(terse=True),
        device=device_name,
    )


@dataclass(frozen=True)
class ClaimResult:
    """Values measured for one claim over several seeds.

    Attributes:
        paper_id: Folder name of the paper under `papers/`.
        claim: The claim being reproduced.
        seeds: Seeds of the independent runs, without repetitions.
        values: Measured value of `claim.metric` for each seed.
        environment: Where the values were measured.
        date: ISO date of the measurement.
    """

    paper_id: str
    claim: Claim
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    environment: Environment
    date: str

    def __post_init__(self) -> None:
        _check_identifier("paper_id", self.paper_id)
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(self.seeds) != len(self.values):
            raise ValueError(
                f"got {len(self.seeds)} seeds but {len(self.values)} values"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be distinct, got {self.seeds}")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError(f"values must be finite, got {self.values}")
        datetime.date.fromisoformat(self.date)

    @property
    def mean(self) -> float:
        """Mean of `values` over seeds."""
        return statistics.fmean(self.values)

    @property
    def std(self) -> float:
        """Sample standard deviation of `values`, or 0 for a single seed."""
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def within_tolerance(self) -> bool:
        """Whether the mean lies within the claim's tolerance."""
        return abs(self.mean - self.claim.paper_value) <= self.claim.tolerance

    @property
    def filename(self) -> str:
        """Name of the JSON file holding this record."""
        return f"{self.paper_id}.{self.claim.claim_id}.json"

    def to_json(self) -> str:
        """Serialize the record as indented JSON."""
        return json.dumps(dataclasses.asdict(self), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ClaimResult:
        """Parse a record written by `to_json`.

        Args:
            text: JSON text.

        Returns:
            The record.
        """
        data: dict[str, Any] = json.loads(text)
        return cls(
            paper_id=data["paper_id"],
            claim=Claim(**data["claim"]),
            seeds=tuple(data["seeds"]),
            values=tuple(float(value) for value in data["values"]),
            environment=Environment(**data["environment"]),
            date=data["date"],
        )


def make_result(
    paper_id: str,
    claim: Claim,
    seeds: Sequence[int],
    values: Sequence[float] | torch.Tensor,
    device: torch.device | str,
) -> ClaimResult:
    """Build a record dated today for the current environment.

    Args:
        paper_id: Folder name of the paper under `papers/`.
        claim: The claim being reproduced.
        seeds: Seeds of the independent runs.
        values: One value per seed. A tensor is copied to the host once.
        device: Device the experiment ran on.

    Returns:
        The record.
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().double().cpu().tolist()
    return ClaimResult(
        paper_id=paper_id,
        claim=claim,
        seeds=tuple(seeds),
        values=tuple(float(value) for value in values),
        environment=capture_environment(device),
        date=datetime.date.today().isoformat(),
    )


def write_result(result: ClaimResult, directory: str | Path) -> Path:
    """Write a record to `directory`, replacing an earlier record of the claim.

    Args:
        result: The record.
        directory: Target folder, created if missing.

    Returns:
        Path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / result.filename
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def read_results(directory: str | Path) -> list[ClaimResult]:
    """Read every record in `directory`, sorted by paper and claim.

    Args:
        directory: Folder holding `<paper_id>.<claim_id>.json` files.

    Returns:
        The records. Empty if the folder has none.

    Raises:
        ValueError: If a file name does not match the record it holds.
    """
    results = []
    for path in sorted(Path(directory).glob("*.json")):
        result = ClaimResult.from_json(path.read_text(encoding="utf-8"))
        if path.name != result.filename:
            raise ValueError(f"{path} holds the record {result.filename}")
        results.append(result)
    return sorted(results, key=lambda result: (result.paper_id, result.claim.claim_id))


def write_curves(
    paper_id: str, curves: Mapping[str, Any], directory: str | Path
) -> Path:
    """Write the curve record of a paper, replacing an earlier one.

    Args:
        paper_id: Folder name of the paper under `papers/`.
        curves: Series of the paper's figures as plain JSON types. Every
            number must be finite.
        directory: Results folder. The record goes to its `curves`
            subfolder, which is created if missing.

    Returns:
        Path of the written file.
    """
    _check_identifier("paper_id", paper_id)
    folder = Path(directory) / CURVES_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{paper_id}.json"
    path.write_text(json.dumps(curves, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_curves(paper_id: str, directory: str | Path) -> dict[str, Any] | None:
    """Read the curve record of a paper.

    Args:
        paper_id: Folder name of the paper under `papers/`.
        directory: Results folder.

    Returns:
        The record, or `None` if the paper has none yet.
    """
    path = Path(directory) / CURVES_FOLDER / f"{paper_id}.json"
    if not path.exists():
        return None
    curves: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return curves
