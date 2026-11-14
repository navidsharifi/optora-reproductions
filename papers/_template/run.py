"""Run the experiment and write one result record per claim.

Usage::

    python -m papers._template.run --config papers/_template/configs/smoke.toml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import optora
import torch

from optora_reproductions.config import load_config
from optora_reproductions.results import ClaimResult, make_result, write_result
from optora_reproductions.seeding import seed_everything

from . import PAPER_ID
from .claims import CLAIMS
from .config import Config
from .data import sample_losses

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def run(config: Config) -> list[ClaimResult]:
    """Run every seed and measure every claim.

    The seeds are independent instances, stacked into one batch so that
    Optora solves all of them in a single call.

    Args:
        config: Settings of the run.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    dtype = getattr(torch, config.dtype)
    device = torch.device(config.device)
    losses = torch.stack(
        [
            sample_losses(config.num_samples, seed_everything(seed), dtype, device)
            for seed in config.seeds
        ]
    )
    nominal = torch.full(
        (config.num_samples,), 1.0 / config.num_samples, dtype=dtype, device=device
    )

    mean = losses.mean(dim=-1)
    variance = losses.var(dim=-1, correction=0)
    largest_radius = variance / (mean - losses.amin(dim=-1)) ** 2
    radius = config.radius_fraction * largest_radius

    ambiguity_set = optora.ChiSquareAmbiguitySet(nominal, radius=radius)
    worst_case = ambiguity_set.worst_case_expectation(losses)
    error = (worst_case - (mean + torch.sqrt(radius * variance))).abs()
    return [make_result(PAPER_ID, CLAIMS[0], config.seeds, error, device)]


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line, run, and write the records.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Run the reproduction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    for result in run(load_config(args.config, Config)):
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{result.claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
