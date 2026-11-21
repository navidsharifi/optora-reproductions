"""Run the experiment and write the claim records and the curve record.

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
from optora_reproductions.results import (
    ClaimResult,
    make_result,
    write_curves,
    write_result,
)
from optora_reproductions.seeding import seed_everything

from . import PAPER_ID
from .claims import CLAIMS
from .config import Config
from .data import sample_losses

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def measure(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the worst case and its closed form for every seed.

    The seeds are independent instances, stacked into one batch so that
    Optora solves all of them in a single call.

    Args:
        config: Settings of the run.

    Returns:
        The worst-case expectation computed by Optora and the closed form
        it should equal, each of shape `(len(config.seeds),)`.
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
    return worst_case, mean + torch.sqrt(radius * variance)


def claim_results(
    config: Config, worst_case: torch.Tensor, closed_form: torch.Tensor
) -> list[ClaimResult]:
    """Build one record per entry of `CLAIMS` from the output of `measure`."""
    error = (worst_case - closed_form).abs()
    return [make_result(PAPER_ID, CLAIMS[0], config.seeds, error, config.device)]


def curve_record(
    worst_case: torch.Tensor, closed_form: torch.Tensor
) -> dict[str, list[float]]:
    """Return the series behind the figures, one entry per seed.

    A curve record holds what the figures of the report plot and a claim
    record has no place for. `figures.py` reads it back.
    """
    return {
        "worst_case": worst_case.double().tolist(),
        "closed_form": closed_form.double().tolist(),
    }


def run(config: Config) -> list[ClaimResult]:
    """Run every seed and measure every claim.

    Args:
        config: Settings of the run.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    return claim_results(config, *measure(config))


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line, run, and write the records and the curves.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Run the reproduction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    config = load_config(args.config, Config)

    worst_case, closed_form = measure(config)
    write_curves(PAPER_ID, curve_record(worst_case, closed_form), args.output_dir)
    for result in claim_results(config, worst_case, closed_form):
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{result.claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
