"""Run the Section 5.2 simulation and write one result record per claim.

Usage::

    python -m papers.duchi2019_variance_regularization.run \
        --config papers/duchi2019_variance_regularization/configs/smoke.toml
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
from .data import sample_signs
from .model import (
    chi_square_radius,
    erm_solution,
    minimize_projected,
    optimum,
    population_risk,
    radius_rho,
    refine,
    robust_objective,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "results"
RUNS_DIR = REPOSITORY_ROOT / "runs" / PAPER_ID

# A descent point this close to v is accepted without the subgradient phase
# (see README, Setup).
SETTLED = 1e-10

C2_CELL = (1000, 1.0)
C3_CELL = (100, 1.0)


@dataclass(frozen=True)
class CellRisks:
    """Population risks of every replication of one seed and grid cell.

    Attributes:
        erm: Risks of the empirical risk minimizers, shape `(replications,)`.
        descent: Risks after the descent phase alone, the solver of
            Section 5.1, shape `(replications,)`.
        robust: Risks of the robust solutions, shape `(replications,)`.
        refined: Replications that needed the subgradient phase because
            the descent phase did not end at $v$.
        seconds: Wall-clock time of the robust solves of this cell.
    """

    erm: torch.Tensor
    descent: torch.Tensor
    robust: torch.Tensor
    refined: int
    seconds: float


Risks = dict[tuple[int, int, float], CellRisks]


def _chunk_size(config: Config, num_samples: int) -> int:
    bytes_per_replication = num_samples * config.dimension * 8
    budget = config.chunk_megabytes * 2**20
    return max(1, min(config.replications_per_seed, budget // bytes_per_replication))


def simulate(config: Config) -> Risks:
    """Solve ERM and the robust problem for every seed, cell and replication.

    Args:
        config: Settings of the run.

    Returns:
        Map from `(seed, n, B)` to the risks of that cell.
    """
    dtype = getattr(torch, config.dtype)
    device = torch.device(config.device)
    v = optimum(config.dimension, config.domain_radius, dtype, device)
    risks: Risks = {}
    for seed in config.seeds:
        seed_everything(seed)
        for n_index, num_samples in enumerate(config.sample_sizes):
            chunk = _chunk_size(config, num_samples)
            for b_index, noise in enumerate(config.noise_levels):
                cell = n_index * len(config.noise_levels) + b_index
                signs = sample_signs(
                    seed,
                    cell,
                    config.replications_per_seed,
                    num_samples,
                    config.dimension,
                )
                rho = radius_rho(
                    num_samples,
                    noise,
                    config.dimension,
                    config.domain_radius,
                    config.delta,
                )
                radius = chi_square_radius(rho, num_samples)
                erm_parts, descent_parts, robust_parts = [], [], []
                refined = 0
                start = time.perf_counter()
                for first in range(0, config.replications_per_seed, chunk):
                    block = signs[first : first + chunk]
                    features = block.to(device=device, dtype=dtype) * noise
                    erm = erm_solution(features, v, config.domain_radius)
                    descent = minimize_projected(
                        robust_objective(features, v, radius),
                        erm,
                        config.domain_radius,
                        config.descent_iterations,
                    )
                    robust = descent.clone()
                    distance = torch.linalg.vector_norm(descent - v, dim=-1)
                    pending = torch.nonzero(distance > SETTLED).squeeze(-1)
                    if config.subgradient_iterations and pending.numel():
                        robust[pending] = refine(
                            robust_objective(features[pending], v, radius),
                            descent[pending],
                            config.domain_radius,
                            config.subgradient_iterations,
                        )
                        refined += pending.numel()
                    erm_parts.append(population_risk(erm, v))
                    descent_parts.append(population_risk(descent, v))
                    robust_parts.append(population_risk(robust, v))
                risks[(seed, num_samples, noise)] = CellRisks(
                    erm=torch.cat(erm_parts).double().cpu(),
                    descent=torch.cat(descent_parts).double().cpu(),
                    robust=torch.cat(robust_parts).double().cpu(),
                    refined=refined,
                    seconds=time.perf_counter() - start,
                )
                print(
                    f"seed {seed}, n = {num_samples}, B = {noise:g}: "
                    f"{time.perf_counter() - start:.0f} s, refined {refined}",
                    flush=True,
                )
    return risks


def claim_values(config: Config, risks: Risks) -> list[list[float]]:
    """Measure every claim for every seed.

    Args:
        config: Settings of the run.
        risks: Output of `simulate`.

    Returns:
        One list per claim in `CLAIMS` order, holding one value per seed.

    Raises:
        ValueError: If the grid misses a cell that a claim needs.
    """
    for num_samples, noise in (C2_CELL, C3_CELL):
        if num_samples not in config.sample_sizes or noise not in config.noise_levels:
            raise ValueError(f"the grid must contain n = {num_samples}, B = {noise}")
    values: list[list[float]] = [[], [], []]
    for seed in config.seeds:
        cells = [cell for key, cell in risks.items() if key[0] == seed]
        wins = torch.cat([cell.robust < cell.erm for cell in cells])
        values[0].append(wins.double().mean().item())
        values[1].append(risks[(seed, *C2_CELL)].robust.mean().item())
        c3 = risks[(seed, *C3_CELL)]
        values[2].append((c3.robust.mean() / c3.erm.mean()).item())
    return values


def run(config: Config) -> list[ClaimResult]:
    """Simulate and measure every claim.

    Args:
        config: Settings of the run.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    values = claim_values(config, simulate(config))
    return [
        make_result(PAPER_ID, claim, config.seeds, claim_value, config.device)
        for claim, claim_value in zip(CLAIMS, values, strict=True)
    ]


CURVE_COLUMNS = (
    "num_samples",
    "noise",
    "erm",
    "descent",
    "robust",
    "wins",
    "refined",
    "seconds",
)


def curve_record(config: Config, risks: Risks) -> dict[str, list[float]]:
    """Summarize the grid like Table 1 of the paper.

    Args:
        config: Settings of the run.
        risks: Output of `simulate`.

    Returns:
        One entry per grid cell under each name of `CURVE_COLUMNS`, sample
        sizes in the outer order and noise levels in the inner one. The
        cell's `num_samples` and `noise`, the mean risk over all seeds and
        replications of ERM (`erm`), of the descent phase alone (`descent`)
        and of the robust solution (`robust`), the share of replications
        where the robust solution beats ERM (`wins`), the share that needed
        the subgradient phase (`refined`), and the time spent in robust
        solves (`seconds`).
    """
    rows = []
    for num_samples in config.sample_sizes:
        for noise in config.noise_levels:
            cells = [risks[(seed, num_samples, noise)] for seed in config.seeds]
            erm = torch.cat([cell.erm for cell in cells])
            descent = torch.cat([cell.descent for cell in cells])
            robust = torch.cat([cell.robust for cell in cells])
            rows.append(
                (
                    float(num_samples),
                    noise,
                    erm.mean().item(),
                    descent.mean().item(),
                    robust.mean().item(),
                    (robust < erm).double().mean().item(),
                    sum(cell.refined for cell in cells) / erm.numel(),
                    sum(cell.seconds for cell in cells),
                )
            )
    return {
        name: list(column)
        for name, column in zip(CURVE_COLUMNS, zip(*rows, strict=True), strict=True)
    }


def table(config: Config, risks: Risks) -> str:
    """Format `curve_record` as a Markdown table.

    Args:
        config: Settings of the run.
        risks: Output of `simulate`.

    Returns:
        Markdown table with one row per grid cell.
    """
    lines = [
        "| n | B | mean R(erm) | mean R(descent only) | mean R(rob) | rob < erm "
        "| refined | seconds |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for num_samples, noise, erm, descent, robust, wins, refined, seconds in zip(
        *curve_record(config, risks).values(), strict=True
    ):
        lines.append(
            f"| {num_samples:g} | {noise:g} | {erm:.3e} | {descent:.3e} | "
            f"{robust:.3e} | {wins:.4f} | {refined:.4f} | {seconds:.1f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line, run, and write the records, curves and table.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Run the reproduction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)
    config = load_config(args.config, Config)

    risks = simulate(config)
    values = claim_values(config, risks)
    summary = table(config, risks)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    (args.table_dir / f"{args.config.stem}_table.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(summary)
    write_curves(PAPER_ID, curve_record(config, risks), args.output_dir)
    for claim, claim_value in zip(CLAIMS, values, strict=True):
        result = make_result(PAPER_ID, claim, config.seeds, claim_value, config.device)
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
