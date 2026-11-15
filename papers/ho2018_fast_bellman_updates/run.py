"""Run the Section 5.1 experiment and write one result record per claim.

Usage::

    python -m papers.ho2018_fast_bellman_updates.run \
        --config papers/ho2018_fast_bellman_updates/configs/smoke.toml
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from optora_reproductions.config import load_config
from optora_reproductions.results import ClaimResult, make_result, write_result
from optora_reproductions.seeding import seed_everything

from . import PAPER_ID
from .claims import CLAIMS
from .config import Config
from .data import inventory_instances, random_instance
from .model import lp_q, median_seconds, optora_q, optora_single_q

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "results"
RUNS_DIR = REPOSITORY_ROOT / "runs" / PAPER_ID


@dataclass(frozen=True)
class Measurement:
    """Errors and timings of one seed and one number of states.

    Attributes:
        random_error: Max absolute error against HiGHS, random instance.
        inventory_error: Max absolute error against HiGHS, all orders of
            the inventory instance.
        lp_seconds: Mean over budgets of the median HiGHS time per call.
        optora_seconds: Mean over budgets of the median Optora time per call.
    """

    random_error: float
    inventory_error: float
    lp_seconds: float
    optora_seconds: float


Measurements = dict[tuple[int, int], Measurement]


def max_error(
    nominal: torch.Tensor, z: torch.Tensor, budgets: Sequence[float], device: str
) -> float:
    """Largest gap between Optora and HiGHS over instances and budgets.

    Args:
        nominal: Nominal distributions of shape `(b, S)`, float64 on CPU.
        z: Vectors $z$ of shape `(b, S)`, float64 on CPU.
        budgets: Budgets $\\xi$.
        device: Device of the Optora call.

    Returns:
        The maximum absolute difference.
    """
    budget_tensor = torch.tensor(budgets, dtype=nominal.dtype, device=device)
    values = optora_q(nominal.to(device), z.to(device), budget_tensor).double().cpu()
    reference = torch.tensor(
        [
            [lp_q(p.numpy(), v.numpy(), budget) for budget in budgets]
            for p, v in zip(nominal.double(), z.double(), strict=True)
        ],
        dtype=torch.float64,
    )
    return float((values - reference).abs().max())


def mean_times(
    nominal: torch.Tensor, z: torch.Tensor, budgets: Sequence[float], repeats: int
) -> tuple[float, float]:
    """Time one robust Q-value with HiGHS and with Optora on CPU.

    Args:
        nominal: Nominal distribution of shape `(S,)`, float64 on CPU.
        z: Vector $z$ of shape `(S,)`, float64 on CPU.
        budgets: Budgets $\\xi$; times are averaged over them.
        repeats: Timed calls per budget.

    Returns:
        Mean HiGHS seconds and mean Optora seconds per call.
    """
    nominal_np, z_np = nominal.numpy(), z.numpy()
    lp = [median_seconds(partial(lp_q, nominal_np, z_np, b), repeats) for b in budgets]
    fast = [
        median_seconds(partial(optora_single_q, nominal, z, b), repeats)
        for b in budgets
    ]
    return sum(lp) / len(lp), sum(fast) / len(fast)


def measure(config: Config) -> Measurements:
    """Check exactness and time both solvers for every seed and size.

    Args:
        config: Settings of the run.

    Returns:
        Map from `(seed, S)` to its measurement.
    """
    dtype = getattr(torch, config.dtype)
    states = sorted(set(config.state_counts) | {config.timing_states})
    inventory = {
        num_states: max_error(
            *inventory_instances(num_states, config.discount, dtype),
            config.budgets,
            config.device,
        )
        for num_states in config.state_counts
    }
    measurements: Measurements = {}
    for seed in config.seeds:
        seed_everything(seed)
        for num_states in states:
            nominal, z = random_instance(seed, num_states, dtype)
            lp_seconds, optora_seconds = mean_times(
                nominal, z, config.budgets, config.timing_repeats
            )
            measurements[(seed, num_states)] = Measurement(
                random_error=max_error(
                    nominal.unsqueeze(0), z.unsqueeze(0), config.budgets, config.device
                ),
                inventory_error=inventory.get(num_states, 0.0),
                lp_seconds=lp_seconds,
                optora_seconds=optora_seconds,
            )
            print(f"seed {seed}, S = {num_states} done", flush=True)
    return measurements


def claim_values(config: Config, measurements: Measurements) -> list[list[float]]:
    """Measure every claim for every seed.

    Args:
        config: Settings of the run.
        measurements: Output of `measure`.

    Returns:
        One list per claim in `CLAIMS` order, holding one value per seed.
    """
    values: list[list[float]] = [[], []]
    for seed in config.seeds:
        errors = [
            max(m.random_error, m.inventory_error)
            for (s, n), m in measurements.items()
            if s == seed and n in config.state_counts
        ]
        values[0].append(max(errors))
        timed = measurements[(seed, config.timing_states)]
        values[1].append(math.log10(timed.lp_seconds / timed.optora_seconds))
    return values


def run(config: Config) -> list[ClaimResult]:
    """Measure and build one record per claim.

    Args:
        config: Settings of the run.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    values = claim_values(config, measure(config))
    return [
        make_result(PAPER_ID, claim, config.seeds, claim_value, "cpu")
        for claim, claim_value in zip(CLAIMS, values, strict=True)
    ]


def table(config: Config, measurements: Measurements) -> str:
    """Summarize errors and timings per number of states, like Figure 4.

    Args:
        config: Settings of the run.
        measurements: Output of `measure`.

    Returns:
        Markdown table with the worst error over seeds and the mean times.
    """
    lines = [
        "| S | max error (random) | max error (inventory) | HiGHS ms | Optora ms "
        "| speedup |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for num_states in sorted({n for _, n in measurements}):
        rows = [measurements[(seed, num_states)] for seed in config.seeds]
        lp = sum(m.lp_seconds for m in rows) / len(rows)
        fast = sum(m.optora_seconds for m in rows) / len(rows)
        lines.append(
            f"| {num_states} | {max(m.random_error for m in rows):.2e} | "
            f"{max(m.inventory_error for m in rows):.2e} | {1e3 * lp:.3f} | "
            f"{1e3 * fast:.3f} | {lp / fast:.1f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line, run, and write the records and the table.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Run the reproduction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)
    config = load_config(args.config, Config)

    measurements = measure(config)
    values = claim_values(config, measurements)
    summary = table(config, measurements)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    (args.table_dir / f"{args.config.stem}_table.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(summary)
    for claim, claim_value in zip(CLAIMS, values, strict=True):
        result = make_result(PAPER_ID, claim, config.seeds, claim_value, "cpu")
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
