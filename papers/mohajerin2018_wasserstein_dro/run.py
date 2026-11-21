"""Run the Section 7.2 experiment and write one result record per claim.

Usage::

    python -m papers.mohajerin2018_wasserstein_dro.run \
        --config papers/mohajerin2018_wasserstein_dro/configs/smoke.toml
"""

from __future__ import annotations

import argparse
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
from .data import NUM_ASSETS, return_moments, sample_returns
from .model import lp_portfolio, out_of_sample, robust_portfolios

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "results"
RUNS_DIR = REPOSITORY_ROOT / "runs" / PAPER_ID


@dataclass(frozen=True)
class Measurements:
    r"""Solutions of every simulation, on CPU.

    The first axis of every tensor is the simulation. Tensors of the sweep
    have one column per entry of `Config.radii`.

    Attributes:
        portfolios: $\hat x_N$ at the certificate radius, shape `(s, m)`.
        certificate: $\hat J_N$ at the certificate radius, shape `(s,)`.
        performance: $J(\hat x_N)$ at the certificate radius, shape `(s,)`.
        lp_portfolios: HiGHS portfolios at the certificate radius, `(s, m)`.
        lp_values: HiGHS optimal values at the certificate radius, `(s,)`.
        sweep_certificate: $\hat J_N(\varepsilon)$ of the sweep, `(s, r)`.
        sweep_performance: $J(\hat x_N(\varepsilon))$ of the sweep, `(s, r)`.
        sweep_lp_values: HiGHS optimal values of the sweep, `(s, r)`.
    """

    portfolios: torch.Tensor
    certificate: torch.Tensor
    performance: torch.Tensor
    lp_portfolios: torch.Tensor
    lp_values: torch.Tensor
    sweep_certificate: torch.Tensor
    sweep_performance: torch.Tensor
    sweep_lp_values: torch.Tensor


def solve(
    config: Config, num_samples: int, radii: Sequence[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve every simulation at every radius with Optora and with HiGHS.

    Args:
        config: Settings of the run.
        num_samples: Sample size $N$ of every training dataset.
        radii: Wasserstein radii.

    Returns:
        Optora portfolios `(s, r, m)`, certificates `(s, r)`, out-of-sample
        performance `(s, r)`, HiGHS portfolios `(s, r, m)` and HiGHS values
        `(s, r)`, all on CPU.
    """
    dtype = getattr(torch, config.dtype)
    device = torch.device(config.device)
    samples = sample_returns(config.seeds, num_samples, dtype)
    x, _, certificate = robust_portfolios(
        samples.to(device),
        torch.tensor(radii, dtype=dtype, device=device),
        config.far,
        config.iterations,
    )
    x, certificate = x.double().cpu(), certificate.double().cpu()
    mean, covariance = return_moments()
    reference = [
        [lp_portfolio(dataset.double().numpy(), radius) for radius in radii]
        for dataset in samples
    ]
    lp_x = torch.tensor([[row[0].tolist() for row in rows] for rows in reference])
    lp_values = torch.tensor([[row[2] for row in rows] for rows in reference])
    return x, certificate, out_of_sample(x, mean, covariance), lp_x, lp_values


def measure(config: Config) -> Measurements:
    """Solve the certificate setting and the radius sweep.

    Args:
        config: Settings of the run.

    Returns:
        The solutions of every simulation.
    """
    seed_everything(config.seeds[0])
    x, certificate, performance, lp_x, lp_values = solve(
        config, config.certificate_samples, [config.certificate_radius]
    )
    print(f"N = {config.certificate_samples} done", flush=True)
    _, sweep_certificate, sweep_performance, _, sweep_lp = solve(
        config, config.sweep_samples, config.radii
    )
    print(f"N = {config.sweep_samples} done", flush=True)
    return Measurements(
        portfolios=x[:, 0],
        certificate=certificate[:, 0],
        performance=performance[:, 0],
        lp_portfolios=lp_x[:, 0],
        lp_values=lp_values[:, 0],
        sweep_certificate=sweep_certificate,
        sweep_performance=sweep_performance,
        sweep_lp_values=sweep_lp,
    )


def best_radius(measurements: Measurements) -> int:
    """Index of the sweep radius with the lowest mean out-of-sample $J$.

    Args:
        measurements: Output of `measure`.

    Returns:
        Column index into `Config.radii`.
    """
    return int(measurements.sweep_performance.mean(dim=0).argmin())


def claim_values(config: Config, measurements: Measurements) -> list[torch.Tensor]:
    """Measure every claim for every simulation.

    Args:
        config: Settings of the run.
        measurements: Output of `measure`.

    Returns:
        One tensor of shape `(s,)` per claim, in `CLAIMS` order.
    """
    deviation = (measurements.portfolios - 1.0 / NUM_ASSETS).abs().amax(dim=-1)
    reliable = (measurements.performance <= measurements.certificate).double()
    sweep = measurements.sweep_performance
    small = config.radii.index(config.small_radius)
    return [deviation, reliable, sweep[:, small] - sweep[:, best_radius(measurements)]]


CURVE_COLUMNS = (
    "num_samples",
    "radius",
    "performance_mean",
    "performance_low",
    "performance_high",
    "reliability",
    "certificate_gap",
)


def curve_record(config: Config, measurements: Measurements) -> dict[str, list[float]]:
    """Summarize the sweep like Figure 5(a) and the certificate setting.

    Args:
        config: Settings of the run.
        measurements: Output of `measure`.

    Returns:
        One entry per sweep radius, followed by one for the certificate
        setting, under each name of `CURVE_COLUMNS`. The number of samples
        and the radius, the mean, 20% and 80% quantiles of $J$ over the
        simulations, the reliability, and the largest gap between the
        certificate and the HiGHS optimal value.
    """

    def row(
        num_samples: int,
        radius: float,
        performance: torch.Tensor,
        certificate: torch.Tensor,
        lp_values: torch.Tensor,
    ) -> tuple[float, ...]:
        low, high = torch.quantile(performance, torch.tensor([0.2, 0.8]).double())
        return (
            float(num_samples),
            radius,
            float(performance.mean()),
            float(low),
            float(high),
            float((performance <= certificate).double().mean()),
            float((certificate - lp_values).abs().max()),
        )

    rows = [
        row(
            config.sweep_samples,
            radius,
            measurements.sweep_performance[:, column],
            measurements.sweep_certificate[:, column],
            measurements.sweep_lp_values[:, column],
        )
        for column, radius in enumerate(config.radii)
    ]
    rows.append(
        row(
            config.certificate_samples,
            config.certificate_radius,
            measurements.performance,
            measurements.certificate,
            measurements.lp_values,
        )
    )
    return {
        name: list(column)
        for name, column in zip(CURVE_COLUMNS, zip(*rows, strict=True), strict=True)
    }


def table(config: Config, measurements: Measurements) -> str:
    """Format `curve_record` as Markdown, with the portfolio gaps below it.

    Args:
        config: Settings of the run.
        measurements: Output of `measure`.

    Returns:
        Markdown with one row per sweep radius and one for the certificate
        setting, and a line with the largest portfolio gaps.
    """
    lines = [
        "| N | radius | mean J | J 20% | J 80% | reliability | max certificate - LP |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for num_samples, radius, mean, low, high, reliability, gap in zip(
        *curve_record(config, measurements).values(), strict=True
    ):
        lines.append(
            f"| {num_samples:g} | {radius:g} | {mean:.4f} | {low:.4f} | "
            f"{high:.4f} | {reliability:.3f} | {gap:.1e} |"
        )
    portfolio_gap = (measurements.portfolios - measurements.lp_portfolios).abs().max()
    lp_deviation = (measurements.lp_portfolios - 1.0 / NUM_ASSETS).abs().max()
    lines += [
        "",
        f"N = {config.certificate_samples}, radius {config.certificate_radius:g}: "
        f"max |x - x_LP| = {float(portfolio_gap):.1e}, "
        f"max |x_LP - 1/m| = {float(lp_deviation):.1e}, "
        f"best sweep radius {config.radii[best_radius(measurements)]:g}",
    ]
    return "\n".join(lines)


def run(config: Config) -> list[ClaimResult]:
    """Measure and build one record per claim.

    Args:
        config: Settings of the run.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    values = claim_values(config, measure(config))
    return [
        make_result(PAPER_ID, claim, config.seeds, claim_value, config.device)
        for claim, claim_value in zip(CLAIMS, values, strict=True)
    ]


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

    measurements = measure(config)
    summary = table(config, measurements)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    (args.table_dir / f"{args.config.stem}_table.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(summary)
    write_curves(PAPER_ID, curve_record(config, measurements), args.output_dir)
    for claim, claim_value in zip(
        CLAIMS, claim_values(config, measurements), strict=True
    ):
        result = make_result(PAPER_ID, claim, config.seeds, claim_value, config.device)
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
