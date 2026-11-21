"""Run the Digits experiment and write one result record per claim.

Usage::

    python -m papers.levy2020_large_scale_dro.run \
        --config papers/levy2020_large_scale_dro/configs/smoke.toml
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from .data import Digits, load_digits, synthetic_digits
from .model import (
    ERM,
    ERM_STEP_SIZES,
    FULL_BATCH,
    OBJECTIVES,
    STEP_SIZES,
    WEIGHT_DECAY,
    Run,
    Setting,
    epochs_to_reach,
    train,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "results"
RUNS_DIR = REPOSITORY_ROOT / "runs" / PAPER_ID
TOLERANCE = 0.02
"""Relative distance to the optimal training objective, Table 5."""

REPORTED_EPOCHS = (1, 2, 5, 10, 20, 50, 100, 200)
"""Epochs listed in the table of claim c4 by epoch, besides the last one."""

Key = tuple[str, int, float, int]
"""A run's objective, batch size, weight decay and seed."""


def settings(config: Config) -> list[Setting]:
    """Return every training run of the experiment, minibatch runs first.

    Args:
        config: Settings of the experiment.

    Returns:
        The minibatch runs of every objective, batch size and seed, the
        full-batch run of every objective, and the ERM runs of both weight
        decays and every seed. Minibatch and ERM runs are evaluated after
        every epoch, so their test losses can be compared epoch by epoch.
    """
    minibatch = [
        Setting(
            objective,
            size,
            STEP_SIZES[(objective, size)],
            WEIGHT_DECAY,
            config.epochs,
            1,
            seed,
        )
        for objective in OBJECTIVES
        for size in config.batch_sizes
        for seed in config.seeds
    ]
    full_batch = [
        Setting(
            objective,
            FULL_BATCH,
            STEP_SIZES[(objective, FULL_BATCH)],
            WEIGHT_DECAY,
            config.full_batch_epochs,
            config.full_batch_eval_interval,
            config.full_batch_seed,
        )
        for objective in OBJECTIVES
    ]
    erm = [
        Setting(
            ERM,
            config.erm_batch_size,
            step_size,
            weight_decay,
            config.epochs,
            1,
            seed,
        )
        for weight_decay, step_size in ERM_STEP_SIZES.items()
        for seed in config.seeds
    ]
    return minibatch + full_batch + erm


def key(setting: Setting) -> Key:
    """Return the lookup key of a run."""
    return (setting.objective, setting.batch_size, setting.weight_decay, setting.seed)


def load_data(config: Config) -> Digits:
    """Load the dataset named by `config.data`."""
    if config.data == "digits":
        return load_digits(config.typed_train)
    return synthetic_digits()


def measure(config: Config) -> dict[Key, Run]:
    """Train every run of the experiment.

    Args:
        config: Settings of the experiment.

    Returns:
        Every run, keyed by objective, batch size, weight decay and seed.
    """
    data = load_data(config)
    device = torch.device(config.device)
    train_split = (data.train.features.to(device), data.train.targets.to(device))
    test_split = (
        data.test.features.to(device),
        data.test.targets.to(device),
        data.test.groups.to(device),
    )
    runs: dict[Key, Run] = {}
    for setting in settings(config):
        seed_everything(setting.seed)
        start = time.perf_counter()
        runs[key(setting)] = train(setting, train_split, test_split, config.cuda_graphs)
        print(
            f"{setting.objective} n={setting.batch_size or 'full'} "
            f"mu={setting.weight_decay:g} seed={setting.seed} "
            f"done in {time.perf_counter() - start:.0f} s",
            flush=True,
        )
    return runs


def optimum(runs: dict[Key, Run], objective: str) -> float:
    """Return the lowest training objective of any run of `objective`."""
    return min(
        float(run.objective.min())
        for run in runs.values()
        if run.setting.objective == objective
    )


def speed_ups(config: Config, runs: dict[Key, Run], objective: str) -> list[float]:
    """Return the speed-up over the full batch of every seed, Table 5.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.
        objective: One of `OBJECTIVES`.

    Returns:
        Per seed, the epochs the full-batch run needs to get within 2% of the
        optimum divided by the fewest epochs any minibatch size needs.
    """
    target = (1.0 + TOLERANCE) * optimum(runs, objective)
    full = runs[(objective, FULL_BATCH, WEIGHT_DECAY, config.full_batch_seed)]
    full_epochs = epochs_to_reach(full, target)
    return [
        full_epochs
        / min(
            epochs_to_reach(runs[(objective, size, WEIGHT_DECAY, seed)], target)
            for size in config.batch_sizes
        )
        for seed in config.seeds
    ]


def worst_group_curve(run: Run) -> torch.Tensor:
    """Return the mean test log loss of the run's worst subgroup.

    A subgroup without test samples has no mean loss and is skipped. Every
    subgroup of the Digits test set has samples.

    Args:
        run: A training run.

    Returns:
        The worst-group test loss at every evaluated epoch of the run, in
        float64, shape `(len(run.epochs),)`.
    """
    return run.group_loss.double().nan_to_num(nan=-torch.inf).amax(dim=-1)


def worst_group_loss(run: Run) -> float:
    """Return the worst-group test loss of the run after the last epoch."""
    return float(worst_group_curve(run)[-1])


def worst_group_curves(
    config: Config, runs: dict[Key, Run], objective: str, weight_decay: float
) -> torch.Tensor:
    """Return the worst-group test loss of every seed of one compared run.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.
        objective: One of `OBJECTIVES`, at `config.comparison_batch_size`, or
            `ERM`, at `config.erm_batch_size`.
        weight_decay: Weight decay of the run.

    Returns:
        Shape `(len(config.seeds), config.epochs)`, one row per seed.
    """
    size = config.erm_batch_size if objective == ERM else config.comparison_batch_size
    return torch.stack(
        [
            worst_group_curve(runs[(objective, size, weight_decay, seed)])
            for seed in config.seeds
        ]
    )


def group_loss_reduction_curves(config: Config, runs: dict[Key, Run]) -> torch.Tensor:
    """Return claim c4 for every seed at every epoch.

    The pre-registered claim is the last column. The other columns apply
    the same comparison at an earlier epoch, with both the DRO and the ERM
    runs read at that epoch.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.

    Returns:
        Shape `(len(config.seeds), config.epochs)`. Entry `[s, t - 1]` is
        the relative reduction of the worst-group test loss of each
        objective at `config.comparison_batch_size` against the better ERM
        run, both after epoch `t` with seed `s`, averaged over the three
        objectives.
    """
    baseline = torch.stack(
        [
            worst_group_curves(config, runs, ERM, weight_decay)
            for weight_decay in ERM_STEP_SIZES
        ]
    ).amin(dim=0)
    robust = torch.stack(
        [
            worst_group_curves(config, runs, objective, WEIGHT_DECAY)
            for objective in OBJECTIVES
        ]
    )
    return (1.0 - robust / baseline).mean(dim=0)


def group_loss_reductions(config: Config, runs: dict[Key, Run]) -> list[float]:
    """Return claim c4 for every seed, after the last epoch."""
    values: list[float] = group_loss_reduction_curves(config, runs)[:, -1].tolist()
    return values


def claim_values(config: Config, runs: dict[Key, Run]) -> list[list[float]]:
    """Return the per-seed values of c1 to c4, in the order of `CLAIMS`."""
    return [speed_ups(config, runs, objective) for objective in OBJECTIVES] + [
        group_loss_reductions(config, runs)
    ]


def _mean_std(values: Sequence[float]) -> str:
    """Format a sample mean and standard deviation."""
    if len(values) < 2:
        return f"{statistics.fmean(values):.4g}"
    return f"{statistics.fmean(values):.4g} +- {statistics.stdev(values):.2g}"


def table(config: Config, runs: dict[Key, Run]) -> str:
    """Summarize every run of the experiment as a markdown table.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.

    Returns:
        One row per objective and batch size, with the epochs to 2% of the
        optimum, the final training objective, the test metrics and the
        training time per epoch, means over seeds.
    """
    lines = [
        "| objective | batch | epochs to 2% of opt | final objective "
        "| worst-group test loss | test loss | test accuracy | s per epoch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    optima = {objective: optimum(runs, objective) for objective in OBJECTIVES}
    groups: dict[tuple[str, int, float], list[Run]] = {}
    for run in runs.values():
        setting = run.setting
        groups.setdefault(
            (setting.objective, setting.batch_size, setting.weight_decay), []
        ).append(run)
    for (objective, size, weight_decay), members in groups.items():
        label = objective if objective != ERM else f"erm mu={weight_decay:g}"
        reached = (
            _mean_std(
                [
                    epochs_to_reach(run, (1.0 + TOLERANCE) * optima[objective])
                    for run in members
                ]
            )
            if objective in optima
            else "-"
        )
        lines.append(
            f"| {label} | {size or 'full'} | {reached} "
            f"| {_mean_std([float(run.objective[-1]) for run in members])} "
            f"| {_mean_std([worst_group_loss(run) for run in members])} "
            f"| {_mean_std([run.test_loss for run in members])} "
            f"| {_mean_std([run.test_accuracy for run in members])} "
            f"| {_mean_std([run.seconds_per_epoch for run in members])} |"
        )
    lines.append("")
    lines.append(
        "Optimal training objective: "
        + ", ".join(f"{objective} {value:.6g}" for objective, value in optima.items())
    )
    return "\n".join(lines)


def worst_group_means(config: Config, runs: dict[Key, Run]) -> dict[str, torch.Tensor]:
    """Return the mean worst-group test loss of every compared run by epoch.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.

    Returns:
        The mean over seeds at every epoch, shape `(config.epochs,)`, for
        each objective at `config.comparison_batch_size` and for each ERM
        run, named `erm mu=<weight decay>`.
    """
    columns = {
        objective: worst_group_curves(config, runs, objective, WEIGHT_DECAY)
        for objective in OBJECTIVES
    }
    for weight_decay in ERM_STEP_SIZES:
        columns[f"erm mu={weight_decay:g}"] = worst_group_curves(
            config, runs, ERM, weight_decay
        )
    return {name: values.mean(dim=0) for name, values in columns.items()}


def epoch_table(config: Config, runs: dict[Key, Run]) -> str:
    """Summarize claim c4 and the worst-group test loss by epoch.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.

    Returns:
        A markdown table with one row per epoch of `REPORTED_EPOCHS` within
        the budget and one for the last epoch. It lists the worst-group test
        loss of each objective at `config.comparison_batch_size` and of
        each ERM run, means over seeds, and claim c4 at that epoch. Two
        lines below it give the epoch at which each mean is best.
    """
    means = worst_group_means(config, runs)
    reductions = group_loss_reduction_curves(config, runs)
    reduction = reductions.mean(dim=0)
    epochs = sorted(
        {epoch for epoch in REPORTED_EPOCHS if epoch < config.epochs} | {config.epochs}
    )
    lines = [
        "Worst-group test loss by epoch, means over seeds, and claim c4 "
        "with both runs read at the same epoch.",
        "",
        "| epoch | " + " | ".join(means) + " | c4 |",
        "| --- |" + " --- |" * (len(means) + 1),
    ]
    for epoch in epochs:
        lines.append(
            f"| {epoch} | "
            + " | ".join(f"{float(mean[epoch - 1]):.4g}" for mean in means.values())
            + f" | {_mean_std(reductions[:, epoch - 1].tolist())} |"
        )
    best_loss = ", ".join(
        f"{name} {float(mean.min()):.4g} at epoch {int(mean.argmin()) + 1}"
        for name, mean in means.items()
    )
    best = int(reduction.argmax())
    lines += [
        "",
        f"Lowest mean worst-group test loss: {best_loss}",
        f"Largest mean c4: {float(reduction[best]):.4g} at epoch {best + 1}",
    ]
    return "\n".join(lines)


def curves(runs: dict[Key, Run]) -> list[dict[str, object]]:
    """Return the training curve and test metrics of every run, for plotting.

    The entry `group_loss` holds one list of the 20 subgroup test losses per
    entry of `epochs`.
    """
    return [
        {
            "objective": run.setting.objective,
            "batch_size": run.setting.batch_size,
            "weight_decay": run.setting.weight_decay,
            "seed": run.setting.seed,
            "epochs": run.epochs.tolist(),
            "training_objective": run.objective.tolist(),
            "group_loss": run.group_loss.tolist(),
            "test_loss": run.test_loss,
            "test_accuracy": run.test_accuracy,
            "seconds_per_epoch": run.seconds_per_epoch,
        }
        for run in runs.values()
    ]


def curve_record(config: Config, runs: dict[Key, Run]) -> dict[str, Any]:
    """Return the series behind the figures of the report.

    Args:
        config: Settings of the experiment.
        runs: Every run of the experiment.

    Returns:
        `epochs_to_optimum` maps each objective to its batch sizes, with 0
        for the full batch, and the epochs every seed needs to get within
        2% of the optimal training objective, one list per batch size.
        `epochs` lists the epochs of the comparison runs,
        `worst_group_loss` the mean worst-group test loss of each of them
        at those epochs, and `reduction_mean` and `reduction_std` the mean
        and sample standard deviation over seeds of claim c4 read at each
        epoch.
    """
    reached: dict[str, dict[str, list[Any]]] = {}
    for objective in OBJECTIVES:
        target = (1.0 + TOLERANCE) * optimum(runs, objective)
        sizes = [*config.batch_sizes, FULL_BATCH]
        reached[objective] = {
            "batch_size": sizes,
            "epochs": [
                [
                    epochs_to_reach(run, target)
                    for run in runs.values()
                    if (run.setting.objective, run.setting.batch_size)
                    == (objective, size)
                ]
                for size in sizes
            ],
        }
    reductions = group_loss_reduction_curves(config, runs)
    spread = (
        reductions.std(dim=0)
        if len(config.seeds) > 1
        else torch.zeros_like(reductions[0])
    )
    return {
        "epochs_to_optimum": reached,
        "epochs": list(range(1, config.epochs + 1)),
        "worst_group_loss": {
            name: mean.tolist()
            for name, mean in worst_group_means(config, runs).items()
        },
        "reduction_mean": reductions.mean(dim=0).tolist(),
        "reduction_std": spread.tolist(),
    }


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
    """Parse the command line, run, and write the records, table and curves.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Run the reproduction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args(argv)
    config = load_config(args.config, Config)

    runs = measure(config)
    summary = table(config, runs) + "\n\n" + epoch_table(config, runs)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    (args.table_dir / f"{args.config.stem}_table.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    (args.table_dir / f"{args.config.stem}_curves.json").write_text(
        json.dumps(curves(runs)) + "\n", encoding="utf-8"
    )
    print(summary)
    write_curves(PAPER_ID, curve_record(config, runs), args.output_dir)
    for claim, claim_value in zip(CLAIMS, claim_values(config, runs), strict=True):
        result = make_result(PAPER_ID, claim, config.seeds, claim_value, config.device)
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
