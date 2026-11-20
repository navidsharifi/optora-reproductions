"""Run the Waterbirds experiment and write one result record per claim.

Usage::

    python -m papers.sagawa2020_group_dro.run \
        --config papers/sagawa2020_group_dro/configs/smoke.toml

Every finished training run is saved as JSON under
`runs/sagawa2020_group_dro/<config stem>/`. With `--resume`, a later call
reuses the saved runs whose settings match, so an interrupted full run
continues at the first missing run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

import torch

from optora_reproductions.config import load_config
from optora_reproductions.results import ClaimResult, make_result, write_result
from optora_reproductions.seeding import seed_everything

from . import PAPER_ID
from .claims import CLAIMS
from .config import Config
from .data import GROUP_NAMES, Waterbirds, load_waterbirds, synthetic_waterbirds
from .model import (
    EARLY_STOPPING,
    ERM,
    GROUP_DRO,
    METHODS,
    REGIMES,
    STANDARD,
    STRONG_L2,
    Run,
    Setting,
    average_accuracy,
    train,
    worst_group_accuracy,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs" / PAPER_ID
MEMORY_FRACTION = 0.9

Key = tuple[str, str, int]


def settings(config: Config) -> list[Setting]:
    """Return every training run of the configuration, the long ones first."""
    every: list[Setting] = []
    for regime in (STANDARD, STRONG_L2, EARLY_STOPPING):
        epochs = REGIMES[regime].epochs or config.epochs
        seeds = (
            config.early_stopping_seeds if regime == EARLY_STOPPING else config.seeds
        )
        every.extend(
            Setting(regime, method, seed, epochs)
            for method in METHODS
            for seed in seeds
        )
    return every


def key(setting: Setting) -> Key:
    """Identify a run by regime, method and seed."""
    return (setting.regime, setting.method, setting.seed)


def load_data(config: Config) -> Waterbirds:
    """Load the dataset that the configuration names."""
    if config.data == "waterbirds":
        return load_waterbirds()
    return synthetic_waterbirds()


def measure(
    config: Config, cache_dir: Path | None = None, resume: bool = False
) -> dict[Key, Run]:
    """Train every run and save it in `cache_dir`.

    Args:
        config: Settings of the run.
        cache_dir: Folder of the saved runs, or `None` to keep nothing.
        resume: Whether to reuse the runs already saved in `cache_dir` whose
            settings match instead of training them again.

    Returns:
        Every run by its key.
    """
    device = torch.device(config.device)
    if device.type == "cuda":
        # Fail with an out-of-memory error instead of spilling into host
        # memory, which the Windows driver otherwise does silently.
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, index)
    data = load_data(config)
    runs: dict[Key, Run] = {}
    for setting in settings(config):
        name = "_".join(map(str, key(setting))) + ".json"
        path = None if cache_dir is None else cache_dir / name
        if resume and path is not None and path.exists():
            saved = Run.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if saved.setting == setting:
                runs[key(setting)] = saved
                continue
        generator = seed_everything(setting.seed)
        run_ = train(
            setting,
            data,
            generator,
            batch_size=config.batch_size,
            micro_batch_size=config.micro_batch_size,
            eval_interval=config.eval_interval,
            precision=config.precision,
            pretrained=config.pretrained,
            device=device,
        )
        runs[key(setting)] = run_
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(run_.to_dict()) + "\n", encoding="utf-8")
        print(
            f"finished {key(setting)} in {run_.seconds_per_epoch:.1f} s per epoch",
            flush=True,
        )
    return runs


def _worst_test(run_: Run) -> float:
    return worst_group_accuracy(run_.final.test)


def _average_test(run_: Run) -> float:
    return average_accuracy(run_.final.test, run_.train_proportions)


def claim_values(config: Config, runs: dict[Key, Run]) -> list[list[float]]:
    """Compute the per-seed values of every claim, in the order of `CLAIMS`."""

    def get(regime: str, method: str, seed: int) -> Run:
        return runs[(regime, method, seed)]

    c1 = [
        min(
            worst_group_accuracy(get(STANDARD, method, seed).final.train)
            for method in METHODS
        )
        for seed in config.seeds
    ]
    c2 = [
        _average_test(get(STANDARD, ERM, seed)) - _worst_test(get(STANDARD, ERM, seed))
        for seed in config.seeds
    ]
    c3 = [
        _worst_test(get(STRONG_L2, GROUP_DRO, seed))
        - _worst_test(get(STRONG_L2, ERM, seed))
        for seed in config.seeds
    ]
    c4 = [
        _worst_test(get(EARLY_STOPPING, GROUP_DRO, seed))
        - _worst_test(get(EARLY_STOPPING, ERM, seed))
        for seed in config.early_stopping_seeds
    ]
    c5 = [
        statistics.fmean(
            _average_test(get(regime, ERM, seed))
            - _average_test(get(regime, GROUP_DRO, seed))
            for regime in (STRONG_L2, EARLY_STOPPING)
        )
        for seed in config.seeds
    ]
    return [c1, c2, c3, c4, c5]


def claim_seeds(config: Config) -> list[tuple[int, ...]]:
    """Return the seeds behind each claim, in the order of `CLAIMS`."""
    return [
        config.seeds,
        config.seeds,
        config.seeds,
        config.early_stopping_seeds,
        config.seeds,
    ]


def _percent(value: float) -> str:
    return f"{100 * value:.1f}"


def table(config: Config, runs: dict[Key, Run]) -> str:
    """Return a Markdown table of the final accuracies of every run, in percent."""
    header = (
        "| regime | method | seed | train avg | train worst | test avg | test worst "
        "| " + " | ".join(f"test {name}" for name in GROUP_NAMES) + " | s per epoch |"
    )
    lines = [header, "| " + " | ".join(["---"] * (8 + len(GROUP_NAMES))) + " |"]
    for setting in settings(config):
        run_ = runs[key(setting)]
        final = run_.final
        cells = [
            setting.regime,
            setting.method,
            str(setting.seed),
            _percent(average_accuracy(final.train, run_.train_proportions)),
            _percent(worst_group_accuracy(final.train)),
            _percent(_average_test(run_)),
            _percent(_worst_test(run_)),
            *(_percent(float(value)) for value in final.test),
            f"{run_.seconds_per_epoch:.3g}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(config: Config, cache_dir: Path | None = None) -> list[ClaimResult]:
    """Measure and build one record per claim.

    Args:
        config: Settings of the run.
        cache_dir: Folder of the saved runs, or `None` to keep nothing.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    runs = measure(config, cache_dir)
    return [
        make_result(PAPER_ID, claim, seeds, values, config.device)
        for claim, seeds, values in zip(
            CLAIMS, claim_seeds(config), claim_values(config, runs), strict=True
        )
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
    parser.add_argument(
        "--resume", action="store_true", help="reuse the runs saved by an earlier call"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, Config)

    runs = measure(config, args.table_dir / args.config.stem, args.resume)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    summary = table(config, runs)
    (args.table_dir / f"{args.config.stem}_table.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    curves = [runs[key(setting)].to_dict() for setting in settings(config)]
    (args.table_dir / f"{args.config.stem}_curves.json").write_text(
        json.dumps(curves) + "\n", encoding="utf-8"
    )
    print(summary)
    values = claim_values(config, runs)
    for claim, seeds, claim_value in zip(
        CLAIMS, claim_seeds(config), values, strict=True
    ):
        result = make_result(PAPER_ID, claim, seeds, claim_value, config.device)
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
