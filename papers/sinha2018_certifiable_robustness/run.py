"""Run the MNIST and synthetic experiments and write one record per claim.

Usage::

    python -m papers.sinha2018_certifiable_robustness.run \
        --config papers/sinha2018_certifiable_robustness/configs/smoke.toml

Every finished seed is saved as JSON under
`runs/sinha2018_certifiable_robustness/<config stem>/`. With `--resume`, a
later call reuses the saved seeds whose settings match.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from optora_reproductions.config import load_config
from optora_reproductions.results import ClaimResult, make_result, write_result
from optora_reproductions.seeding import seed_everything

from . import PAPER_ID
from .claims import CLAIMS
from .config import Config
from .data import Dataset, Split, load_mnist, random_images, two_dimensional
from .evaluate import WorstCase, accuracy, loss_at, pgm_error, worst_case_curve
from .model import (
    ERM,
    FGM,
    IFGM,
    METHODS,
    PENALTY,
    PGM,
    SYNTHETIC_PENALTY,
    SYNTHETIC_STEPS,
    WRM,
    WRM_STEPS,
    mnist_network,
    one_hot,
    smoothed,
    synthetic_network,
    train_mnist,
    train_synthetic,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs" / PAPER_ID
L2_RANGE = 0.26
LINF_RANGE = 0.2
LINF_CLAIM = 0.1
SCALE_RANGE = (1.0, 100.0)
SYNTHETIC_PENALTIES = (50.0, 0.5)


@dataclass(frozen=True)
class MethodResult:
    r"""Evaluation of one trained MNIST model on the test set.

    Attributes:
        accuracy: Clean test accuracy.
        seconds_per_epoch: Training time per epoch.
        l2_errors: Pairs of radius and error under the $\ell_2$ PGM attack.
        linf_errors: Pairs of radius and error under the $\ell_\infty$ PGM
            attack.
        l2_claim_error: Error under the $\ell_2$ attack at the training
            radius $\sqrt{\hat\rho_n(\theta_{\mathrm{WRM}})}$.
        linf_claim_error: Error under the $\ell_\infty$ attack at
            $0.1\, C_\infty$.
        worst_cases: Worst case for every penalty of the sweep.
        at_penalty: Worst case at the training penalty $\gamma$.
    """

    accuracy: float
    seconds_per_epoch: float
    l2_errors: list[tuple[float, float]]
    linf_errors: list[tuple[float, float]]
    l2_claim_error: float
    linf_claim_error: float
    worst_cases: list[WorstCase]
    at_penalty: WorstCase


@dataclass(frozen=True)
class SyntheticResult:
    r"""Certificate and test worst case of the synthetic WRM model.

    Attributes:
        accuracy: Clean test accuracy.
        certificate: Worst case on the training set at $\gamma = 2$.
        worst_cases: Worst case on the test set for every penalty.
    """

    accuracy: float
    certificate: WorstCase
    worst_cases: list[WorstCase]


@dataclass(frozen=True)
class SeedResult:
    r"""Everything measured for one seed.

    Attributes:
        seed: The seed.
        settings: Settings of the configuration that change the result.
        l2_scale: $C_2 = \mathbb E \lVert X \rVert_2$ on the training set.
        linf_scale: $C_\infty = \mathbb E \lVert X \rVert_\infty$.
        certificate: Worst case of the WRM model on the training set at its
            training penalty.
        radius: $\ell_2$ radius of FGM, IFGM and PGM.
        methods: Evaluation of every MNIST model by method.
        synthetic: Result of the synthetic experiment.
    """

    seed: int
    settings: dict[str, Any]
    l2_scale: float
    linf_scale: float
    certificate: WorstCase
    radius: float
    methods: dict[str, MethodResult]
    synthetic: SyntheticResult

    def to_dict(self) -> dict[str, Any]:
        """Return the result as plain JSON types."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> SeedResult:
        """Rebuild a result from `to_dict`."""

        def curve(points: list[dict[str, float]]) -> list[WorstCase]:
            return [WorstCase(**point) for point in points]

        def pairs(values: list[list[float]]) -> list[tuple[float, float]]:
            return [(radius, error) for radius, error in values]

        methods = {
            name: MethodResult(
                accuracy=method["accuracy"],
                seconds_per_epoch=method["seconds_per_epoch"],
                l2_errors=pairs(method["l2_errors"]),
                linf_errors=pairs(method["linf_errors"]),
                l2_claim_error=method["l2_claim_error"],
                linf_claim_error=method["linf_claim_error"],
                worst_cases=curve(method["worst_cases"]),
                at_penalty=WorstCase(**method["at_penalty"]),
            )
            for name, method in record["methods"].items()
        }
        synthetic = record["synthetic"]
        return cls(
            seed=record["seed"],
            settings=record["settings"],
            l2_scale=record["l2_scale"],
            linf_scale=record["linf_scale"],
            certificate=WorstCase(**record["certificate"]),
            radius=record["radius"],
            methods=methods,
            synthetic=SyntheticResult(
                accuracy=synthetic["accuracy"],
                certificate=WorstCase(**synthetic["certificate"]),
                worst_cases=curve(synthetic["worst_cases"]),
            ),
        )


def settings_of(config: Config) -> dict[str, Any]:
    """Return the settings that a saved seed must share to be reused."""
    record = dataclasses.asdict(config)
    for name in ("seeds", "eval_batch_size", "device"):
        record.pop(name)
    return record


def load_data(config: Config) -> Dataset:
    """Load the dataset that the configuration names."""
    if config.data == "mnist":
        return load_mnist()
    return random_images()


@contextmanager
def default_precision() -> Iterator[None]:
    """Fix PyTorch's default precision for the duration of a run.

    cuDNN convolutions may use TensorFloat-32 and matrix products stay in
    float32, the defaults of PyTorch, set explicitly so that no earlier
    change of these global flags alters the run. With the deterministic
    kernels that `seed_everything` requests, float32 convolutions are about
    four times slower on the RTX 3050, which would stretch the full run from
    about 4 to 17 hours.
    """
    cudnn, matmul = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cuda.matmul.allow_tf32 = matmul


def _evaluate(
    config: Config,
    model: nn.Module,
    test: Split,
    seconds_per_epoch: float,
    radius: float,
    scales: tuple[float, float],
) -> MethodResult:
    targets = one_hot(test.labels)
    l2_scale, linf_scale = scales
    batch = config.eval_batch_size

    def errors(norm: float, radii: np.ndarray) -> list[tuple[float, float]]:
        return [
            (float(r), pgm_error(model, test.inputs, targets, float(r), norm, batch))
            for r in radii
        ]

    l2_radii = l2_scale * np.linspace(0.0, L2_RANGE, config.num_l2_radii)
    linf_radii = linf_scale * np.linspace(0.0, LINF_RANGE, config.num_linf_radii)
    penalties = l2_scale / np.geomspace(*SCALE_RANGE, config.num_penalties)
    curve = worst_case_curve(
        model, test.inputs, targets, [float(p) for p in penalties], WRM_STEPS, batch
    )
    (at_penalty,) = worst_case_curve(
        model, test.inputs, targets, [PENALTY], WRM_STEPS, batch
    )
    return MethodResult(
        accuracy=accuracy(model, test.inputs, test.labels, batch),
        seconds_per_epoch=seconds_per_epoch,
        l2_errors=errors(2.0, l2_radii),
        linf_errors=errors(float("inf"), linf_radii),
        l2_claim_error=pgm_error(model, test.inputs, targets, radius, 2.0, batch),
        linf_claim_error=pgm_error(
            model, test.inputs, targets, LINF_CLAIM * linf_scale, float("inf"), batch
        ),
        worst_cases=curve,
        at_penalty=at_penalty,
    )


def _synthetic(config: Config, seed: int, device: torch.device) -> SyntheticResult:
    generator = torch.Generator().manual_seed(seed)
    train = two_dimensional(config.synthetic_size, generator).to(device)
    test = two_dimensional(config.synthetic_size, generator).to(device)
    torch.manual_seed(seed)
    model = synthetic_network().to(device)
    train_synthetic(
        model,
        train,
        generator,
        epochs=config.synthetic_epochs,
        batch_size=config.synthetic_batch_size,
    )
    batch = config.eval_batch_size
    (certificate,) = worst_case_curve(
        model,
        train.inputs,
        one_hot(train.labels, 2),
        [SYNTHETIC_PENALTY],
        SYNTHETIC_STEPS,
        batch,
    )
    penalties = np.geomspace(*SYNTHETIC_PENALTIES, config.num_penalties)
    curve = worst_case_curve(
        model,
        test.inputs,
        one_hot(test.labels, 2),
        [float(p) for p in penalties],
        SYNTHETIC_STEPS,
        batch,
    )
    return SyntheticResult(
        accuracy(model, test.inputs, test.labels, batch), certificate, curve
    )


def measure_seed(
    config: Config, data: Dataset, seed: int, device: torch.device
) -> SeedResult:
    """Train and evaluate the five MNIST models and the synthetic model.

    WRM is trained first, since the radius of FGM, IFGM and PGM is the
    square root of the robustness level it reaches on the training set
    (Eq. 19). Every MNIST model of a seed starts from the same weights and
    visits the training set in the same order.

    Args:
        config: Settings of the run.
        data: MNIST or its stand-in, on `device`.
        seed: Seed of the run.
        device: Device of the models.

    Returns:
        The evaluation of every model of the seed.
    """
    generator = seed_everything(seed)
    train, test = data.train, data.test
    order = torch.randperm(len(train.labels), generator=generator).to(device)
    flat = train.inputs.flatten(1)
    scales = (
        float(flat.norm(dim=1).mean()),
        float(flat.abs().amax(dim=1).mean()),
    )
    methods: dict[str, MethodResult] = {}
    certificate = WorstCase(PENALTY, 0.0, 0.0)
    radius = 0.0
    for method in (WRM, ERM, FGM, IFGM, PGM):
        torch.manual_seed(seed)
        model = mnist_network().to(device)
        seconds = train_mnist(
            model,
            method,
            train,
            order,
            epochs=config.epochs,
            batch_size=config.batch_size,
            radius=radius,
        )
        if method == WRM:
            (certificate,) = worst_case_curve(
                model,
                train.inputs,
                smoothed(train.labels),
                [PENALTY],
                WRM_STEPS,
                config.eval_batch_size,
            )
            radius = math.sqrt(certificate.distance)
        methods[method] = _evaluate(config, model, test, seconds, radius, scales)
        print(f"seed {seed}: {method} done in {seconds:.1f} s per epoch", flush=True)
    return SeedResult(
        seed=seed,
        settings=settings_of(config),
        l2_scale=scales[0],
        linf_scale=scales[1],
        certificate=certificate,
        radius=radius,
        methods={method: methods[method] for method in METHODS},
        synthetic=_synthetic(config, seed, device),
    )


def measure(
    config: Config, cache_dir: Path | None = None, resume: bool = False
) -> dict[int, SeedResult]:
    """Run every seed and save it in `cache_dir`.

    Args:
        config: Settings of the run.
        cache_dir: Folder of the saved seeds, or `None` to keep nothing.
        resume: Whether to reuse the seeds already saved in `cache_dir`
            whose settings match instead of running them again.

    Returns:
        The result of every seed.
    """
    device = torch.device(config.device)
    data = load_data(config).to(device)
    results: dict[int, SeedResult] = {}
    with default_precision():
        for seed in config.seeds:
            path = None if cache_dir is None else cache_dir / f"seed_{seed}.json"
            if resume and path is not None and path.exists():
                saved = SeedResult.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
                if saved.settings == settings_of(config):
                    results[seed] = saved
                    continue
            results[seed] = measure_seed(config, data, seed, device)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(results[seed].to_dict()) + "\n", encoding="utf-8"
                )
    return results


def claim_values(result: SeedResult) -> list[float]:
    """Compute the value of every claim for one seed, in the order of `CLAIMS`."""
    synthetic = result.synthetic
    c1 = synthetic.certificate.loss - loss_at(
        synthetic.worst_cases, synthetic.certificate.distance
    )
    c2 = min(method.accuracy for method in result.methods.values())
    wrm = result.methods[WRM]
    c3 = result.certificate.loss - loss_at(wrm.worst_cases, result.certificate.distance)
    c4 = result.methods[ERM].l2_claim_error - wrm.l2_claim_error
    c5 = (
        min(result.methods[name].linf_claim_error for name in (FGM, IFGM, PGM))
        - wrm.linf_claim_error
    )
    c6 = math.log10(result.methods[ERM].at_penalty.distance / wrm.at_penalty.distance)
    return [c1, c2, c3, c4, c5, c6]


def per_claim(results: dict[int, SeedResult]) -> list[list[float]]:
    """Return the values of every claim over the seeds, in the order of `CLAIMS`."""
    rows = [claim_values(results[seed]) for seed in sorted(results)]
    return [list(values) for values in zip(*rows, strict=True)]


def table(results: dict[int, SeedResult]) -> str:
    """Return a Markdown table of every MNIST model and the certificates."""
    lines = [
        "| seed | method | clean acc | l2 PGM error at radius | linf PGM error at 0.1 "
        "| rho_test at gamma | s per epoch |",
        "| " + " | ".join(["---"] * 7) + " |",
    ]
    for seed in sorted(results):
        for name, method in results[seed].methods.items():
            cells = [
                str(seed),
                name,
                f"{100 * method.accuracy:.2f}",
                f"{100 * method.l2_claim_error:.2f}",
                f"{100 * method.linf_claim_error:.2f}",
                f"{method.at_penalty.distance:.4g}",
                f"{method.seconds_per_epoch:.3g}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "| seed | rho_hat_n | radius | certificate | test worst case at rho_hat_n "
        "| synthetic rho_hat_n | synthetic certificate | synthetic test worst case |",
        "| " + " | ".join(["---"] * 8) + " |",
    ]
    for seed in sorted(results):
        result = results[seed]
        synthetic = result.synthetic
        test = loss_at(result.methods[WRM].worst_cases, result.certificate.distance)
        cells = [
            str(seed),
            f"{result.certificate.distance:.4f}",
            f"{result.radius:.4f}",
            f"{result.certificate.loss:.4f}",
            f"{test:.4f}",
            f"{synthetic.certificate.distance:.4f}",
            f"{synthetic.certificate.loss:.4f}",
            f"{loss_at(synthetic.worst_cases, synthetic.certificate.distance):.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(config: Config, cache_dir: Path | None = None) -> list[ClaimResult]:
    """Measure and build one record per claim.

    Args:
        config: Settings of the run.
        cache_dir: Folder of the saved seeds, or `None` to keep nothing.

    Returns:
        One record per entry of `CLAIMS`, in the same order.
    """
    results = measure(config, cache_dir)
    seeds = tuple(sorted(results))
    return [
        make_result(PAPER_ID, claim, seeds, values, config.device)
        for claim, values in zip(CLAIMS, per_claim(results), strict=True)
    ]


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line, run, and write the records and the table.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Run the reproduction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--table-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--resume", action="store_true", help="reuse the seeds saved by an earlier call"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, Config)

    results = measure(config, args.table_dir / args.config.stem, args.resume)
    summary = table(results)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    (args.table_dir / f"{args.config.stem}_table.md").write_text(
        summary + "\n", encoding="utf-8"
    )
    print(summary)
    seeds = tuple(sorted(results))
    for claim, values in zip(CLAIMS, per_claim(results), strict=True):
        result = make_result(PAPER_ID, claim, seeds, values, config.device)
        path = write_result(result, args.output_dir)
        status = "within" if result.within_tolerance else "outside"
        print(
            f"{claim.claim_id}: mean {result.mean:.6g}, std {result.std:.3g} "
            f"({status} tolerance) -> {path}"
        )


if __name__ == "__main__":
    main()
