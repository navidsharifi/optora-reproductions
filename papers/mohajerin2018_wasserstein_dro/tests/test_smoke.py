"""Smoke test of the reproduction, run in CI on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from optora_reproductions.config import load_config
from optora_reproductions.figures import THEMES, render, stored_digest
from optora_reproductions.results import read_curves, read_results

from .. import PAPER_ID
from ..claims import CLAIMS
from ..config import Config
from ..figures import FIGURES
from ..run import main, run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Folder holding the `results` and `runs` of one smoke run of `main`."""
    directory = tmp_path_factory.mktemp("smoke")
    main(
        [
            "--config",
            str(CONFIGS / "smoke.toml"),
            "--output-dir",
            str(directory / "results"),
            "--table-dir",
            str(directory / "runs"),
        ]
    )
    return directory


def test_every_config_loads() -> None:
    for path in sorted(CONFIGS.glob("*.toml")):
        load_config(path, Config)


def test_smoke_run_measures_every_claim() -> None:
    config = load_config(CONFIGS / "smoke.toml", Config)

    results = run(config)

    assert [result.claim for result in results] == list(CLAIMS)
    assert all(result.seeds == config.seeds for result in results)


def test_main_writes_one_record_per_claim(outputs: Path) -> None:
    records = read_results(outputs / "results")
    assert [record.claim for record in records] == list(CLAIMS)


def test_every_figure_draws_from_the_smoke_run(outputs: Path) -> None:
    curves = read_curves(PAPER_ID, outputs / "results")
    assert curves is not None
    results = read_results(outputs / "results")

    names = [spec.name for spec in FIGURES]
    assert len(set(names)) == len(names)
    for spec in FIGURES:
        for theme in THEMES:
            assert stored_digest(render(spec, theme, curves, results)) is not None
