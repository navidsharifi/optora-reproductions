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
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Results folder written by one smoke run of `main`."""
    directory = tmp_path_factory.mktemp("results")
    main(["--config", str(CONFIGS / "smoke.toml"), "--output-dir", str(directory)])
    return directory


def test_every_config_loads() -> None:
    for path in sorted(CONFIGS.glob("*.toml")):
        load_config(path, Config)


def test_smoke_run_measures_every_claim() -> None:
    config = load_config(CONFIGS / "smoke.toml", Config)

    results = run(config)

    assert [result.claim for result in results] == list(CLAIMS)
    assert all(result.seeds == config.seeds for result in results)


def test_main_writes_one_record_per_claim(output_dir: Path) -> None:
    records = read_results(output_dir)
    assert [record.claim for record in records] == list(CLAIMS)


def test_every_figure_draws_from_the_smoke_run(output_dir: Path) -> None:
    curves = read_curves(PAPER_ID, output_dir)
    assert curves is not None
    results = read_results(output_dir)

    names = [spec.name for spec in FIGURES]
    assert len(set(names)) == len(names)
    for spec in FIGURES:
        for theme in THEMES:
            assert stored_digest(render(spec, theme, curves, results)) is not None
