"""Smoke test of the reproduction, run in CI on CPU."""

from __future__ import annotations

from pathlib import Path

from optora_reproductions.config import load_config
from optora_reproductions.results import read_results

from ..claims import CLAIMS
from ..config import Config
from ..run import main, run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_every_config_loads() -> None:
    for path in sorted(CONFIGS.glob("*.toml")):
        load_config(path, Config)


def test_smoke_run_measures_every_claim() -> None:
    config = load_config(CONFIGS / "smoke.toml", Config)

    results = run(config)

    assert [result.claim for result in results] == list(CLAIMS)
    assert all(result.seeds == config.seeds for result in results)


def test_main_writes_one_record_per_claim(tmp_path: Path) -> None:
    main(
        [
            "--config",
            str(CONFIGS / "smoke.toml"),
            "--output-dir",
            str(tmp_path),
            "--table-dir",
            str(tmp_path / "runs"),
        ]
    )

    records = read_results(tmp_path)
    assert [record.claim for record in records] == list(CLAIMS)
