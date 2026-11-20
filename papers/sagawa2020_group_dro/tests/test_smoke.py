"""Smoke test of the reproduction, run in CI on CPU."""

from __future__ import annotations

from pathlib import Path

from optora_reproductions.config import load_config
from optora_reproductions.results import read_results

from ..claims import CLAIMS
from ..config import Config
from ..run import claim_seeds, main, run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_every_config_loads() -> None:
    for path in sorted(CONFIGS.glob("*.toml")):
        load_config(path, Config)


def test_smoke_run_measures_every_claim() -> None:
    config = load_config(CONFIGS / "smoke.toml", Config)

    results = run(config)

    assert [result.claim for result in results] == list(CLAIMS)
    assert [result.seeds for result in results] == claim_seeds(config)


def test_main_writes_one_record_per_claim(tmp_path: Path) -> None:
    main(
        [
            "--config",
            str(CONFIGS / "smoke.toml"),
            "--output-dir",
            str(tmp_path / "results"),
            "--table-dir",
            str(tmp_path / "runs"),
        ]
    )

    records = read_results(tmp_path / "results")
    assert [record.claim for record in records] == list(CLAIMS)
    assert (tmp_path / "runs" / "smoke_table.md").exists()
    assert len(list((tmp_path / "runs" / "smoke").glob("*.json"))) == 8
