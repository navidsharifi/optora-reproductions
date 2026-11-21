"""Smoke test of the reproduction, run in CI on CPU."""

from __future__ import annotations

from pathlib import Path

from optora_reproductions.config import load_config
from optora_reproductions.results import read_results

from ..claims import CLAIMS
from ..config import Config
from ..run import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_every_config_loads() -> None:
    for path in sorted(CONFIGS.glob("*.toml")):
        load_config(path, Config)


def test_main_writes_one_record_per_claim(tmp_path: Path) -> None:
    config = load_config(CONFIGS / "smoke.toml", Config)

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
    assert all(record.seeds == config.seeds for record in records)
    assert (tmp_path / "runs" / "smoke_table.md").exists()
    saved = sorted(path.name for path in (tmp_path / "runs" / "smoke").glob("*.json"))
    assert saved == [f"seed_{seed}.json" for seed in config.seeds]
