from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from optora_reproductions.config import load_config

from ..config import Config
from ..run import CellRisks, _chunk_size, claim_values, main, table

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def smoke() -> Config:
    return load_config(CONFIGS / "smoke.toml", Config)


def cell(erm: list[float], robust: list[float]) -> CellRisks:
    as_tensor = torch.tensor
    return CellRisks(
        as_tensor(erm, dtype=torch.float64),
        as_tensor(robust, dtype=torch.float64),
        as_tensor(robust, dtype=torch.float64),
        1,
        1.0,
    )


def test_claim_values_follow_their_definitions() -> None:
    config = smoke()
    risks = {
        (0, 100, 1.0): cell([0.4, 0.2], [0.1, 0.3]),
        (0, 1000, 1.0): cell([0.04, 0.02], [0.0, 1e-12]),
    }

    c1, c2, c3 = claim_values(config, risks)

    assert c1 == [0.75]
    assert c2 == [pytest.approx(5e-13)]
    assert c3 == [pytest.approx(0.2 / 0.3)]


def test_claim_values_need_the_claimed_cells() -> None:
    config = dataclasses.replace(smoke(), sample_sizes=(100,))
    with pytest.raises(ValueError, match="n = 1000"):
        claim_values(config, {})


def test_table_lists_every_cell() -> None:
    config = smoke()
    risks = {
        (0, 100, 1.0): cell([0.4, 0.2], [0.1, 0.3]),
        (0, 1000, 1.0): cell([0.04, 0.02], [0.0, 0.0]),
    }

    lines = table(config, risks).splitlines()

    assert len(lines) == 4
    assert lines[2].startswith("| 100 | 1 | 3.000e-01 |")
    assert lines[3].endswith("| 1.0000 | 0.5000 | 1.0 |")


@pytest.mark.parametrize(
    ("num_samples", "megabytes", "expected"),
    [(100, 512, 4), (10_000, 8, 2), (10_000_000, 1, 1)],
)
def test_chunk_size_respects_the_memory_budget(
    num_samples: int, megabytes: int, expected: int
) -> None:
    config = dataclasses.replace(smoke(), chunk_megabytes=megabytes)
    assert _chunk_size(config, num_samples) == expected


def test_main_writes_the_table(tmp_path: Path) -> None:
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

    assert (tmp_path / "runs" / "smoke_table.md").read_text(encoding="utf-8").count(
        "\n"
    ) == 4
    assert len(list((tmp_path / "results").glob("*.json"))) == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": ()},
        {"replications_per_seed": 0},
        {"sample_sizes": (1,)},
        {"noise_levels": (0.0,)},
        {"delta": 1.0},
        {"descent_iterations": 0},
        {"subgradient_iterations": -1},
        {"dtype": "float16"},
    ],
)
def test_config_rejects_invalid_settings(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(smoke(), **changes)  # type: ignore[arg-type]
