from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from optora_reproductions.config import load_config

from ..config import Config
from ..data import NUM_ASSETS
from ..run import Measurements, best_radius, claim_values, table

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
F64 = torch.float64


def smoke() -> Config:
    return load_config(CONFIGS / "smoke.toml", Config)


def measurements() -> Measurements:
    equal = torch.full((2, NUM_ASSETS), 1.0 / NUM_ASSETS, dtype=F64)
    portfolios = equal.clone()
    portfolios[1, 0] += 0.004
    portfolios[1, 1] -= 0.004
    return Measurements(
        portfolios=portfolios,
        certificate=torch.tensor([-0.5, -0.9], dtype=F64),
        performance=torch.tensor([-1.0, -0.8], dtype=F64),
        lp_portfolios=equal,
        lp_values=torch.tensor([-0.5, -0.9], dtype=F64),
        sweep_certificate=torch.tensor(
            [[-2.0, -1.5, -1.0], [-2.0, -1.5, -1.0]], dtype=F64
        ),
        sweep_performance=torch.tensor(
            [[-0.9, -1.2, -1.1], [-1.1, -1.0, -1.05]], dtype=F64
        ),
        sweep_lp_values=torch.tensor(
            [[-2.0, -1.5, -1.0], [-2.0, -1.5, -1.0]], dtype=F64
        ),
    )


def test_claim_values_follow_their_definitions() -> None:
    config = dataclasses.replace(smoke(), seeds=(0, 1))

    c1, c2, c3 = claim_values(config, measurements())

    assert c1.tolist() == pytest.approx([0.0, 0.004], abs=1e-15)
    assert c2.tolist() == [1.0, 0.0]
    assert c3.tolist() == pytest.approx([0.3, -0.1])


def test_best_radius_minimizes_the_mean_over_simulations() -> None:
    assert best_radius(measurements()) == 1


def test_small_radius_is_found_anywhere_in_the_grid() -> None:
    config = dataclasses.replace(smoke(), radii=(1.0, 1e-2, 1e-4), seeds=(0, 1))
    c3 = claim_values(config, measurements())[2]
    assert c3.tolist() == pytest.approx([0.1, -0.05])


def test_table_has_one_row_per_radius_and_one_for_the_certificate() -> None:
    config = dataclasses.replace(smoke(), seeds=(0, 1))

    lines = table(config, measurements()).splitlines()

    assert len(lines) == 2 + len(config.radii) + 1 + 2
    assert lines[2].startswith(f"| {config.sweep_samples} | 0.0001 | -1.0000 |")
    assert lines[5].startswith(f"| {config.certificate_samples} | 1 | -0.9000 |")
    assert "max |x - x_LP| = 4.0e-03" in lines[-1]
    assert "best sweep radius 0.01" in lines[-1]


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": ()},
        {"radii": ()},
        {"radii": (0.0, 1e-4)},
        {"small_radius": 0.5},
        {"certificate_samples": 0},
        {"sweep_samples": 0},
        {"certificate_radius": 0.0},
        {"far": 0.0},
        {"iterations": 0},
        {"dtype": "float16"},
    ],
)
def test_config_rejects_invalid_settings(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(smoke(), **changes)  # type: ignore[arg-type]
