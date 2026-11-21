from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from optora_reproductions.config import load_config

from ..config import Config
from ..run import Measurement, claim_values, curve_record, table

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def smoke() -> Config:
    return load_config(CONFIGS / "smoke.toml", Config)


def test_claim_values_follow_their_definitions() -> None:
    config = dataclasses.replace(smoke(), seeds=(0, 1), state_counts=(10, 20))
    measurements = {
        (0, 10): Measurement(1e-16, 3e-16, 1e-3, 1e-5),
        (0, 20): Measurement(2e-16, 1e-16, 2e-3, 2e-6),
        (1, 10): Measurement(5e-16, 0.0, 1e-3, 1e-5),
        (1, 20): Measurement(1e-16, 1e-16, 1e-3, 1e-4),
    }

    c1, c2 = claim_values(config, measurements)

    assert c1 == [3e-16, 5e-16]
    assert c2 == [pytest.approx(3.0), pytest.approx(1.0)]


def test_timing_size_outside_the_grid_does_not_enter_the_error() -> None:
    config = dataclasses.replace(smoke(), state_counts=(10,), timing_states=30)
    measurements = {
        (0, 10): Measurement(1e-16, 0.0, 1e-3, 1e-4),
        (0, 30): Measurement(1.0, 0.0, 1e-2, 1e-4),
    }

    c1, c2 = claim_values(config, measurements)

    assert c1 == [1e-16]
    assert c2 == [pytest.approx(math.log10(100.0))]


def test_table_lists_every_size() -> None:
    config = smoke()
    measurements = {(0, 20): Measurement(1e-16, 2e-16, 2e-3, 1e-4)}

    lines = table(config, measurements).splitlines()

    assert len(lines) == 3
    assert lines[2] == "| 20 | 1.00e-16 | 2.00e-16 | 2.000 | 0.100 | 20.0 |"


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": ()},
        {"state_counts": (1,)},
        {"budgets": (-0.1,)},
        {"budgets": (2.5,)},
        {"discount": 1.0},
        {"timing_repeats": 0},
        {"dtype": "float16"},
    ],
)
def test_config_rejects_invalid_settings(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(smoke(), **changes)  # type: ignore[arg-type]


def test_curve_record_takes_the_worst_error_and_the_mean_time_over_seeds() -> None:
    config = dataclasses.replace(smoke(), seeds=(0, 1), state_counts=(10, 20))
    measurements = {
        (0, 20): Measurement(2e-16, 1e-16, 2e-3, 2e-6),
        (0, 10): Measurement(1e-16, 3e-16, 1e-3, 1e-5),
        (1, 10): Measurement(5e-16, 0.0, 1e-3, 1e-5),
        (1, 20): Measurement(1e-16, 1e-16, 1e-3, 1e-4),
    }

    record = curve_record(config, measurements)

    assert record["num_states"] == [10.0, 20.0]
    assert record["random_error"] == [5e-16, 2e-16]
    assert record["inventory_error"] == [3e-16, 1e-16]
    assert record["lp_seconds"] == pytest.approx([1e-3, 1.5e-3])
    assert record["optora_seconds"] == pytest.approx([1e-5, 5.1e-5])
