"""Checks of the template's own claim, kept outside the folder that is copied."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from optora_reproductions.config import load_config
from papers._template import PAPER_ID
from papers._template.config import Config
from papers._template.run import run

CONFIGS = Path(__file__).resolve().parents[2] / "papers" / "_template" / "configs"


def smoke() -> Config:
    return load_config(CONFIGS / "smoke.toml", Config)


def test_paper_id_comes_from_package_name() -> None:
    assert PAPER_ID == "_template"


@pytest.mark.parametrize("radius_fraction", [1e-6, 0.5, 1.0])
def test_identity_holds_in_the_interior_regime(radius_fraction: float) -> None:
    config = dataclasses.replace(smoke(), radius_fraction=radius_fraction)
    assert all(result.within_tolerance for result in run(config))


def test_full_config_reproduces_the_claim() -> None:
    assert all(
        result.within_tolerance
        for result in run(load_config(CONFIGS / "full.toml", Config))
    )


def test_batched_run_equals_per_seed_runs() -> None:
    config = dataclasses.replace(smoke(), seeds=(3, 4))
    batched = run(config)[0].values
    single = tuple(
        run(dataclasses.replace(config, seeds=(seed,)))[0].values[0] for seed in (3, 4)
    )
    assert batched == pytest.approx(single, abs=1e-14)


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": ()},
        {"num_samples": 1},
        {"radius_fraction": 0.0},
        {"radius_fraction": 1.5},
        {"dtype": "float16"},
    ],
)
def test_config_rejects_invalid_settings(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(smoke(), **changes)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_runs_on_cuda() -> None:
    results = run(dataclasses.replace(smoke(), device="cuda"))
    assert all(result.within_tolerance for result in results)
