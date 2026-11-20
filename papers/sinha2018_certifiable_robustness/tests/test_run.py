from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from .. import run as run_module
from ..claims import CLAIMS
from ..config import Config
from ..data import Dataset
from ..evaluate import WorstCase
from ..model import ERM, FGM, IFGM, METHODS, PGM, WRM
from ..run import (
    MethodResult,
    SeedResult,
    SyntheticResult,
    claim_values,
    default_precision,
    measure,
    per_claim,
    settings_of,
    table,
)

CONFIG = Config(
    data="random",
    seeds=(0, 1),
    epochs=1,
    batch_size=16,
    synthetic_size=60,
    synthetic_epochs=1,
    synthetic_batch_size=30,
    num_penalties=3,
    num_l2_radii=2,
    num_linf_radii=2,
    eval_batch_size=32,
)
CURVE = [WorstCase(4.0, 0.1, 0.2), WorstCase(1.0, 0.3, 0.6), WorstCase(0.2, 0.9, 1.4)]
ACCURACY = {ERM: 0.991, FGM: 0.992, IFGM: 0.993, PGM: 0.994, WRM: 0.995}
L2_ERROR = {ERM: 0.11, FGM: 0.03, IFGM: 0.04, PGM: 0.025, WRM: 0.02}
LINF_ERROR = {ERM: 0.9, FGM: 0.3, IFGM: 0.6, PGM: 0.28, WRM: 0.15}
DISTANCE = {ERM: 160.0, FGM: 2.0, IFGM: 20.0, PGM: 1.5, WRM: 0.5}


def _result(seed: int, shift: float = 0.0) -> SeedResult:
    methods = {
        name: MethodResult(
            accuracy=ACCURACY[name],
            seconds_per_epoch=1.0,
            l2_errors=[(0.0, 0.01), (2.4, 0.5)],
            linf_errors=[(0.0, 0.01), (0.2, 0.9)],
            l2_claim_error=L2_ERROR[name] + shift,
            linf_claim_error=LINF_ERROR[name],
            worst_cases=CURVE,
            at_penalty=WorstCase(1 / 2.6, DISTANCE[name], 1.0),
        )
        for name in METHODS
    }
    synthetic = SyntheticResult(0.99, WorstCase(2.0, 0.2, 0.43), CURVE)
    return SeedResult(
        seed=seed,
        settings=settings_of(CONFIG),
        l2_scale=9.21,
        linf_scale=1.0,
        certificate=WorstCase(1 / 2.6, 0.2, 0.8),
        radius=math.sqrt(0.2),
        methods=methods,
        synthetic=synthetic,
    )


def test_claim_values_follow_the_definitions() -> None:
    c1, c2, c3, c4, c5, c6 = claim_values(_result(0))

    assert c1 == pytest.approx(0.43 - 0.4)
    assert c2 == pytest.approx(0.991)
    assert c3 == pytest.approx(0.8 - 0.4)
    assert c4 == pytest.approx(0.11 - 0.02)
    assert c5 == pytest.approx(0.28 - 0.15)
    assert c6 == pytest.approx(math.log10(160.0 / 0.5))


def test_per_claim_groups_the_seeds_of_every_claim_in_seed_order() -> None:
    values = per_claim({1: _result(1, shift=0.01), 0: _result(0)})

    assert len(values) == len(CLAIMS)
    assert values[3] == pytest.approx([0.09, 0.09])
    assert values[1] == pytest.approx([0.991, 0.991])


def test_a_seed_result_survives_json() -> None:
    result = _result(3)
    restored = SeedResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored == result


def test_settings_ignore_the_seeds_the_device_and_the_evaluation_batch() -> None:
    changed = dataclasses.replace(
        CONFIG, seeds=(7,), device="cuda", eval_batch_size=1000
    )
    assert settings_of(changed) == settings_of(CONFIG)
    assert settings_of(dataclasses.replace(CONFIG, epochs=2)) != settings_of(CONFIG)


def test_table_has_one_row_per_model_and_per_seed() -> None:
    lines = table({0: _result(0), 1: _result(1)}).splitlines()

    assert sum(line.startswith("| 0 | ") for line in lines) == len(METHODS) + 1
    assert "| 1 | wrm | 99.50 | 2.00 | 15.00 | 0.5 | 1 |" in lines


def test_measure_saves_every_seed_and_resume_reuses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(
        config: Config, data: Dataset, seed: int, device: torch.device
    ) -> SeedResult:
        return _result(seed)

    monkeypatch.setattr(run_module, "measure_seed", fake)
    first = measure(CONFIG, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["seed_0.json", "seed_1.json"]

    def fail(*args: object) -> SeedResult:
        raise AssertionError("a saved seed was run again")

    monkeypatch.setattr(run_module, "measure_seed", fail)
    assert measure(CONFIG, tmp_path, resume=True) == first


def test_resume_runs_a_saved_seed_whose_settings_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = dataclasses.replace(_result(0), settings={"epochs": 99})
    (tmp_path / "seed_0.json").write_text(json.dumps(stale.to_dict()), encoding="utf-8")
    measured: list[int] = []

    def fake(
        config: Config, data: Dataset, seed: int, device: torch.device
    ) -> SeedResult:
        measured.append(seed)
        return _result(seed)

    monkeypatch.setattr(run_module, "measure_seed", fake)
    measure(CONFIG, tmp_path, resume=True)

    assert measured == [0, 1]


def test_default_precision_sets_the_pytorch_defaults_and_restores_the_flags() -> None:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = True
    with default_precision():
        assert torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_runs_every_model_on_cuda() -> None:
    config = dataclasses.replace(CONFIG, seeds=(0,), device="cuda")

    (result,) = measure(config).values()

    assert list(result.methods) == list(METHODS)
    assert all(len(values) == 1 for values in per_claim({0: result}))
    assert 0 < result.radius == pytest.approx(math.sqrt(result.certificate.distance))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"data": "cifar10"}, "data must be one of"),
        ({"seeds": ()}, "at least one seed"),
        ({"seeds": (0, 0)}, "must not repeat"),
        ({"epochs": 0}, "epochs must be at least 1"),
        ({"synthetic_size": 0}, "synthetic_size must be at least 1"),
        ({"num_penalties": 1}, "num_penalties must be at least 2"),
    ],
)
def test_config_rejects_invalid_settings(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(CONFIG, **change)  # type: ignore[arg-type]


def test_run_builds_one_record_per_claim_over_the_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(
        config: Config, data: Dataset, seed: int, device: torch.device
    ) -> SeedResult:
        return _result(seed)

    monkeypatch.setattr(run_module, "measure_seed", fake)

    records = run_module.run(CONFIG)

    assert [record.claim for record in records] == list(CLAIMS)
    assert all(record.seeds == (0, 1) for record in records)
    assert records[5].mean == pytest.approx(math.log10(320.0))
