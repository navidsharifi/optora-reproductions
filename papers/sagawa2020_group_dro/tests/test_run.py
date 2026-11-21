from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from .. import run as run_module
from ..claims import CLAIMS
from ..config import Config
from ..model import (
    EARLY_STOPPING,
    ERM,
    GROUP_DRO,
    STANDARD,
    STRONG_L2,
    Evaluation,
    Run,
    Setting,
)
from ..run import (
    Key,
    claim_seeds,
    claim_values,
    curve_record,
    key,
    measure,
    settings,
    table,
)

CONFIG = Config(
    data="synthetic",
    seeds=(0, 1),
    early_stopping_seeds=(0, 1, 2),
    epochs=3,
    batch_size=16,
    micro_batch_size=8,
    eval_interval=1,
    precision="float32",
    pretrained=False,
)
PROPORTIONS = torch.tensor([0.5, 0.1, 0.1, 0.3])


def _run(setting: Setting, train: list[float], test: list[float]) -> Run:
    evaluation = Evaluation(torch.tensor(train), torch.tensor(test), torch.tensor(test))
    return Run(setting, (setting.epochs,), (evaluation,), PROPORTIONS, 0.5)


def _runs() -> dict[Key, Run]:
    """Runs whose final accuracies follow Table 1 of the paper for every seed."""
    accuracies = {
        (STANDARD, ERM): ([1.0, 1.0, 1.0, 1.0], [0.99, 0.8, 0.6, 0.95]),
        (STANDARD, GROUP_DRO): ([1.0, 1.0, 0.99, 1.0], [0.99, 0.85, 0.77, 0.95]),
        (STRONG_L2, ERM): ([1.0, 0.5, 0.36, 0.9], [0.99, 0.5, 0.2, 0.9]),
        (STRONG_L2, GROUP_DRO): ([1.0, 0.98, 0.97, 1.0], [0.97, 0.9, 0.85, 0.96]),
        (EARLY_STOPPING, ERM): ([1.0, 0.1, 0.07, 0.7], [0.99, 0.3, 0.05, 0.8]),
        (EARLY_STOPPING, GROUP_DRO): ([0.9, 0.8, 0.75, 0.8], [0.95, 0.88, 0.85, 0.9]),
    }
    return {
        key(setting): _run(setting, *accuracies[(setting.regime, setting.method)])
        for setting in settings(CONFIG)
    }


def _average(test: list[float]) -> float:
    return float((torch.tensor(test) * PROPORTIONS).sum())


def test_settings_cover_every_run_once_with_the_long_runs_first() -> None:
    every = settings(CONFIG)
    keys = [key(setting) for setting in every]

    assert len(keys) == len(set(keys)) == 2 * 2 * 2 + 2 * 3
    assert [setting.regime for setting in every[:8]] == [STANDARD] * 4 + [STRONG_L2] * 4
    early = [setting for setting in every if setting.regime == EARLY_STOPPING]
    assert all(setting.epochs == 1 for setting in early)
    assert sorted({setting.seed for setting in early}) == [0, 1, 2]
    assert all(
        setting.epochs == 3 for setting in every if setting.regime != EARLY_STOPPING
    )


def test_claim_values_follow_the_definitions() -> None:
    c1, c2, c3, c4, c5 = claim_values(CONFIG, _runs())

    assert c1 == pytest.approx([0.99, 0.99])
    erm_test = [0.99, 0.8, 0.6, 0.95]
    assert c2 == pytest.approx([_average(erm_test) - 0.6] * 2)
    assert c3 == pytest.approx([0.85 - 0.2] * 2)
    assert c4 == pytest.approx([0.85 - 0.05] * 3)
    strong = _average([0.99, 0.5, 0.2, 0.9]) - _average([0.97, 0.9, 0.85, 0.96])
    early = _average([0.99, 0.3, 0.05, 0.8]) - _average([0.95, 0.88, 0.85, 0.9])
    assert c5 == pytest.approx([(strong + early) / 2] * 2)


def test_claim_seeds_pair_each_claim_with_its_regime() -> None:
    seeds = claim_seeds(CONFIG)

    assert len(seeds) == len(CLAIMS)
    assert seeds[3] == CONFIG.early_stopping_seeds
    assert all(seeds[i] == CONFIG.seeds for i in (0, 1, 2, 4))


def test_table_has_one_row_per_run() -> None:
    lines = table(CONFIG, _runs()).splitlines()

    assert len(lines) == 2 + len(settings(CONFIG))
    assert "| strong_l2 | group_dro | 1 | " in lines[2 + 7]


def test_measure_saves_every_run_and_resume_reuses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _runs()

    def fake_train(setting: Setting, *args: object, **kwargs: object) -> Run:
        return saved[key(setting)]

    monkeypatch.setattr(run_module, "train", fake_train)
    measure(CONFIG, tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == len(saved)

    def fail(*args: object, **kwargs: object) -> Run:
        raise AssertionError("a saved run was trained again")

    monkeypatch.setattr(run_module, "train", fail)
    resumed = measure(CONFIG, tmp_path, resume=True)

    assert resumed.keys() == saved.keys()
    for ours, theirs in zip(
        claim_values(CONFIG, resumed), claim_values(CONFIG, saved), strict=True
    ):
        assert ours == pytest.approx(theirs)


def test_resume_trains_a_saved_run_whose_settings_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _runs()
    stale = dataclasses.replace(saved[(STANDARD, ERM, 0)].setting, epochs=99)
    path = tmp_path / "standard_erm_0.json"
    path.write_text(
        json.dumps(_run(stale, [1.0] * 4, [1.0] * 4).to_dict()), encoding="utf-8"
    )
    trained: list[Key] = []

    def fake_train(setting: Setting, *args: object, **kwargs: object) -> Run:
        trained.append(key(setting))
        return saved[key(setting)]

    monkeypatch.setattr(run_module, "train", fake_train)
    measure(CONFIG, tmp_path, resume=True)

    assert (STANDARD, ERM, 0) in trained


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_measure_trains_every_run_on_cuda_in_bfloat16(tmp_path: Path) -> None:
    config = dataclasses.replace(
        CONFIG,
        seeds=(0,),
        early_stopping_seeds=(0,),
        epochs=1,
        precision="bfloat16",
        device="cuda",
    )

    runs = measure(config, tmp_path)

    assert len(runs) == len(settings(config))
    values = claim_values(config, runs)
    assert 0.0 <= values[0][0] <= 1.0
    assert all(len(value) == 1 for value in values)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"data": "celeba"}, "data must be one of"),
        ({"seeds": ()}, "at least one seed"),
        ({"seeds": (0, 7)}, "must include every entry"),
        ({"seeds": (0, 0)}, "must not repeat"),
        ({"epochs": 0}, "epochs must be at least 1"),
        ({"micro_batch_size": 32}, "must not exceed batch_size"),
        ({"precision": "float16"}, "precision must be one of"),
    ],
)
def test_config_rejects_invalid_settings(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(CONFIG, **change)  # type: ignore[arg-type]


def test_curve_record_holds_the_test_accuracies_of_every_seed() -> None:
    record = curve_record(CONFIG, _runs())

    assert set(record) == {STANDARD, STRONG_L2, EARLY_STOPPING}
    assert all(set(methods) == {ERM, GROUP_DRO} for methods in record.values())
    standard = record[STANDARD][ERM]
    assert standard["epochs"] == [CONFIG.epochs]
    assert standard["worst_test"] == [[pytest.approx(0.6)]] * len(CONFIG.seeds)
    assert standard["average_test"] == [
        [pytest.approx(_average([0.99, 0.8, 0.6, 0.95]))]
    ] * len(CONFIG.seeds)
    early = record[EARLY_STOPPING][GROUP_DRO]
    assert early["epochs"] == [1]
    assert early["worst_test"] == [[pytest.approx(0.85)]] * len(
        CONFIG.early_stopping_seeds
    )
