from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from ..claims import CLAIMS
from ..config import Config
from ..model import ERM, FULL_BATCH, NUM_GROUPS, OBJECTIVES, WEIGHT_DECAY, Run, Setting
from ..run import (
    Key,
    claim_values,
    curve_record,
    epoch_table,
    group_loss_reduction_curves,
    group_loss_reductions,
    key,
    optimum,
    settings,
    speed_ups,
    table,
    worst_group_curve,
    worst_group_loss,
)

CONFIG = Config(
    data="synthetic",
    seeds=(1, 2),
    batch_sizes=(50, 500),
    epochs=4,
    full_batch_epochs=30,
    full_batch_eval_interval=10,
)


def _run(
    setting: Setting, values: list[float], worst: float | list[float] = 1.0
) -> Run:
    """Return a run whose subgroup 3 has the loss `worst` at every evaluation."""
    group_loss = torch.full((len(values), NUM_GROUPS), 0.5)
    group_loss[:, 3] = torch.tensor(worst)
    return Run(
        setting=setting,
        epochs=torch.arange(1, len(values) + 1) * setting.eval_interval,
        objective=torch.tensor(values),
        group_loss=group_loss,
        test_loss=0.5,
        test_accuracy=0.9,
        seconds_per_epoch=0.1,
    )


WORST_GROUP = {"cvar": 1.2, "chi_square": 1.28, "penalty": 1.36}


def _runs() -> dict[Key, Run]:
    """Runs whose optimum is 1.0, so the target is 1.02."""
    curves = {
        (50, 1): [3.0, 1.5, 1.01, 1.0],
        (50, 2): [3.0, 2.0, 1.5, 1.2],
        (500, 1): [3.0, 1.02, 1.0, 1.0],
        (500, 2): [3.0, 1.5, 1.01, 1.0],
    }
    runs: dict[Key, Run] = {}
    for setting in settings(CONFIG):
        if setting.objective == ERM:
            worst = 2.0 if setting.weight_decay == 1e-4 else 1.6
            runs[key(setting)] = _run(setting, [1.0] * 4, worst)
        elif setting.batch_size == FULL_BATCH:
            runs[key(setting)] = _run(setting, [5.0, 1.1, 1.0])
        else:
            runs[key(setting)] = _run(
                setting,
                curves[(setting.batch_size, setting.seed)],
                WORST_GROUP[setting.objective],
            )
    return runs


def test_settings_cover_every_run_once() -> None:
    every = settings(CONFIG)
    keys = [key(setting) for setting in every]

    assert len(keys) == len(set(keys))
    assert len(every) == 3 * 2 * 2 + 3 + 2 * 2
    full = [setting for setting in every if setting.batch_size == FULL_BATCH]
    assert [setting.objective for setting in full] == list(OBJECTIVES)
    assert all(setting.epochs == 30 and setting.eval_interval == 10 for setting in full)
    others = [setting for setting in every if setting.batch_size != FULL_BATCH]
    assert all(setting.epochs == 4 and setting.eval_interval == 1 for setting in others)


def test_optimum_is_the_lowest_value_of_any_run_of_the_objective() -> None:
    assert optimum(_runs(), "cvar") == 1.0


def test_speed_up_divides_full_batch_epochs_by_the_best_minibatch() -> None:
    # The full batch reaches 1.02 at epoch 30. Seed 1 reaches it at epoch 2
    # with batch 500, and seed 2 at epoch 3 with batch 500.
    assert speed_ups(CONFIG, _runs(), "cvar") == [15.0, 10.0]


def test_a_seed_that_never_reaches_gets_a_lower_bound() -> None:
    runs = _runs()
    for size in (50, 500):
        stuck = runs[("cvar", size, WEIGHT_DECAY, 2)]
        runs[("cvar", size, WEIGHT_DECAY, 2)] = dataclasses.replace(
            stuck, objective=torch.full((4,), 9.0)
        )
    assert speed_ups(CONFIG, runs, "cvar") == [15.0, 30.0 / 5.0]


def test_worst_group_loss_skips_empty_groups() -> None:
    run = _run(settings(CONFIG)[0], [1.0], worst=0.7)
    run.group_loss[:, 5] = float("nan")
    assert worst_group_loss(run) == pytest.approx(0.7)


def test_worst_group_curve_follows_every_evaluation() -> None:
    run = _run(settings(CONFIG)[0], [1.0] * 4, worst=[0.4, 0.9, 0.7, 0.6])
    run.group_loss[2, 3] = float("nan")

    curve = worst_group_curve(run)

    assert curve.dtype == torch.float64
    assert curve.tolist() == pytest.approx([0.5, 0.9, 0.5, 0.6])
    assert worst_group_loss(run) == pytest.approx(0.6)


def test_group_loss_reduction_against_the_better_erm() -> None:
    # The better ERM worst group is 1.6, so the reductions are 0.25, 0.2 and
    # 0.15, whose mean is 0.2.
    assert group_loss_reductions(CONFIG, _runs()) == pytest.approx([0.2, 0.2])


def test_group_loss_reduction_curves_compare_runs_at_the_same_epoch() -> None:
    runs = _runs()
    for seed in CONFIG.seeds:
        erm = runs[(ERM, 500, 1e-4, seed)]
        runs[(ERM, 500, 1e-4, seed)] = _run(
            erm.setting, [1.0] * 4, worst=[1.5, 1.6, 2.0, 2.0]
        )
        cvar = runs[("cvar", 500, WEIGHT_DECAY, seed)]
        runs[("cvar", 500, WEIGHT_DECAY, seed)] = _run(
            cvar.setting, [1.0] * 4, worst=[1.5, 1.2, 1.2, 1.2]
        )

    curves = group_loss_reduction_curves(CONFIG, runs)

    # At epoch 1 the better ERM is the run with mu = 1e-4 at 1.5, so CVaR
    # gains nothing, chi-square 1 - 1.28 / 1.5 and penalty 1 - 1.36 / 1.5.
    # From epoch 2 on the better ERM is 1.6 and the mean is 0.2.
    first = (0.0 + (1 - 1.28 / 1.5) + (1 - 1.36 / 1.5)) / 3
    assert curves.shape == (2, 4)
    assert curves[:, 0].tolist() == pytest.approx([first, first])
    assert curves[:, 1:].flatten().tolist() == pytest.approx([0.2] * 6)
    assert group_loss_reductions(CONFIG, runs) == curves[:, -1].tolist()


def test_claim_values_follow_the_order_of_the_claims() -> None:
    values = claim_values(CONFIG, _runs())

    assert len(values) == len(CLAIMS)
    assert values[0] == values[1] == values[2] == [15.0, 10.0]
    assert values[3] == pytest.approx([0.2, 0.2])


def test_table_has_one_row_per_objective_and_batch_size() -> None:
    lines = table(CONFIG, _runs()).splitlines()

    rows = [line for line in lines if line.startswith("| ") and "---" not in line]
    assert len(rows) == 1 + 3 * 2 + 3 + 2
    assert lines[-1].startswith("Optimal training objective")


def test_epoch_table_lists_the_reported_and_the_last_epochs() -> None:
    lines = epoch_table(CONFIG, _runs()).splitlines()

    rows = [line for line in lines if line.startswith("| ") and "---" not in line]
    assert rows[0].split(" | ")[1:-1] == [
        "cvar",
        "chi_square",
        "penalty",
        "erm mu=0.0001",
        "erm mu=0.001",
    ]
    assert [row.split(" | ")[0] for row in rows[1:]] == ["| 1", "| 2", "| 4"]
    assert rows[-1].endswith("| 0.2 +- 0 |")
    assert lines[-2].startswith("Lowest mean worst-group test loss: cvar 1.2")
    assert lines[-1] == "Largest mean c4: 0.2 at epoch 1"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"data": "imagenet"}, "data must be one of"),
        ({"seeds": ()}, "at least one seed"),
        ({"batch_sizes": (10,)}, "batch_sizes must be"),
        ({"comparison_batch_size": 5000}, "comparison_batch_size"),
        ({"epochs": 0}, "epochs must be at least 1"),
        ({"full_batch_epochs": 25}, "multiple of"),
        ({"erm_batch_size": 0}, "must be positive"),
        ({"cuda_graphs": True}, "needs a CUDA device"),
    ],
)
def test_config_rejects_invalid_settings(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(CONFIG, **change)  # type: ignore[arg-type]


def test_curve_record_holds_the_series_of_both_figures() -> None:
    runs = _runs()

    record = curve_record(CONFIG, runs)

    assert record["epochs"] == [1, 2, 3, 4]
    assert list(record["epochs_to_optimum"]) == list(OBJECTIVES)
    for objective in OBJECTIVES:
        reached = record["epochs_to_optimum"][objective]
        assert reached["batch_size"] == [50, 500, FULL_BATCH]
        assert reached["epochs"] == [[3, 5], [2, 3], [30]]
    assert list(record["worst_group_loss"]) == [
        *OBJECTIVES,
        "erm mu=0.0001",
        "erm mu=0.001",
    ]
    for objective in OBJECTIVES:
        assert record["worst_group_loss"][objective] == pytest.approx(
            [WORST_GROUP[objective]] * 4
        )
    reductions = group_loss_reduction_curves(CONFIG, runs)
    assert record["reduction_mean"] == pytest.approx(reductions.mean(dim=0).tolist())
    assert record["reduction_mean"] == pytest.approx([0.2] * 4)
    assert record["reduction_std"] == pytest.approx([0.0] * 4, abs=1e-12)


def test_curve_record_of_a_single_seed_has_no_spread_and_is_finite() -> None:
    config = dataclasses.replace(CONFIG, seeds=(1,))

    record = curve_record(config, _runs())

    assert record["reduction_std"] == [0.0] * 4
    json.dumps(record, allow_nan=False)
