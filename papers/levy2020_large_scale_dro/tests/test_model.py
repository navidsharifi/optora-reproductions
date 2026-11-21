from __future__ import annotations

import dataclasses

import pytest
import torch

from ..data import synthetic_digits
from ..model import (
    ALPHA,
    ERM,
    ERM_STEP_SIZES,
    FULL_BATCH,
    NUM_GROUPS,
    OBJECTIVES,
    PENALTY,
    RADIUS,
    STEP_SIZES,
    WEIGHT_DECAY,
    Run,
    Setting,
    averaging_weight,
    epochs_to_reach,
    robust_loss,
    train,
)

CPU = torch.device("cpu")


TrainSplit = tuple[torch.Tensor, torch.Tensor]
TestSplit = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _splits(
    device: torch.device = CPU, sizes: tuple[int, int] = (120, 13)
) -> tuple[TrainSplit, TestSplit]:
    digits = synthetic_digits(sizes=sizes, dimension=6)
    train_split = (digits.train.features.to(device), digits.train.targets.to(device))
    test_split = (
        digits.test.features.to(device),
        digits.test.targets.to(device),
        digits.test.groups.to(device),
    )
    return train_split, test_split


def _setting(**changes: object) -> Setting:
    setting = Setting("cvar", 50, 0.01, WEIGHT_DECAY, 3, 1, 1)
    return dataclasses.replace(setting, **changes)  # type: ignore[arg-type]


def test_step_sizes_are_the_authors_values() -> None:
    assert {key for key in STEP_SIZES} == {
        (objective, size)
        for objective in OBJECTIVES
        for size in (50, 500, 5000, FULL_BATCH)
    }
    assert STEP_SIZES[("cvar", 500)] == 1e-3
    assert STEP_SIZES[("chi_square", 5000)] == 2e-2
    assert STEP_SIZES[("penalty", FULL_BATCH)] == 1e-2
    assert ERM_STEP_SIZES == {1e-4: 5e-2, 1e-3: 2e-2}


def test_paper_parameters_in_optora_units() -> None:
    # The paper's chi-square divergence is half of Optora's.
    assert (ALPHA, RADIUS, PENALTY) == (0.02, 2 * 1.0, 0.05 / 2)


def test_cvar_loss_is_the_mean_of_the_upper_tail() -> None:
    loss = torch.arange(100.0)
    value = robust_loss("cvar", 100, CPU)(loss)
    assert value.item() == pytest.approx(98.5)


def test_chi_square_loss_has_the_variance_form_inside_the_ball() -> None:
    # A Bernoulli(0.2) loss keeps every scenario weighted up to radius
    # Var / (mean - min)^2 = 4, above the radius 2 used here.
    loss = torch.cat([torch.ones(10), torch.zeros(40)]).double()
    value = robust_loss("chi_square", 50, CPU)(loss)
    variance = torch.var(loss, unbiased=False)
    expected = loss.mean() + (RADIUS * variance).sqrt()
    assert value.item() == pytest.approx(expected.item())


def test_penalized_loss_has_the_variance_form_without_truncation() -> None:
    # sup_q E_q[l] - penalty * D(q || p) = E_p[l] + Var_p(l) / (4 penalty).
    loss = torch.linspace(1.0, 1.01, 40, dtype=torch.float64)
    value = robust_loss("penalty", 40, CPU)(loss)
    variance = torch.var(loss, unbiased=False)
    expected = loss.mean() + variance / (4 * PENALTY)
    assert value.item() == pytest.approx(expected.item())


def test_erm_loss_is_the_mean() -> None:
    loss = torch.rand(3, 7)
    assert torch.allclose(robust_loss(ERM, 7, CPU)(loss), loss.mean(dim=-1))


def test_unknown_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        robust_loss("kl", 5, CPU)


@pytest.mark.parametrize("objective", [*OBJECTIVES, ERM])
def test_training_lowers_the_objective(objective: str) -> None:
    train_split, test_split = _splits()

    result = train(
        _setting(objective=objective, epochs=6), train_split, test_split, False
    )

    assert result.objective.shape == (6,)
    assert result.objective[-1] < result.objective[0]
    assert result.group_loss.shape == (6, NUM_GROUPS)
    assert 0.0 <= result.test_accuracy <= 1.0


def test_training_is_reproducible_for_a_seed() -> None:
    train_split, test_split = _splits()
    first = train(_setting(), train_split, test_split, False)
    second = train(_setting(), train_split, test_split, False)
    other = train(_setting(seed=2), train_split, test_split, False)

    assert torch.equal(first.objective, second.objective)
    assert not torch.equal(first.objective, other.objective)


def test_a_batch_larger_than_the_set_is_the_whole_set() -> None:
    train_split, test_split = _splits()
    whole = train(_setting(batch_size=5000), train_split, test_split, False)
    full = train(
        _setting(batch_size=133, step_size=0.01), train_split, test_split, False
    )
    assert torch.allclose(whole.objective, full.objective)


def test_full_batch_is_evaluated_at_the_interval_and_not_averaged() -> None:
    train_split, test_split = _splits()
    setting = _setting(batch_size=FULL_BATCH, epochs=6, eval_interval=3)

    result = train(setting, train_split, test_split, False)

    assert result.epochs.tolist() == [3, 6]
    assert result.group_loss.shape == (2, NUM_GROUPS)
    assert not setting.averaged
    assert _setting().averaged


def test_group_loss_is_recorded_at_every_evaluation() -> None:
    train_split, test_split = _splits()
    short = train(_setting(epochs=2), train_split, test_split, False)
    long = train(_setting(epochs=4), train_split, test_split, False)

    # The small synthetic test set leaves some subgroups empty, hence nan.
    assert torch.allclose(long.group_loss[:2], short.group_loss, equal_nan=True)
    assert not torch.allclose(long.group_loss[1], long.group_loss[3], equal_nan=True)


@pytest.mark.parametrize(
    ("epochs", "eval_interval", "message"),
    [
        (0, 1, "must be positive"),
        (3, 0, "must be positive"),
        (5, 2, "multiple of eval_interval"),
    ],
)
def test_setting_needs_the_last_epoch_evaluated(
    epochs: int, eval_interval: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _setting(epochs=epochs, eval_interval=eval_interval)


def test_averaging_weight_follows_shamir_and_zhang() -> None:
    # The first step gives 0.2 x_0 + 0.8 x_1, and the weight tends to 4 / t.
    assert averaging_weight(2) == pytest.approx(0.8)
    assert averaging_weight(3) == pytest.approx(4 / 6)
    assert averaging_weight(10**6) == pytest.approx(4e-6, rel=1e-5)


def test_minibatch_runs_report_the_averaged_iterate() -> None:
    train_split, test_split = _splits()
    setting = _setting(batch_size=5000, epochs=2, objective=ERM)
    averaged = train(setting, train_split, test_split, False)
    plain = train(
        dataclasses.replace(setting, batch_size=FULL_BATCH),
        train_split,
        test_split,
        False,
    )
    assert not torch.allclose(averaged.objective, plain.objective)


def _run(values: list[float], epochs: int = 4) -> Run:
    return Run(
        setting=_setting(epochs=epochs),
        epochs=torch.arange(1, len(values) + 1),
        objective=torch.tensor(values),
        group_loss=torch.zeros(len(values), NUM_GROUPS),
        test_loss=0.0,
        test_accuracy=0.0,
        seconds_per_epoch=0.0,
    )


def test_epochs_to_reach_counts_the_first_epoch_at_or_below_the_target() -> None:
    run = _run([5.0, 3.0, 2.0, 2.0])
    assert epochs_to_reach(run, 3.0) == 2
    assert epochs_to_reach(run, 2.0) == 3


def test_a_run_that_never_reaches_counts_one_epoch_over_its_budget() -> None:
    assert epochs_to_reach(_run([5.0, 4.0, 3.0, 2.5]), 2.0) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_cuda_graph_replay_matches_eager_training(objective: str) -> None:
    device = torch.device("cuda")
    train_split, test_split = _splits(device)
    setting = _setting(objective=objective, epochs=3)

    eager = train(setting, train_split, test_split, False)
    graphed = train(setting, train_split, test_split, True)

    assert torch.allclose(eager.objective, graphed.objective, atol=1e-6)
    assert torch.allclose(
        eager.group_loss, graphed.group_loss, atol=1e-6, equal_nan=True
    )
