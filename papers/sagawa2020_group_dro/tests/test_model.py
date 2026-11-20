from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ..data import NUM_GROUPS, Split, Waterbirds, group_counts, synthetic_waterbirds
from ..model import (
    EARLY_STOPPING,
    ERM,
    GROUP_DRO,
    REGIMES,
    ROBUST_STEP_SIZE,
    STANDARD,
    STRONG_L2,
    Evaluation,
    Inputs,
    Run,
    Setting,
    _Trainer,
    average_accuracy,
    build_model,
    epoch_order,
    evaluate,
    frozen_running_statistics,
    train,
    worst_group_accuracy,
)

CPU = torch.device("cpu")


def _split(n: int = 40, size: int = 4, seed: int = 0) -> Split:
    generator = torch.Generator().manual_seed(seed)
    groups = torch.randint(0, NUM_GROUPS, (n,), generator=generator)
    images = torch.randint(0, 256, (n, 3, size, size), generator=generator).to(
        torch.uint8
    )
    return Split(images, groups // 2, groups)


def _linear(size: int = 4) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * size * size, 2))


def _batch_norm_model(size: int = 4) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.Flatten(),
        nn.Linear(4 * size * size, 2),
    )


def _authors_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    adv_probs: torch.Tensor,
    images: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    """One step of `LossComputer.compute_robust_loss` in the authors' `loss.py`."""
    losses = F.cross_entropy(model(images), targets, reduction="none")
    group_map = (groups == torch.arange(NUM_GROUPS).unsqueeze(1)).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()
    group_loss = (group_map @ losses.view(-1)) / group_denom
    adv_probs = adv_probs * torch.exp(ROBUST_STEP_SIZE * group_loss.data)
    adv_probs = adv_probs / adv_probs.sum()
    optimizer.zero_grad()
    torch.autograd.backward(group_loss @ adv_probs)
    optimizer.step()
    return adv_probs


def test_regimes_follow_appendix_c2() -> None:
    assert REGIMES[STANDARD].learning_rate == 1e-3
    assert REGIMES[STANDARD].weight_decay == 1e-4
    assert REGIMES[STRONG_L2].learning_rate == 1e-5
    assert REGIMES[STRONG_L2].weight_decay == 1.0
    assert REGIMES[EARLY_STOPPING].epochs == 1
    assert REGIMES[EARLY_STOPPING].weight_decay == REGIMES[STANDARD].weight_decay


@pytest.mark.parametrize("micro_batch_size", [3, 8, 16, 32])
def test_group_dro_step_matches_the_authors_full_batch_step(
    micro_batch_size: int,
) -> None:
    split = _split()
    inputs = Inputs("float32", CPU)
    model = _linear()
    reference = copy.deepcopy(model)
    trainer = _Trainer(
        model, Setting(STANDARD, GROUP_DRO, 0, 1), split, micro_batch_size, inputs
    )
    optimizer = torch.optim.SGD(
        reference.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
    )
    adv_probs = torch.full((NUM_GROUPS,), 1.0 / NUM_GROUPS)

    for index in torch.split(
        torch.randperm(40, generator=torch.Generator().manual_seed(1)), 16
    ):
        trainer.step(index)
        adv_probs = _authors_step(
            reference,
            optimizer,
            adv_probs,
            inputs(split.images[index]),
            split.targets[index],
            split.groups[index],
        )

    torch.testing.assert_close(
        trainer.adversary.weights, adv_probs, rtol=1e-6, atol=1e-7
    )
    for ours, theirs in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-7)


def test_group_dro_leaves_the_weight_of_an_absent_group_unnormalized_unchanged() -> (
    None
):
    split = _split()
    trainer = _Trainer(
        _linear(), Setting(STANDARD, GROUP_DRO, 0, 1), split, 4, Inputs("float32", CPU)
    )
    index = torch.nonzero(split.groups != 3).flatten()[:12]

    trainer.step(index)

    weights = trainer.adversary.weights
    # q_g after the step is q_g exp(eta l_g) / Z with l_3 = 0, so q_3 Z = 1 / 4.
    others = torch.arange(NUM_GROUPS) != 3
    assert float(weights[3]) < 0.25
    assert float(weights[others].sum() + weights[3]) == pytest.approx(1.0)


@pytest.mark.parametrize("micro_batch_size", [5, 16])
def test_erm_step_with_micro_batches_matches_one_full_batch(
    micro_batch_size: int,
) -> None:
    split = _split()
    inputs = Inputs("float32", CPU)
    model = _linear()
    reference = copy.deepcopy(model)
    trainer = _Trainer(
        model, Setting(STANDARD, ERM, 0, 1), split, micro_batch_size, inputs
    )
    optimizer = torch.optim.SGD(
        reference.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4
    )
    index = torch.arange(16)

    trainer.step(index)
    optimizer.zero_grad()
    torch.autograd.backward(
        F.cross_entropy(reference(inputs(split.images[index])), split.targets[index])
    )
    optimizer.step()

    for ours, theirs in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-7)


def test_group_dro_updates_batch_norm_statistics_once_per_micro_batch() -> None:
    split = _split()
    inputs = Inputs("float32", CPU)
    model = _batch_norm_model()
    reference = copy.deepcopy(model).train()
    trainer = _Trainer(model, Setting(STANDARD, GROUP_DRO, 0, 1), split, 8, inputs)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    index = torch.arange(16)

    trainer.step(index)
    with torch.no_grad():
        for part in torch.split(index, 8):
            reference(inputs(split.images[part]))

    for ours, theirs in zip(model.buffers(), reference.buffers(), strict=True):
        torch.testing.assert_close(ours, theirs)


def test_frozen_running_statistics_normalize_with_batch_statistics() -> None:
    model = _batch_norm_model().train()
    before = [buffer.clone() for buffer in model.buffers()]
    images = torch.randn(8, 3, 4, 4)
    expected = copy.deepcopy(model)(images)

    with torch.no_grad(), frozen_running_statistics(model):
        output = model(images)

    torch.testing.assert_close(output, expected)
    batch_norm = model[1]
    assert isinstance(batch_norm, nn.BatchNorm2d)
    torch.testing.assert_close(batch_norm.running_mean, before[0])
    torch.testing.assert_close(batch_norm.running_var, before[1])
    assert batch_norm.momentum == 0.1


def test_frozen_running_statistics_restore_the_momentum_after_an_error() -> None:
    model = _batch_norm_model()
    with pytest.raises(RuntimeError), frozen_running_statistics(model):
        raise RuntimeError
    batch_norm = model[1]
    assert isinstance(batch_norm, nn.BatchNorm2d)
    assert batch_norm.momentum == 0.1


def test_erm_epoch_is_a_permutation() -> None:
    groups = _split().groups
    order = epoch_order(ERM, groups, torch.Generator().manual_seed(0))
    assert sorted(order.tolist()) == list(range(len(groups)))


def test_group_dro_epoch_draws_every_group_equally_often() -> None:
    groups = torch.repeat_interleave(
        torch.arange(NUM_GROUPS), torch.tensor([700, 40, 10, 250])
    )
    draws = torch.cat(
        [
            epoch_order(GROUP_DRO, groups, torch.Generator().manual_seed(s))
            for s in range(20)
        ]
    )
    frequency = group_counts(groups[draws]).double() / len(draws)

    assert len(draws) == 20 * len(groups)
    torch.testing.assert_close(
        frequency, torch.full((4,), 0.25, dtype=torch.float64), atol=0.01, rtol=0
    )


def test_inputs_standardize_like_torchvision() -> None:
    from torchvision.transforms import Normalize

    images = torch.randint(0, 256, (2, 3, 5, 5)).to(torch.uint8)
    expected = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(
        images.float() / 255
    )

    torch.testing.assert_close(Inputs("float32", CPU)(images), expected)
    channels_last = Inputs("bfloat16", CPU)(images)
    assert channels_last.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(channels_last, expected)


def test_evaluate_reports_the_accuracy_of_each_group() -> None:
    split = _split()
    data = Waterbirds(split, split, split)
    model = _linear()
    inputs = Inputs("float32", CPU)
    predictions = model(inputs(split.images)).argmax(dim=1)
    correct = (predictions == split.targets).float()
    expected = torch.stack(
        [correct[split.groups == g].mean() for g in range(NUM_GROUPS)]
    )

    evaluation = evaluate(model, data, inputs)

    torch.testing.assert_close(evaluation.test, expected)
    assert model.training


def test_average_accuracy_on_the_training_proportions_is_the_plain_accuracy() -> None:
    split = _split()
    correct = (torch.arange(40) % 3 == 0).float()
    group_accuracy = torch.stack(
        [correct[split.groups == g].mean() for g in range(NUM_GROUPS)]
    )
    proportions = group_counts(split.groups).float() / 40

    assert average_accuracy(group_accuracy, proportions) == pytest.approx(
        float(correct.mean())
    )
    assert worst_group_accuracy(group_accuracy) == float(group_accuracy.min())


def test_build_model_has_two_outputs() -> None:
    model = build_model(pretrained=False).eval()
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 2)


def test_train_evaluates_at_every_interval_and_after_the_last_epoch() -> None:
    data = synthetic_waterbirds(size=4)
    run = train(
        Setting(STANDARD, GROUP_DRO, 0, 5),
        data,
        torch.Generator().manual_seed(0),
        batch_size=16,
        micro_batch_size=8,
        eval_interval=2,
        precision="float32",
        pretrained=False,
        device=CPU,
        model_factory=lambda _: _linear(),
    )

    assert run.epochs == (2, 4, 5)
    assert run.final.test.shape == (NUM_GROUPS,)
    torch.testing.assert_close(run.train_proportions.sum(), torch.tensor(1.0))
    assert run.seconds_per_epoch > 0


def test_run_survives_a_json_round_trip() -> None:
    evaluation = Evaluation(torch.rand(4), torch.rand(4), torch.rand(4))
    run = Run(
        Setting(STRONG_L2, ERM, 3, 300),
        (10, 20),
        (evaluation, evaluation),
        torch.rand(4),
        1.5,
    )

    restored = Run.from_dict(run.to_dict())

    assert restored.setting == run.setting
    assert restored.epochs == run.epochs
    torch.testing.assert_close(restored.final.validation, evaluation.validation)
    torch.testing.assert_close(restored.train_proportions, run.train_proportions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_group_dro_step_runs_on_cuda(precision: str) -> None:
    device = torch.device("cuda")
    split = _split()
    split = Split(
        split.images.to(device), split.targets.to(device), split.groups.to(device)
    )
    inputs = Inputs(precision, device)
    model = _batch_norm_model().to(device).to(memory_format=inputs.memory_format)
    trainer = _Trainer(model, Setting(STANDARD, GROUP_DRO, 0, 1), split, 8, inputs)

    trainer.step(torch.arange(16, device=device))

    assert torch.isfinite(trainer.adversary.weights).all()
    assert trainer.adversary.weights.device.type == "cuda"
