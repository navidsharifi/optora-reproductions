"""Figures of the report, drawn from the curve record and the claim records."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme, end_label
from optora_reproductions.results import ClaimResult

REGIME_LABELS = {
    "standard": "standard",
    "strong_l2": "strong $\\ell_2$ penalty",
    "early_stopping": "early stopping",
}
METHOD_LABELS = {"group_dro": "group DRO", "erm": "ERM"}
"""Methods in the order of their colours."""
METRIC_TITLES = {
    "average_test": "Average test accuracy",
    "worst_test": "Worst-group test accuracy",
}


def _draw_test_accuracy(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the final test accuracies of every regime and method.

    The two panels are the average and the worst-group columns of Table 1
    of the paper, in percent. A bar is the mean over the seeds of its
    regime and the line on it spans one sample standard deviation each way.
    """
    regimes = [regime for regime in REGIME_LABELS if regime in curves]
    figure = Figure(figsize=(8.4, 3.6), layout="constrained")
    panels = figure.subplots(1, 2, sharey=True)
    width = 0.34
    for axes, (metric, title) in zip(panels, METRIC_TITLES.items(), strict=True):
        for slot, (method, label) in enumerate(METHOD_LABELS.items()):
            for group, regime in enumerate(regimes):
                final = [100.0 * seed[-1] for seed in curves[regime][method][metric]]
                mean = statistics.fmean(final)
                position = group + (slot - 0.5) * (width + 0.03)
                axes.bar(
                    position,
                    mean,
                    width,
                    color=theme.series[slot],
                    label=label if group == 0 else None,
                )
                if len(final) > 1:
                    spread = statistics.stdev(final)
                    axes.plot(
                        [position, position],
                        [mean - spread, mean + spread],
                        color=theme.ink,
                        linewidth=1.2,
                    )
                axes.annotate(
                    f"{mean:.1f}",
                    (position, mean),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=theme.secondary,
                )
        axes.set_xticks(
            range(len(regimes)), [REGIME_LABELS[regime] for regime in regimes]
        )
        axes.grid(visible=False, axis="x")
        axes.set_ylim(0.0, 112.0)
        axes.set_yticks(range(0, 101, 20))
        axes.set_title(title)
    panels[0].set_ylabel("accuracy (%)")
    figure.legend(*panels[1].get_legend_handles_labels(), loc="outside right upper")
    return figure


def _draw_worst_group_by_epoch(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the worst-group test accuracy at every evaluation of the long runs.

    One panel per regime that is evaluated more than once, in percent, mean
    over the seeds of the regime.
    """
    regimes = [
        regime
        for regime in REGIME_LABELS
        if regime in curves and len(curves[regime]["erm"]["epochs"]) > 1
    ]
    figure = Figure(figsize=(8.4, 3.4), layout="constrained")
    panels = figure.subplots(1, max(len(regimes), 1), sharey=True, squeeze=False)[0]
    for axes, regime in zip(panels, regimes, strict=False):
        for slot, (method, label) in enumerate(METHOD_LABELS.items()):
            record = curves[regime][method]
            accuracy = [
                100.0 * statistics.fmean(point)
                for point in zip(*record["worst_test"], strict=True)
            ]
            axes.plot(record["epochs"], accuracy, color=theme.series[slot], label=label)
            end_label(
                axes, theme, record["epochs"][-1], accuracy[-1], f"{accuracy[-1]:.1f}"
            )
        axes.set_xlim(0.0, 1.14 * record["epochs"][-1])
        axes.set_title(REGIME_LABELS[regime].capitalize())
        axes.set_xlabel("epoch")
    panels[0].set_ylabel("worst-group test accuracy (%)")
    panels[0].set_ylim(0.0, 100.0)
    figure.legend(*panels[0].get_legend_handles_labels(), loc="outside right upper")
    return figure


FIGURES = (
    FigureSpec("test_accuracy", _draw_test_accuracy),
    FigureSpec("worst_group_by_epoch", _draw_worst_group_by_epoch),
)
