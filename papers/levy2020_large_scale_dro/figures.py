"""Figures of the report, drawn from the curve record and the claim records."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme, end_label, paper_band
from optora_reproductions.results import ClaimResult

OBJECTIVE_LABELS = {
    "cvar": "CVaR",
    "chi_square": "constrained $\\chi^2$",
    "penalty": "penalized $\\chi^2$",
}
"""Objectives in the order of claims c1 to c3."""


def _draw_epochs_to_optimum(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the epochs each batch size needs to get within 2% of the optimum.

    One group of bars per objective, as in Table 5 of the paper, with the
    speed-up of the best minibatch size over the full batch that claims c1
    to c3 measure.
    """
    budget = len(curves["epochs"])
    figure = Figure(figsize=(7.6, 3.8), layout="constrained")
    axes = figure.subplots()
    width = 0.19
    for group, objective in enumerate(OBJECTIVE_LABELS):
        record = curves["epochs_to_optimum"][objective]
        means = [statistics.fmean(epochs) for epochs in record["epochs"]]
        for slot, (size, mean) in enumerate(
            zip(record["batch_size"], means, strict=True)
        ):
            position = group + (slot - (len(means) - 1) / 2.0) * (width + 0.02)
            axes.bar(
                position,
                mean,
                width,
                color=theme.series[slot],
                label=(f"batch {size}" if size else "full batch")
                if group == 0
                else None,
            )
            if size and mean > budget:
                axes.annotate(
                    "not reached",
                    (position, mean),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=theme.secondary,
                )
        claim = results[f"c{group + 1}"]
        axes.annotate(
            f"speed-up {claim.mean:.1f} (paper {claim.claim.paper_value:g})",
            (group, max(means)),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=theme.secondary,
        )
    axes.set_ymargin(0.12)
    axes.set_xticks(range(len(OBJECTIVE_LABELS)), list(OBJECTIVE_LABELS.values()))
    axes.grid(visible=False, axis="x")
    axes.set_title("Epochs to reach within 2% of the optimal training objective")
    axes.set_ylabel("epochs, mean over seeds")
    figure.legend(loc="outside right upper")
    return figure


def _draw_group_loss_reduction(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw claim c4 read at every epoch against the range of the paper.

    The pre-registered claim is the value after the last epoch. The other
    points apply the same comparison with the DRO and ERM runs both read at
    an earlier epoch.
    """
    epochs = curves["epochs"]
    mean = curves["reduction_mean"]
    spread = curves["reduction_std"]
    color = theme.series[0]
    figure = Figure(figsize=(6.4, 3.8), layout="constrained")
    axes = figure.subplots()
    paper_band(axes, theme, results["c4"])
    axes.fill_between(
        epochs,
        [m - s for m, s in zip(mean, spread, strict=True)],
        [m + s for m, s in zip(mean, spread, strict=True)],
        color=color,
        alpha=0.16,
    )
    axes.plot(epochs, mean, color=color, label="this run, mean over seeds")
    axes.plot(epochs[-1], mean[-1], "o", color=color, markersize=8)
    end_label(axes, theme, epochs[-1], mean[-1], f"{mean[-1]:.3f}")
    axes.set_xscale("log")
    axes.set_xlim(right=epochs[-1] * 1.9)
    axes.set_title("Reduction of the worst-group test loss against ERM")
    axes.set_xlabel("epoch at which both runs are read")
    axes.set_ylabel("claim c4")
    axes.legend(loc="upper right")
    return figure


FIGURES = (
    FigureSpec("epochs_to_optimum", _draw_epochs_to_optimum),
    FigureSpec("group_loss_reduction", _draw_group_loss_reduction),
)
