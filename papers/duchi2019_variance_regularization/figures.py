"""Figures of the report, drawn from the curve record and the claim records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme, paper_band
from optora_reproductions.results import ClaimResult


def _draw_risk_ratio(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the robust risk over the ERM risk at the smallest sample size.

    Claim c3 reads this ratio at one noise level. The figure shows it at
    every noise level for the descent phase alone, the solver of the paper,
    and for the full solver with the subgradient phase, against the value
    of the paper and its tolerance.
    """
    smallest = min(curves["num_samples"])
    cells = [
        index
        for index, num_samples in enumerate(curves["num_samples"])
        if num_samples == smallest
    ]
    positions = range(len(cells))
    solvers = {
        "descent phase alone": ("descent", theme.series[1]),
        "with the subgradient phase": ("robust", theme.series[0]),
    }
    figure = Figure(figsize=(5.6, 3.6), layout="constrained")
    axes = figure.subplots()
    paper_band(axes, theme, results["c3"])
    for label, (name, color) in solvers.items():
        ratios = [curves[name][index] / curves["erm"][index] for index in cells]
        axes.plot(positions, ratios, "o-", color=color, label=label)
    axes.set_yscale("log")
    axes.set_xticks(positions, [f"{curves['noise'][index]:g}" for index in cells])
    axes.set_xmargin(0.15)
    axes.set_title(f"Robust risk over ERM risk at $n = {smallest:g}$")
    axes.set_xlabel("noise level $B$")
    axes.set_ylabel("mean robust risk / mean ERM risk")
    axes.legend(loc="center right")
    return figure


FIGURES = (FigureSpec("risk_ratio", _draw_risk_ratio),)
