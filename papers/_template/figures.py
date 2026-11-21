"""Figures of the report, drawn from the curve record and the claim records.

Each figure is a function of the theme, the curve record that `run.py`
writes and the claim records by claim id. Declare it in `FIGURES`, run
`python -m tools.figures`, and embed the two files it writes in the README.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme
from optora_reproductions.results import ClaimResult


def _draw_closed_form(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the worst case of every seed against its closed form."""
    closed_form = curves["closed_form"]
    figure = Figure(figsize=(4.4, 3.6), layout="constrained")
    axes = figure.subplots()
    axes.plot(
        [min(closed_form), max(closed_form)],
        [min(closed_form), max(closed_form)],
        color=theme.muted,
        linewidth=1.2,
        label="closed form",
    )
    axes.plot(
        closed_form,
        curves["worst_case"],
        "o",
        color=theme.series[0],
        markersize=8,
        label="Optora, one seed each",
    )
    axes.set_title("Chi-square worst-case expectation")
    axes.set_xlabel(r"$\mathrm{mean} + \sqrt{\rho\, \mathrm{Var}}$")
    axes.set_ylabel("worst-case expectation")
    axes.legend(loc="upper left")
    return figure


FIGURES = (FigureSpec("closed_form", _draw_closed_form),)
