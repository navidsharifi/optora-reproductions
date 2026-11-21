"""Figures of the report, drawn from the curve record and the claim records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme, end_label
from optora_reproductions.results import ClaimResult


def _draw_bellman_update(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the error against the linear program and the time of one update.

    The left panel shows the largest error of claim c1 for each number of
    states with its tolerance. The right panel shows the mean time of one
    call for HiGHS and Optora, as in Figure 4 of the paper, and the ratio
    that claim c2 measures.
    """
    states = curves["num_states"]
    errors = [
        max(pair)
        for pair in zip(curves["random_error"], curves["inventory_error"], strict=True)
    ]
    optora, highs = theme.series[0], theme.series[1]
    figure = Figure(figsize=(8.4, 3.5), layout="constrained")
    left, right = figure.subplots(1, 2)

    left.plot(states, errors, "o-", color=optora, label="Optora")
    tolerance = results["c1"].claim.tolerance
    left.axhline(tolerance, color=theme.muted, linewidth=1.2, label="tolerance of c1")
    left.set_yscale("log")
    left.set_ylim(min(errors) / 10.0, tolerance * 10.0)
    left.set_title("Largest error against the linear program")
    left.set_xlabel("number of states $S$")
    left.set_ylabel(r"$\max\, |q_{\mathrm{Optora}} - q_{\mathrm{LP}}|$")
    left.legend(loc="center right")

    milliseconds = {
        "HiGHS": [1e3 * seconds for seconds in curves["lp_seconds"]],
        "Optora": [1e3 * seconds for seconds in curves["optora_seconds"]],
    }
    for (name, values), color in zip(
        milliseconds.items(), (highs, optora), strict=True
    ):
        right.plot(states, values, "o-", color=color, label=name)
        end_label(right, theme, states[-1], values[-1], f"{values[-1]:.3g} ms")
    speed_up = 10.0 ** results["c2"].mean
    paper = 10.0 ** results["c2"].claim.paper_value
    right.set_yscale("log")
    right.set_xticks(states)
    right.set_xlim(right=states[-1] + 0.2 * (states[-1] - states[0] + 1.0))
    right.set_title(
        f"Time of one update, {speed_up:.0f} times faster (paper {paper:.0f})"
    )
    right.set_xlabel("number of states $S$")
    right.set_ylabel("time per call (ms)")
    right.legend(loc="center left")
    return figure


FIGURES = (FigureSpec("bellman_update", _draw_bellman_update),)
