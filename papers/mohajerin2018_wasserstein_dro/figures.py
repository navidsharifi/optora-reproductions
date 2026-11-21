"""Figures of the report, drawn from the curve record and the claim records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme
from optora_reproductions.results import ClaimResult


def _draw_radius_sweep(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the out-of-sample performance and the reliability by radius.

    Figure 5(a) of the paper puts both on one plot with two vertical axes.
    Here they share the horizontal axis in two panels. The last entry of
    the curve record is the certificate setting and is not part of the
    sweep.
    """
    radii = curves["radius"][:-1]
    mean = curves["performance_mean"][:-1]
    best = min(range(len(radii)), key=mean.__getitem__)
    performance, reliability = theme.series[0], theme.series[1]
    claim = results["c3"]
    figure = Figure(figsize=(6.4, 5.0), layout="constrained")
    top, bottom = figure.subplots(2, 1, sharex=True)

    top.fill_between(
        radii,
        curves["performance_low"][:-1],
        curves["performance_high"][:-1],
        color=performance,
        alpha=0.16,
        label="20% to 80% quantiles",
    )
    top.plot(radii, mean, color=performance, label="mean over simulations")
    top.plot(radii[best], mean[best], "o", color=performance, markersize=8)
    top.annotate(
        f"lowest mean at $\\varepsilon = {radii[best]:g}$\n"
        f"{claim.mean:.3f} below $\\varepsilon = {radii[0]:g}$ "
        f"(paper {claim.claim.paper_value:g})",
        (radii[best], mean[best]),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.5,
        color=theme.secondary,
    )
    top.set_ymargin(0.28)
    top.set_title(f"Out-of-sample performance, $N = {curves['num_samples'][0]:g}$")
    top.set_ylabel(r"$J(\hat{x}_N(\varepsilon))$")
    top.legend(loc="lower left")

    bottom.plot(radii, curves["reliability"][:-1], color=reliability)
    bottom.set_xscale("log")
    bottom.set_ylim(-0.05, 1.05)
    bottom.set_title("Reliability of the certificate")
    bottom.set_xlabel(r"Wasserstein radius $\varepsilon$")
    bottom.set_ylabel(r"$\mathbb{P}[J(\hat{x}_N) \leq \hat{J}_N(\varepsilon)]$")
    return figure


FIGURES = (FigureSpec("radius_sweep", _draw_radius_sweep),)
