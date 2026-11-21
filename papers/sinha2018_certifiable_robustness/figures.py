"""Figures of the report, drawn from the curve record and the claim records."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme
from optora_reproductions.results import ClaimResult

METHOD_LABELS = {
    "wrm": "WRM",
    "erm": "ERM",
    "fgm": "FGM",
    "ifgm": "IFGM",
    "pgm": "PGM",
}
"""Methods in the order of their colours."""
ATTACK_PANELS = {
    "l2": (r"$\ell_2$ PGM attack", r"$\epsilon_{\mathrm{adv}} / C_2$", "c4"),
    "linf": (
        r"$\ell_\infty$ PGM attack",
        r"$\epsilon_{\mathrm{adv}} / C_\infty$",
        "c5",
    ),
}
CERTIFICATE_PANELS = {"synthetic": "Synthetic data", "mnist": "MNIST"}
RANGE = 2.5
"""Robustness levels shown, as a multiple of the largest level of a certificate."""


def _draw_attack_error(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    """Draw the test error of every model under attacks of growing radius.

    The two panels follow Figure 3 of the paper, mean over seeds. The
    vertical line marks the radius at which a claim reads the error.
    """
    figure = Figure(figsize=(8.4, 3.7), layout="constrained")
    panels = figure.subplots(1, 2, sharey=True)
    for axes, (name, (title, label, claim)) in zip(
        panels, ATTACK_PANELS.items(), strict=True
    ):
        record = curves[name]
        axes.axvline(record["claim_radius"], color=theme.muted, linewidth=1.2)
        axes.annotate(
            f"radius of {claim}",
            (record["claim_radius"], 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, 4),
            textcoords="offset points",
            va="bottom",
            fontsize=8.5,
            color=theme.secondary,
        )
        for slot, (method, method_label) in enumerate(METHOD_LABELS.items()):
            if method in record["error"]:
                axes.plot(
                    record["radius"],
                    [100.0 * error for error in record["error"][method]],
                    color=theme.series[slot],
                    label=method_label,
                )
        axes.set_yscale("log")
        axes.locator_params(axis="x", nbins=6)
        axes.set_title(title)
        axes.set_xlabel(label)
    panels[0].set_ylabel("test error (%)")
    figure.legend(*panels[1].get_legend_handles_labels(), loc="outside right upper")
    return figure


def _draw_certificate(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    r"""Draw the certificate of the WRM model against its test worst case.

    The two panels follow Figure 2 of the paper, with one pair of lines per
    seed. The certificate at the level $\rho$ is
    $\gamma \rho + \mathbb{E}_{\hat{P}_n}[\phi_\gamma]$, a line of slope
    $\gamma$ through the point the training run reaches, which the marker
    shows.
    """
    certificate, worst_case = theme.series[1], theme.series[0]
    figure = Figure(figsize=(8.4, 3.6), layout="constrained")
    panels = figure.subplots(1, 2)
    for axes, (name, title) in zip(panels, CERTIFICATE_PANELS.items(), strict=True):
        seeds = curves[name]
        limit = RANGE * max(seed["distance"] for seed in seeds)
        for index, seed in enumerate(seeds):
            intercept = seed["loss"] - seed["penalty"] * seed["distance"]
            axes.plot(
                [0.0, limit],
                [intercept, intercept + seed["penalty"] * limit],
                color=certificate,
                linewidth=1.2,
                alpha=0.7,
                label="certificate" if index == 0 else None,
            )
            points = sorted(zip(seed["test_distance"], seed["test_loss"], strict=True))
            axes.plot(
                [distance for distance, _ in points],
                [loss for _, loss in points],
                color=worst_case,
                linewidth=1.2,
                alpha=0.7,
                label="test worst case" if index == 0 else None,
            )
            axes.plot(
                seed["distance"], seed["loss"], "o", color=certificate, markersize=5
            )
        level = statistics.fmean(seed["distance"] for seed in seeds)
        axes.axvline(level, color=theme.muted, linewidth=1.2, zorder=1)
        top = max(
            seed["loss"] + seed["penalty"] * (limit - seed["distance"])
            for seed in seeds
        )
        bottom = min(min(seed["test_loss"]) for seed in seeds)
        axes.set_xlim(0.0, limit)
        axes.set_ylim(bottom - 0.08 * (top - bottom), top + 0.04 * (top - bottom))
        axes.set_title(f"{title}, $\\hat{{\\rho}}_n = {level:.3f}$")
        axes.set_xlabel(r"robustness level $\rho$")
    panels[0].set_ylabel("worst-case loss")
    figure.legend(*panels[1].get_legend_handles_labels(), loc="outside right upper")
    return figure


FIGURES = (
    FigureSpec("attack_error", _draw_attack_error),
    FigureSpec("certificate", _draw_certificate),
)
