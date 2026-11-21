"""Shared style and rendering of the report figures.

Every paper declares its figures in `papers/<id>/figures.py` as a tuple of
`FigureSpec`. A figure is a function of the paper's curve record and claim
records only, so it can be redrawn from the committed data at any time.
`render` draws one as SVG and stores a digest of that data in the file, and
`tools.figures` writes each figure once per colour scheme.

The two colour schemes and the Matplotlib parameters follow the figures of
the Optora documentation site (`tools/figures/_base.py` in Optora), so a
figure looks the same in a paper report and on the site. Matplotlib is an
optional dependency, installed with the `plot` extra.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import matplotlib.style
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .results import ClaimResult

_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


@dataclass(frozen=True)
class Theme:
    """One colour scheme of the figures.

    Attributes:
        name: Suffix of every rendered file, `light` or `dark`.
        surface: Background the figure is drawn on.
        ink: Primary text colour.
        secondary: Text colour of axis labels and legends.
        muted: Colour of tick labels and reference lines.
        grid: Gridline colour.
        axis: Axis spine colour.
        series: Categorical colours in a fixed order. A series keeps its
            position on every figure of a paper.
    """

    name: str
    surface: str
    ink: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#1b1b1b",
    secondary="#52514e",
    muted="#898781",
    grid="#e6e5df",
    axis="#c3c2b7",
    series=(
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
)

DARK = Theme(
    name="dark",
    surface="#1e2129",
    ink="#f0efec",
    secondary="#c3c2b7",
    muted="#8e8f97",
    grid="#30333d",
    axis="#454857",
    series=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
)

THEMES: tuple[Theme, ...] = (LIGHT, DARK)

Draw = Callable[[Theme, Mapping[str, Any], Mapping[str, ClaimResult]], Figure]
"""Signature of a figure, from the theme, the curve record and the claim
records of the paper by claim id to a Matplotlib figure."""


@dataclass(frozen=True)
class FigureSpec:
    """One figure of a paper report.

    Attributes:
        name: File stem of the figure, unique within the paper.
        draw: Function that draws the figure.
    """

    name: str
    draw: Draw


def rc_params(theme: Theme) -> dict[str, Any]:
    """Return the Matplotlib parameters of `theme`.

    The background is opaque, so a figure stays legible on any page. Text
    is drawn as paths in the font that ships with Matplotlib, so a figure
    does not depend on the fonts of the machine that renders or shows it.
    """
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "savefig.edgecolor": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "svg.fonttype": "path",
        "svg.hashsalt": "optora-reproductions",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "text.color": theme.ink,
        "axes.labelcolor": theme.secondary,
        "axes.edgecolor": theme.axis,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(theme.series)),
        "grid.color": theme.grid,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": theme.muted,
        "ytick.color": theme.muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": theme.surface,
        "patch.linewidth": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": theme.secondary,
        "legend.handlelength": 1.6,
    }


@contextmanager
def themed(theme: Theme) -> Iterator[None]:
    """Apply the Matplotlib parameters of `theme` inside the block."""
    with matplotlib.style.context(rc_params(theme)):
        yield


def data_digest(curves: Mapping[str, Any], results: Sequence[ClaimResult]) -> str:
    """Return the SHA-256 digest of the data the figures of a paper show.

    The digest covers the curve record and, for every claim, the claim
    itself, the seeds and the measured values. It does not cover the
    environment or the date of a record, which no figure shows.

    Args:
        curves: Curve record of the paper.
        results: Claim records of the paper.

    Returns:
        The digest as 64 hexadecimal digits.
    """
    payload = {
        "curves": curves,
        "results": [
            [dataclasses.asdict(result.claim), result.seeds, result.values]
            for result in results
        ],
    }
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stored_digest(svg: str) -> str | None:
    """Return the data digest stored in a rendered figure, if it has one."""
    match = _DIGEST.search(svg)
    return match.group(1) if match else None


def render(
    spec: FigureSpec,
    theme: Theme,
    curves: Mapping[str, Any],
    results: Sequence[ClaimResult],
) -> str:
    """Draw one figure in one colour scheme.

    Args:
        spec: The figure.
        theme: The colour scheme.
        curves: Curve record of the paper.
        results: Claim records of the paper.

    Returns:
        The figure as SVG text. Equal inputs give equal text on one
        Matplotlib build. The text holds `data_digest(curves, results)`.
    """
    by_claim = {result.claim.claim_id: result for result in results}
    buffer = io.BytesIO()
    with themed(theme):
        figure = spec.draw(theme, curves, by_claim)
        figure.savefig(
            buffer,
            format="svg",
            metadata={
                "Date": None,
                "Source": f"sha256:{data_digest(curves, results)}",
            },
        )
    return buffer.getvalue().decode("utf-8")


def paper_band(
    axes: Axes, theme: Theme, result: ClaimResult, label: str = "paper"
) -> None:
    """Mark the value a paper reports and the tolerance around it.

    Draws a horizontal line at the paper value of the claim and shades the
    interval of values that count as reproduced.

    Args:
        axes: Axes whose vertical axis is the metric of the claim.
        theme: Colour scheme of the figure.
        result: Record of the claim.
        label: Legend entry of the line.
    """
    claim = result.claim
    axes.axhspan(
        claim.paper_value - claim.tolerance,
        claim.paper_value + claim.tolerance,
        color=theme.muted,
        alpha=0.16,
        label=f"{label} $\\pm$ tolerance",
    )
    axes.axhline(claim.paper_value, color=theme.muted, linewidth=1.2)


def end_label(axes: Axes, theme: Theme, x: float, y: float, text: str) -> None:
    """Write `text` next to the point `(x, y)`, in the secondary text colour."""
    axes.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=theme.secondary,
    )
