from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from optora_reproductions.figures import (
    DARK,
    LIGHT,
    THEMES,
    FigureSpec,
    Theme,
    data_digest,
    end_label,
    paper_band,
    rc_params,
    render,
    stored_digest,
    themed,
)
from optora_reproductions.results import Claim, ClaimResult, Environment

CLAIM = Claim("c1", "Table 1", "accuracy", 0.9, 0.02)
ENVIRONMENT = Environment("3.13.0", "2.14.0", "0.4.0", "Linux", "cpu")
CURVES: dict[str, Any] = {"x": [1.0, 2.0, 3.0], "y": [0.5, 0.7, 0.8]}


def result(
    values: tuple[float, ...] = (0.89, 0.91), claim: Claim = CLAIM
) -> ClaimResult:
    seeds = tuple(range(len(values)))
    return ClaimResult(
        "author2020_topic", claim, seeds, values, ENVIRONMENT, "2026-11-21"
    )


def draw(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    figure = Figure(figsize=(3.0, 2.0))
    axes = figure.subplots()
    paper_band(axes, theme, results["c1"])
    axes.plot(curves["x"], curves["y"], color=theme.series[0], label="ours")
    end_label(axes, theme, curves["x"][-1], curves["y"][-1], "last")
    axes.legend()
    return figure


SPEC = FigureSpec("line", draw)


class TestThemes:
    def test_light_comes_first_and_names_differ(self) -> None:
        assert THEMES == (LIGHT, DARK)
        assert [theme.name for theme in THEMES] == ["light", "dark"]

    @pytest.mark.parametrize("theme", THEMES)
    def test_series_has_eight_distinct_colours(self, theme: Theme) -> None:
        assert len(theme.series) == 8
        assert len(set(theme.series)) == 8

    @pytest.mark.parametrize("theme", THEMES)
    def test_rc_params_draw_on_an_opaque_surface(self, theme: Theme) -> None:
        params = rc_params(theme)

        assert params["figure.facecolor"] == theme.surface
        assert params["savefig.facecolor"] == theme.surface
        assert params["svg.fonttype"] == "path"

    def test_themed_restores_the_parameters(self) -> None:
        before = matplotlib.rcParams["text.color"]

        with themed(DARK):
            assert matplotlib.rcParams["text.color"] == DARK.ink

        assert matplotlib.rcParams["text.color"] == before


class TestDataDigest:
    def test_is_sha256_hex(self) -> None:
        digest = data_digest(CURVES, [result()])

        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")

    def test_ignores_the_key_order_of_the_curves(self) -> None:
        reordered = {"y": CURVES["y"], "x": CURVES["x"]}

        assert data_digest(CURVES, [result()]) == data_digest(reordered, [result()])

    def test_ignores_environment_and_date(self) -> None:
        other = dataclasses.replace(
            result(),
            environment=dataclasses.replace(ENVIRONMENT, optora="9.9.9"),
            date="2030-01-01",
        )

        assert data_digest(CURVES, [result()]) == data_digest(CURVES, [other])

    @pytest.mark.parametrize(
        "results",
        [
            [result((0.89, 0.92))],
            [result(claim=dataclasses.replace(CLAIM, paper_value=0.8))],
            [result(claim=dataclasses.replace(CLAIM, tolerance=0.1))],
            [],
        ],
    )
    def test_changes_with_the_claim_records(self, results: list[ClaimResult]) -> None:
        assert data_digest(CURVES, results) != data_digest(CURVES, [result()])

    def test_changes_with_the_curves(self) -> None:
        changed = {**CURVES, "y": [0.5, 0.7, 0.81]}

        assert data_digest(changed, [result()]) != data_digest(CURVES, [result()])

    def test_rejects_non_finite_values(self) -> None:
        with pytest.raises(ValueError):
            data_digest({"x": [float("nan")]}, [])


class TestRender:
    @pytest.mark.parametrize("theme", THEMES)
    def test_svg_stores_the_digest_of_its_data(self, theme: Theme) -> None:
        svg = render(SPEC, theme, CURVES, [result()])

        assert svg.startswith("<?xml")
        assert stored_digest(svg) == data_digest(CURVES, [result()])

    def test_is_deterministic(self) -> None:
        first = render(SPEC, LIGHT, CURVES, [result()])

        assert first == render(SPEC, LIGHT, CURVES, [result()])
        assert "<dc:date>" not in first

    def test_themes_give_different_files_on_their_own_surface(self) -> None:
        light = render(SPEC, LIGHT, CURVES, [result()])
        dark = render(SPEC, DARK, CURVES, [result()])

        assert light != dark
        assert LIGHT.surface in light
        assert DARK.surface in dark

    def test_draw_receives_the_records_by_claim_id(self) -> None:
        seen: list[list[str]] = []

        def spy(
            theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
        ) -> Figure:
            seen.append(sorted(results))
            return Figure(figsize=(1.0, 1.0))

        second = result(claim=dataclasses.replace(CLAIM, claim_id="c2"))
        render(FigureSpec("spy", spy), LIGHT, CURVES, [second, result()])

        assert seen == [["c1", "c2"]]

    def test_leaves_the_global_parameters_unchanged(self) -> None:
        before = dict(matplotlib.rcParams)

        render(SPEC, DARK, CURVES, [result()])

        assert dict(matplotlib.rcParams) == before


def test_stored_digest_is_none_without_one() -> None:
    assert stored_digest("<svg></svg>") is None


def test_paper_band_spans_the_tolerance_around_the_paper_value() -> None:
    axes = Figure().subplots()

    paper_band(axes, LIGHT, result())

    (band,) = axes.patches
    low, high = band.get_extents().transformed(axes.transData.inverted()).intervaly
    assert (low, high) == pytest.approx((0.88, 0.92))
    (line,) = axes.get_lines()
    assert np.asarray(line.get_ydata()).tolist() == [0.9, 0.9]


def test_end_label_writes_the_text_at_the_point() -> None:
    axes = Figure().subplots()

    end_label(axes, LIGHT, 2.0, 3.0, "final")

    (label,) = axes.texts
    assert isinstance(label, Annotation)
    assert label.get_text() == "final"
    assert label.xy == (2.0, 3.0)
