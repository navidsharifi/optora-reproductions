from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from matplotlib.figure import Figure

from optora_reproductions.figures import FigureSpec, Theme, data_digest, stored_digest
from optora_reproductions.results import (
    Claim,
    ClaimResult,
    Environment,
    write_curves,
    write_result,
)
from tools import figures
from tools.figures import FIGURES_FOLDER, file_names, load_specs, main
from tools.new_paper import create_paper

PAPER = "levy2020_large_scale_dro"
OTHER = "sagawa2020_group_dro"
CURVES: dict[str, Any] = {"x": [1.0, 2.0, 3.0], "y": [0.5, 0.7, 0.8]}
FILES = ["line-light.svg", "line-dark.svg"]


def record(values: tuple[float, ...] = (0.89, 0.91)) -> ClaimResult:
    return ClaimResult(
        paper_id=PAPER,
        claim=Claim("c1", "Table 1", "accuracy", 0.9, 0.02),
        seeds=(0, 1),
        values=values,
        environment=Environment("3.13.0", "2.14.0", "0.4.0", "Linux", "cpu"),
        date="2026-11-21",
    )


def draw(
    theme: Theme, curves: Mapping[str, Any], results: Mapping[str, ClaimResult]
) -> Figure:
    figure = Figure(figsize=(3.0, 2.0))
    figure.subplots().plot(curves["x"], curves["y"], color=theme.series[0])
    return figure


SPECS = (FigureSpec("line", draw),)


@pytest.fixture
def arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Arguments of a repository with two papers, one of them with records.

    The first paper has a claim record, a curve record and a README that
    embeds both files of the one figure it declares.
    """
    monkeypatch.setattr(figures, "load_specs", lambda paper_id: SPECS)
    papers = tmp_path / "papers"
    papers.mkdir()
    for paper_id in (PAPER, OTHER):
        create_paper(paper_id, papers)
    embeds = "".join(f"![figure]({FIGURES_FOLDER}/{name})\n" for name in FILES)
    with (papers / PAPER / "README.md").open("a", encoding="utf-8") as readme:
        readme.write(embeds)
    write_result(record(), tmp_path / "results")
    write_curves(PAPER, CURVES, tmp_path / "results")
    return ["--papers-dir", str(papers), "--results-dir", str(tmp_path / "results")]


def folder(arguments: list[str], paper_id: str = PAPER) -> Path:
    return Path(arguments[1]) / paper_id / FIGURES_FOLDER


def test_file_names_cover_every_figure_and_theme() -> None:
    specs = (*SPECS, FigureSpec("bars", draw))

    assert file_names(specs) == [*FILES, "bars-light.svg", "bars-dark.svg"]


def test_load_specs_reads_the_figures_module_of_the_paper() -> None:
    assert [spec.name for spec in load_specs("_template")] == ["closed_form"]


def test_main_writes_then_check_passes(arguments: list[str]) -> None:
    assert main([*arguments, "--check"]) == 1
    assert main(arguments) == 0

    written = sorted(path.name for path in folder(arguments).iterdir())
    assert written == sorted(FILES)
    assert main([*arguments, "--check"]) == 0


def test_figures_store_the_digest_of_the_records(arguments: list[str]) -> None:
    main(arguments)

    for name in FILES:
        svg = (folder(arguments) / name).read_text(encoding="utf-8")
        assert stored_digest(svg) == data_digest(CURVES, [record()])


def test_unchanged_figures_are_not_rewritten(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    main(arguments)
    assert capsys.readouterr().out.count("wrote") == len(FILES)

    main(arguments)
    assert "wrote" not in capsys.readouterr().out


def test_paper_without_curve_record_has_no_figures(arguments: list[str]) -> None:
    main(arguments)

    assert not folder(arguments, OTHER).exists()


def test_check_fails_on_new_curves(arguments: list[str]) -> None:
    main(arguments)
    write_curves(PAPER, {**CURVES, "y": [0.5, 0.7, 0.9]}, Path(arguments[3]))

    assert main([*arguments, "--check"]) == 1
    assert main(arguments) == 0
    assert main([*arguments, "--check"]) == 0


def test_check_fails_on_new_claim_values(arguments: list[str]) -> None:
    main(arguments)
    write_result(record((0.80, 0.82)), Path(arguments[3]))

    assert main([*arguments, "--check"]) == 1


def test_check_fails_on_missing_figure(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    main(arguments)
    (folder(arguments) / FILES[1]).unlink()
    capsys.readouterr()

    assert main([*arguments, "--check"]) == 1
    assert f"{FILES[1]} is missing" in capsys.readouterr().out


def test_check_fails_on_figure_without_digest(arguments: list[str]) -> None:
    main(arguments)
    (folder(arguments) / FILES[0]).write_text("<svg></svg>", encoding="utf-8")

    assert main([*arguments, "--check"]) == 1


def test_check_fails_when_readme_does_not_embed_a_figure(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    main(arguments)
    readme = Path(arguments[1]) / PAPER / "README.md"
    text = readme.read_text(encoding="utf-8")
    readme.write_text(text.replace(f"({FIGURES_FOLDER}/{FILES[0]})", "()"), "utf-8")
    capsys.readouterr()

    assert main([*arguments, "--check"]) == 1
    assert "is not embedded in the paper README" in capsys.readouterr().out


def test_undeclared_figure_fails_check_and_is_kept(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    main(arguments)
    extra = folder(arguments) / "old-light.svg"
    extra.write_text("<svg></svg>", encoding="utf-8")
    capsys.readouterr()

    assert main([*arguments, "--check"]) == 1
    assert "is not declared in figures.py" in capsys.readouterr().out
    assert main(arguments) == 0
    assert extra.exists()


def test_figure_of_a_paper_without_curve_record_fails_check(
    arguments: list[str],
) -> None:
    main(arguments)
    folder(arguments, OTHER).mkdir()
    (folder(arguments, OTHER) / FILES[0]).write_text("<svg></svg>", encoding="utf-8")

    assert main([*arguments, "--check"]) == 1


def test_repository_figures_are_current() -> None:
    assert main(["--check"]) == 0
