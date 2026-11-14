from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pytest

from optora_reproductions.results import Claim, ClaimResult, Environment, write_result
from tools.new_paper import create_paper
from tools.notebooks import (
    BLOB_URL,
    COLAB_URL,
    NOTEBOOK_NAME,
    RERUN_DIR,
    build_notebook,
    demote_headings,
    main,
    paper_ids,
)
from tools.results_table import EMPTY_TABLE

FIRST, SECOND = "levy2020_large_scale_dro", "sagawa2020_group_dro"


def record(paper_id: str = FIRST) -> ClaimResult:
    return ClaimResult(
        paper_id=paper_id,
        claim=Claim("c1", "Table 1", "worst-group accuracy", 0.9, 0.02),
        seeds=(0, 1),
        values=(0.89, 0.91),
        environment=Environment("3.13.0", "2.14.0", "0.2.0", "Linux", "RTX 3050"),
        date="2026-11-14",
    )


@pytest.fixture
def papers_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "papers"
    directory.mkdir()
    create_paper(FIRST, directory)
    create_paper(SECOND, directory)
    return directory


def cells(text: str) -> list[dict[str, Any]]:
    cell_list: list[dict[str, Any]] = json.loads(text)["cells"]
    return cell_list


def text_of(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


def build(papers_dir: Path, results: list[ClaimResult]) -> str:
    return build_notebook([papers_dir / FIRST, papers_dir / SECOND], results)


@pytest.mark.parametrize("results", [[], [record(), record(SECOND)]])
def test_notebook_passes_the_jupyter_schema(
    papers_dir: Path, results: list[ClaimResult]
) -> None:
    notebook = nbformat.reads(build(papers_dir, results), as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)


def test_empty_notebook_passes_the_jupyter_schema() -> None:
    notebook = nbformat.reads(build_notebook([], []), as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)


def test_code_cells_have_no_outputs(papers_dir: Path) -> None:
    for cell in cells(build(papers_dir, [record()])):
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_header_links_to_colab_and_lists_papers(papers_dir: Path) -> None:
    header = text_of(cells(build(papers_dir, []))[0])

    assert COLAB_URL in header
    assert f"notebooks/{NOTEBOOK_NAME}" in COLAB_URL
    assert f"- `{FIRST}`\n- `{SECOND}`\n" in header


def test_empty_notebook_says_no_paper_yet() -> None:
    texts = [text_of(cell) for cell in cells(build_notebook([], []))]

    assert "No paper has been added yet." in texts[0]
    assert any(EMPTY_TABLE in text for text in texts)


def test_setup_runs_once_before_the_papers(papers_dir: Path) -> None:
    texts = [text_of(cell) for cell in cells(build(papers_dir, []))]
    setups = [i for i, text in enumerate(texts) if "git clone" in text]
    first_run = next(
        i for i, text in enumerate(texts) if "papers." in text and "!python" in text
    )

    assert len(setups) == 1
    assert setups[0] < first_run


def test_each_paper_has_report_results_and_reruns(papers_dir: Path) -> None:
    texts = [text_of(cell) for cell in cells(build(papers_dir, []))]
    readme = (papers_dir / FIRST / "README.md").read_text(encoding="utf-8")

    assert demote_headings(readme.strip()) + "\n" in texts
    for paper_id in (FIRST, SECOND):
        commands = [
            t for t in texts if t.startswith(f"!python -m papers.{paper_id}.run")
        ]
        assert [c.rsplit("/", 1)[1] for c in commands] == [
            "smoke.toml\n",
            "full.toml\n",
        ]
        assert all(f"--output-dir {RERUN_DIR}" in c for c in commands)
        assert any(f'r.paper_id == "{paper_id}"' in t for t in texts)


def test_papers_appear_in_the_given_order(papers_dir: Path) -> None:
    text = build(papers_dir, [])
    assert text.index(f"papers.{FIRST}.run") < text.index(f"papers.{SECOND}.run")


def test_results_table_is_split_per_paper(papers_dir: Path) -> None:
    texts = [text_of(cell) for cell in cells(build(papers_dir, [record()]))]
    first_section = next(
        t for t in texts if t.startswith(f"### Recorded results of `{FIRST}`")
    )
    second_section = next(
        t for t in texts if t.startswith(f"### Recorded results of `{SECOND}`")
    )

    assert f"{BLOB_URL}papers/{FIRST}/README.md" in first_section
    assert EMPTY_TABLE in second_section


def test_results_of_papers_not_shown_are_left_out(papers_dir: Path) -> None:
    text = build(papers_dir, [record("other2021_topic")])
    assert "other2021_topic" not in text


def test_notebook_is_deterministic(papers_dir: Path) -> None:
    first = build(papers_dir, [record()])
    assert first == build(papers_dir, [record()])
    assert first.endswith("}\n")


def test_demote_headings_skips_code_fences() -> None:
    text = "# Title\n\n## Run\n\n```bash\n# a comment\n```\n\n#not a heading\n"
    assert demote_headings(text) == (
        "## Title\n\n### Run\n\n```bash\n# a comment\n```\n\n#not a heading\n"
    )


def test_demote_headings_keeps_level_six() -> None:
    assert demote_headings("###### Deep\n") == "###### Deep\n"


def test_paper_ids_skip_template_and_non_packages(papers_dir: Path) -> None:
    (papers_dir / "_template").mkdir()
    (papers_dir / "_template" / "__init__.py").touch()
    (papers_dir / "notes2020_draft").mkdir()

    assert paper_ids(papers_dir) == [FIRST, SECOND]


def arguments(papers_dir: Path, tmp_path: Path) -> list[str]:
    return [
        "--papers-dir",
        str(papers_dir),
        "--results-dir",
        str(tmp_path / "results"),
        "--output-dir",
        str(tmp_path / "notebooks"),
    ]


def test_main_writes_then_check_passes(papers_dir: Path, tmp_path: Path) -> None:
    write_result(record(), tmp_path / "results")
    args = arguments(papers_dir, tmp_path)

    assert main([*args, "--check"]) == 1
    assert main(args) == 0
    notebook = tmp_path / "notebooks" / NOTEBOOK_NAME
    assert "worst-group accuracy" in notebook.read_text(encoding="utf-8")
    assert main([*args, "--check"]) == 0


def test_check_fails_on_edited_notebook(papers_dir: Path, tmp_path: Path) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    (tmp_path / "notebooks" / NOTEBOOK_NAME).write_text("{}", encoding="utf-8")

    assert main([*args, "--check"]) == 1


def test_new_result_makes_notebook_stale(papers_dir: Path, tmp_path: Path) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    write_result(record(), tmp_path / "results")

    assert main([*args, "--check"]) == 1


def test_new_paper_makes_notebook_stale(papers_dir: Path, tmp_path: Path) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    create_paper("duchi2019_variance_regularization", papers_dir)

    assert main([*args, "--check"]) == 1


def test_other_notebook_fails_check_and_is_kept(
    papers_dir: Path, tmp_path: Path
) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    other = tmp_path / "notebooks" / "scratch.ipynb"
    other.write_text("{}", encoding="utf-8")

    assert main([*args, "--check"]) == 1
    assert main(args) == 0
    assert other.exists()


def test_repository_notebook_is_current() -> None:
    assert main(["--check"]) == 0
