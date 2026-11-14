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
    RERUN_DIR,
    build_all,
    build_notebook,
    main,
    paper_ids,
)
from tools.results_table import EMPTY_TABLE

PAPER_ID = "sagawa2020_group_dro"


def record(paper_id: str = PAPER_ID) -> ClaimResult:
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
    create_paper(PAPER_ID, directory)
    return directory


def text_of(cell: dict[str, Any]) -> str:
    return "".join(cell["source"])


@pytest.mark.parametrize("results", [[], [record()]])
def test_notebook_passes_the_jupyter_schema(
    papers_dir: Path, results: list[ClaimResult]
) -> None:
    text = build_notebook(papers_dir / PAPER_ID, results)
    notebook = nbformat.reads(text, as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)


def test_notebook_is_valid_nbformat_4(papers_dir: Path) -> None:
    notebook = json.loads(build_notebook(papers_dir / PAPER_ID, []))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_embeds_readme_and_runs_the_paper(papers_dir: Path) -> None:
    notebook = json.loads(build_notebook(papers_dir / PAPER_ID, []))
    texts = [text_of(cell) for cell in notebook["cells"]]
    readme = (papers_dir / PAPER_ID / "README.md").read_text(encoding="utf-8")

    assert readme.strip() + "\n" in texts
    assert any(f"notebooks/{PAPER_ID}.ipynb" in text for text in texts)
    commands = [t for t in texts if t.startswith(f"!python -m papers.{PAPER_ID}.run")]
    assert [command.rsplit("/", 1)[1] for command in commands] == [
        "smoke.toml\n",
        "full.toml\n",
    ]
    assert all(f"--output-dir {RERUN_DIR}" in command for command in commands)


def test_notebook_shows_only_this_papers_results(papers_dir: Path) -> None:
    results = [record(), record("other2021_topic")]
    text = build_notebook(papers_dir / PAPER_ID, results)

    assert f"{BLOB_URL}papers/{PAPER_ID}/README.md" in text
    assert "other2021_topic" not in text


def test_notebook_without_results_says_so(papers_dir: Path) -> None:
    assert EMPTY_TABLE in build_notebook(papers_dir / PAPER_ID, [])


def test_notebook_is_deterministic(papers_dir: Path) -> None:
    first = build_notebook(papers_dir / PAPER_ID, [record()])
    assert first == build_notebook(papers_dir / PAPER_ID, [record()])
    assert first.endswith("}\n")


def test_paper_ids_skip_template_and_non_packages(papers_dir: Path) -> None:
    create_paper("levy2020_large_scale_dro", papers_dir)
    (papers_dir / "_template").mkdir()
    (papers_dir / "_template" / "__init__.py").touch()
    (papers_dir / "notes2020_draft").mkdir()

    assert paper_ids(papers_dir) == ["levy2020_large_scale_dro", PAPER_ID]


def test_build_all_names_notebooks_after_papers(
    papers_dir: Path, tmp_path: Path
) -> None:
    assert list(build_all(papers_dir, tmp_path / "results")) == [f"{PAPER_ID}.ipynb"]


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
    notebook = tmp_path / "notebooks" / f"{PAPER_ID}.ipynb"
    assert "worst-group accuracy" in notebook.read_text(encoding="utf-8")
    assert main([*args, "--check"]) == 0


def test_check_fails_on_edited_notebook(papers_dir: Path, tmp_path: Path) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    (tmp_path / "notebooks" / f"{PAPER_ID}.ipynb").write_text("{}", encoding="utf-8")

    assert main([*args, "--check"]) == 1


def test_new_result_makes_notebook_stale(papers_dir: Path, tmp_path: Path) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    write_result(record(), tmp_path / "results")

    assert main([*args, "--check"]) == 1


def test_orphan_notebook_fails_check_and_is_kept(
    papers_dir: Path, tmp_path: Path
) -> None:
    args = arguments(papers_dir, tmp_path)
    main(args)
    orphan = tmp_path / "notebooks" / "gone2019_paper.ipynb"
    orphan.write_text("{}", encoding="utf-8")

    assert main([*args, "--check"]) == 1
    assert main(args) == 0
    assert orphan.exists()


def test_repository_notebooks_are_current() -> None:
    assert main(["--check"]) == 0
