"""Generate the combined Google Colab notebook of all reproductions.

The notebook is built from every paper's `README.md` and the records in
`results/`, so it shows the recorded results without being run. It opens
with the results table of all papers and one setup cell, followed by one
section per paper with its report, its recorded results, and cells that
rerun it on Colab. The notebook is a generated file, never edited by hand.

Usage::

    python -m tools.notebooks          # write notebooks/reproductions.ipynb
    python -m tools.notebooks --check  # exit 1 if the notebook is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from optora_reproductions.results import ClaimResult, read_results
from tools.results_table import render_table

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "navidsharifi/optora-reproductions"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
BLOB_URL = f"{REPOSITORY_URL}/blob/main/"
NOTEBOOK_NAME = "reproductions.ipynb"
COLAB_URL = (
    f"https://colab.research.google.com/github/{REPOSITORY}/blob/main/"
    f"notebooks/{NOTEBOOK_NAME}"
)
CLONE_DIR = "/content/optora-reproductions"
RERUN_DIR = "runs/colab"

_HEADING = re.compile(r"#{1,5} ")

_SETUP = f"""\
# Clone (or update) the repository and install it with its pinned optora
# release. Colab already ships a CUDA build of PyTorch, which pip keeps.
![ -d {CLONE_DIR} ] || git clone --depth 1 {REPOSITORY_URL}.git {CLONE_DIR}
!git -C {CLONE_DIR} pull --ff-only --quiet
%cd {CLONE_DIR}
!pip install --quiet -e ".[vision]"

import optora
import torch
from IPython.display import Markdown, display

from optora_reproductions.results import read_results
from tools.results_table import render_table

print("optora", optora.__version__, "| torch", torch.__version__)
print("GPU:", torch.cuda.get_device_name() if torch.cuda.is_available() else "none")
"""


def _source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _show_rerun(paper_id: str | None) -> str:
    selection = "" if paper_id is None else f' if r.paper_id == "{paper_id}"'
    return (
        f'reruns = [r for r in read_results("{RERUN_DIR}"){selection}]\n'
        "display(Markdown(render_table(reruns)))\n"
    )


def demote_headings(text: str) -> str:
    """Move every Markdown heading one level down, outside code fences.

    A paper README starts at level 1. Inside the combined notebook each
    paper is a level-2 section, so its headings shift by one.

    Args:
        text: Markdown text.

    Returns:
        The text with `#` added to each ATX heading of level 1 to 5.
    """
    lines = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and _HEADING.match(line):
            line = "#" + line
        lines.append(line)
    return "".join(lines)


def paper_ids(papers_dir: str | Path) -> list[str]:
    """List the papers shown in the notebook, skipping `_template`.

    Args:
        papers_dir: Folder holding the paper folders.

    Returns:
        Sorted folder names of the papers.
    """
    return sorted(
        path.name
        for path in Path(papers_dir).iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and (path / "__init__.py").exists()
    )


def _paper_cells(paper_dir: Path, results: Sequence[ClaimResult]) -> list[Any]:
    paper_id = paper_dir.name
    readme = (paper_dir / "README.md").read_text(encoding="utf-8").strip()
    recorded = [result for result in results if result.paper_id == paper_id]
    run = (
        f"!python -m papers.{paper_id}.run --output-dir {RERUN_DIR} "
        f"--config papers/{paper_id}/configs"
    )
    return [
        _markdown(demote_headings(readme) + "\n"),
        _markdown(
            f"### Recorded results of `{paper_id}`\n\n"
            + render_table(recorded, link_base=BLOB_URL)
            + "\n"
        ),
        _markdown(
            f"### Rerun `{paper_id}` on Colab\n\n"
            "The smoke configuration takes seconds. The full configuration "
            "repeats the recorded run and can take much longer.\n"
        ),
        _code(f"{run}/smoke.toml\n"),
        _code(f"{run}/full.toml\n"),
        _code(_show_rerun(paper_id)),
    ]


def build_notebook(
    paper_dirs: Sequence[str | Path], results: Sequence[ClaimResult]
) -> str:
    """Build the combined notebook as `.ipynb` JSON text.

    Args:
        paper_dirs: Folders of the papers to include, in display order,
            each named after its paper id.
        results: Records of every paper.

    Returns:
        The notebook in nbformat 4 JSON, deterministic for equal inputs.
    """
    directories = [Path(path) for path in paper_dirs]
    shown = {directory.name for directory in directories}
    recorded = [result for result in results if result.paper_id in shown]
    contents = "".join(f"- `{directory.name}`\n" for directory in directories) or (
        "No paper has been added yet.\n"
    )

    cells = [
        _markdown(
            f"[![Open In Colab](https://colab.research.google.com/assets/"
            f"colab-badge.svg)]({COLAB_URL})\n\n"
            "# Optora reproductions\n\n"
            "Reproductions of published distributionally robust optimization "
            f"papers with [Optora](https://github.com/navidsharifi/optora). "
            f"This notebook is generated by `python -m tools.notebooks` from "
            f"[`papers/`]({BLOB_URL}papers) and [`results/`]({BLOB_URL}results) "
            f"of [{REPOSITORY}]({REPOSITORY_URL}), so changes belong in those "
            "files, not here.\n\n"
            f"Papers in this notebook\n\n{contents}"
        ),
        _markdown(
            "## Recorded results\n\n"
            + render_table(recorded, link_base=BLOB_URL)
            + "\n\nThese are the committed records of the full runs. A claim "
            "is reproduced when the mean over seeds lies within the tolerance "
            "that was fixed before the run.\n"
        ),
        _markdown(
            "## Setup\n\n"
            "Choose a GPU runtime (Runtime, Change runtime type), then run the "
            "cell below once before rerunning any paper.\n"
        ),
        _code(_SETUP),
    ]
    for directory in directories:
        cells.extend(_paper_cells(directory, results))
    cells.extend(
        [
            _markdown("## All rerun results\n\nEvery claim rerun in this session.\n"),
            _code(_show_rerun(None)),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK_NAME, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the notebook.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        Exit status, 1 when `--check` finds the notebook stale or missing,
        or another notebook in the output folder, and 0 otherwise.
    """
    parser = argparse.ArgumentParser(description="Generate the Colab notebook.")
    parser.add_argument("--check", action="store_true", help="fail if stale")
    parser.add_argument("--papers-dir", type=Path, default=REPOSITORY_ROOT / "papers")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "notebooks"
    )
    args = parser.parse_args(argv)

    papers = [args.papers_dir / paper_id for paper_id in paper_ids(args.papers_dir)]
    expected = build_notebook(papers, read_results(args.results_dir))
    path = args.output_dir / NOTEBOOK_NAME
    stale = not path.exists() or path.read_text(encoding="utf-8") != expected
    others = sorted(
        other.name
        for other in args.output_dir.glob("*.ipynb")
        if other.name != NOTEBOOK_NAME
    )

    if args.check:
        if stale:
            print(f"{path} is stale or missing, run python -m tools.notebooks")
        for name in others:
            print(f"{name} is not generated by tools.notebooks")
        return 1 if stale or others else 0

    if stale:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
        print(f"wrote {path}")
    for name in others:
        print(f"warning: {name} is not generated by tools.notebooks; delete it by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
