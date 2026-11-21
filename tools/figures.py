"""Generate the figures of every paper report from the recorded results.

A paper declares its figures in `papers/<id>/figures.py`. Each one is drawn
from the paper's curve record in `results/curves/` and its claim records in
`results/`, once per colour scheme, and written to `papers/<id>/figures/`
as SVG. The paper README embeds the files, and the Colab notebook and the
Optora site show the same files.

Every SVG stores a digest of the data it was drawn from. `--check` compares
that digest with the current records, so it does not depend on the
Matplotlib build that rendered the file. It also fails when a figure is
missing, is not embedded in the paper README, or is not declared.

Usage::

    python -m tools.figures          # rewrite the figures
    python -m tools.figures --check  # exit 1 if a figure is stale
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from optora_reproductions.figures import (
    THEMES,
    FigureSpec,
    data_digest,
    render,
    stored_digest,
)
from optora_reproductions.results import ClaimResult, read_curves, read_results
from tools.notebooks import paper_ids

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIGURES_FOLDER = "figures"


def load_specs(paper_id: str) -> tuple[FigureSpec, ...]:
    """Return the figures declared by `papers/<paper_id>/figures.py`."""
    module = importlib.import_module(f"papers.{paper_id}.figures")
    specs: tuple[FigureSpec, ...] = module.FIGURES
    return specs


def file_names(specs: Sequence[FigureSpec]) -> list[str]:
    """Return the file names of `specs`, one per figure and colour scheme."""
    return [f"{spec.name}-{theme.name}.svg" for spec in specs for theme in THEMES]


def check_paper(
    paper_dir: Path, specs: Sequence[FigureSpec], digest: str | None
) -> list[str]:
    """List what is wrong with the figure files of one paper.

    Args:
        paper_dir: Folder of the paper.
        specs: Figures the paper declares, empty if it has no curve record.
        digest: Digest of the paper's current data, `None` without specs.

    Returns:
        One message per missing, stale, undeclared or unembedded file.
    """
    folder = paper_dir / FIGURES_FOLDER
    readme = (paper_dir / "README.md").read_text(encoding="utf-8")
    expected = file_names(specs)
    messages = []
    for name in expected:
        path = folder / name
        if not path.exists():
            messages.append(f"{path} is missing")
        elif stored_digest(path.read_text(encoding="utf-8")) != digest:
            messages.append(f"{path} is stale")
        if f"{FIGURES_FOLDER}/{name}" not in readme:
            messages.append(f"{path} is not embedded in the paper README")
    messages.extend(
        f"{path} is not declared in figures.py"
        for path in sorted(folder.glob("*.svg"))
        if path.name not in expected
    )
    return messages


def write_paper(
    paper_dir: Path,
    specs: Sequence[FigureSpec],
    curves: Mapping[str, Any],
    results: Sequence[ClaimResult],
) -> list[Path]:
    """Render the figures of one paper and write the files that changed.

    Args:
        paper_dir: Folder of the paper.
        specs: Figures the paper declares.
        curves: Curve record of the paper.
        results: Claim records of the paper.

    Returns:
        Paths of the files that were written.
    """
    folder = paper_dir / FIGURES_FOLDER
    written = []
    for spec in specs:
        for theme in THEMES:
            path = folder / f"{spec.name}-{theme.name}.svg"
            svg = render(spec, theme, curves, results)
            if not path.exists() or path.read_text(encoding="utf-8") != svg:
                folder.mkdir(parents=True, exist_ok=True)
                path.write_text(svg, encoding="utf-8")
                written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Write or check the figures of every paper.

    A paper without a curve record has no figures yet and is skipped.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        Exit status, 1 when `--check` finds a problem and 0 otherwise.
    """
    parser = argparse.ArgumentParser(description="Generate the report figures.")
    parser.add_argument("--check", action="store_true", help="fail if stale")
    parser.add_argument("--papers-dir", type=Path, default=REPOSITORY_ROOT / "papers")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    args = parser.parse_args(argv)

    every_result = read_results(args.results_dir)
    messages = []
    for paper_id in paper_ids(args.papers_dir):
        paper_dir = args.papers_dir / paper_id
        curves = read_curves(paper_id, args.results_dir)
        results = [result for result in every_result if result.paper_id == paper_id]
        specs = load_specs(paper_id) if curves is not None else ()
        if curves is not None and not args.check:
            for path in write_paper(paper_dir, specs, curves, results):
                print(f"wrote {path}")
        digest = data_digest(curves, results) if curves is not None else None
        messages.extend(check_paper(paper_dir, specs, digest))

    for message in messages:
        print(message)
    if args.check and messages:
        print("run python -m tools.figures and embed every figure in its README")
    return 1 if args.check and messages else 0


if __name__ == "__main__":
    sys.exit(main())
