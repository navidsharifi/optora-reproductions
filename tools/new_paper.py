"""Create a paper folder from `papers/_template`.

Usage::

    python -m tools.new_paper sagawa2020_group_dro
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPOSITORY_ROOT / "papers"
TEMPLATE_DIR = PAPERS_DIR / "_template"
PAPER_ID_PATTERN = re.compile(r"[a-z]+[0-9]{4}(?:_[a-z0-9]+)+")


def create_paper(paper_id: str, papers_dir: str | Path = PAPERS_DIR) -> Path:
    """Copy the template to `papers_dir / paper_id`.

    The copy needs no edits to run, because each paper reads its identifier
    from its own package name.

    Args:
        paper_id: `<firstauthor><year>_<topic>` in lowercase, for example
            `sagawa2020_group_dro`.
        papers_dir: Folder holding the paper folders.

    Returns:
        Path of the new folder.

    Raises:
        ValueError: If `paper_id` does not follow the naming rule.
        FileExistsError: If the folder already exists.
    """
    if not PAPER_ID_PATTERN.fullmatch(paper_id):
        raise ValueError(
            f"paper id must look like <firstauthor><year>_<topic>, got {paper_id!r}"
        )
    destination = Path(papers_dir) / paper_id
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    shutil.copytree(
        TEMPLATE_DIR,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and create the folder.

    Args:
        argv: Command-line arguments without the program name.
    """
    parser = argparse.ArgumentParser(description="Create a paper folder.")
    parser.add_argument("paper_id", help="<firstauthor><year>_<topic>")
    args = parser.parse_args(argv)
    destination = create_paper(args.paper_id)
    print(f"created {destination.relative_to(REPOSITORY_ROOT)}")
    print("next, fill README.md and claims.py, then replace data.py and run.py")


if __name__ == "__main__":
    main()
