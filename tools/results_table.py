"""Generate the results table of `README.md` from `results/*.json`.

Usage::

    python -m tools.results_table          # rewrite the table
    python -m tools.results_table --check  # exit 1 if the table is stale
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from optora_reproductions.results import ClaimResult, read_results

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
START_MARKER = "<!-- results-table:start -->"
END_MARKER = "<!-- results-table:end -->"
EMPTY_TABLE = "No claims have been reproduced yet."
HEADER = (
    "Paper",
    "Claim",
    "Metric",
    "Paper value",
    "Ours, mean (std)",
    "Seeds",
    "Tolerance",
    "Within tolerance",
    "Device",
    "optora",
)


def _number(value: float) -> str:
    return f"{value:.4g}"


def _cell(text: str) -> str:
    return text.replace("|", r"\|")


def _row(result: ClaimResult) -> tuple[str, ...]:
    paper = f"[{result.paper_id}](papers/{result.paper_id}/README.md)"
    return (
        paper,
        result.claim.claim_id,
        _cell(result.claim.metric),
        _number(result.claim.paper_value),
        f"{_number(result.mean)} ({_number(result.std)})",
        str(len(result.seeds)),
        _number(result.claim.tolerance),
        "yes" if result.within_tolerance else "no",
        _cell(result.environment.device),
        result.environment.optora,
    )


def render_table(results: Sequence[ClaimResult]) -> str:
    """Render records as a Markdown table.

    Args:
        results: Records, in the order they should appear.

    Returns:
        The table, or a one-line note when there are no records.
    """
    if not results:
        return EMPTY_TABLE
    lines = [HEADER, tuple("---" for _ in HEADER), *map(_row, results)]
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


def replace_table(text: str, table: str) -> str:
    """Put `table` between the results markers of `text`.

    Args:
        text: Markdown document holding both markers once, start first.
        table: Content to place between them.

    Returns:
        The updated document.

    Raises:
        ValueError: If the markers are missing, repeated or out of order.
    """
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1 or end < start:
        raise ValueError("expected one start marker followed by one end marker")
    return f"{text[: start + len(START_MARKER)]}\n{table}\n{text[end:]}"


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite or check the table.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        Exit status, 1 when `--check` finds a stale table and 0 otherwise.
    """
    parser = argparse.ArgumentParser(description="Generate the results table.")
    parser.add_argument("--check", action="store_true", help="fail if stale")
    parser.add_argument("--readme", type=Path, default=REPOSITORY_ROOT / "README.md")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    args = parser.parse_args(argv)

    text = args.readme.read_text(encoding="utf-8")
    updated = replace_table(text, render_table(read_results(args.results_dir)))
    if args.check:
        if updated != text:
            print(f"{args.readme} is stale, run python -m tools.results_table")
            return 1
        return 0
    if updated != text:
        args.readme.write_text(updated, encoding="utf-8")
        print(f"updated {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
