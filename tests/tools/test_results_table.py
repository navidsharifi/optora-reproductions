from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from optora_reproductions.results import (
    Claim,
    ClaimResult,
    Environment,
    write_result,
)
from tools.results_table import (
    EMPTY_TABLE,
    END_MARKER,
    START_MARKER,
    main,
    render_table,
    replace_table,
)

CLAIM = Claim("c1", "Table 1", "worst-group accuracy", 0.9, 0.02)
RECORD = ClaimResult(
    paper_id="sagawa2020_group_dro",
    claim=CLAIM,
    seeds=(0, 1),
    values=(0.89, 0.91),
    environment=Environment("3.13.0", "2.14.0", "0.2.0", "Linux", "RTX 3050"),
    date="2026-11-14",
)
README = f"# Title\n\n{START_MARKER}\nold\n{END_MARKER}\n\nAfter.\n"


def test_render_empty() -> None:
    assert render_table([]) == EMPTY_TABLE


def test_render_one_record() -> None:
    lines = render_table([RECORD]).splitlines()

    assert len(lines) == 3
    assert lines[1].count("---") == 10
    assert lines[2] == (
        "| [sagawa2020_group_dro](papers/sagawa2020_group_dro/README.md) | c1 "
        "| worst-group accuracy | 0.9 | 0.9 (0.01414) | 2 | 0.02 | yes "
        "| RTX 3050 | 0.2.0 |"
    )


def test_render_marks_failed_claim() -> None:
    failed = dataclasses.replace(RECORD, values=(0.5, 0.5))
    assert render_table([failed]).splitlines()[2].endswith("| no | RTX 3050 | 0.2.0 |")


def test_render_escapes_pipes() -> None:
    record = dataclasses.replace(
        RECORD, claim=dataclasses.replace(CLAIM, metric="|a - b|")
    )
    assert r"\|a - b\|" in render_table([record])


def test_replace_keeps_surrounding_text() -> None:
    updated = replace_table(README, "new")

    assert updated == f"# Title\n\n{START_MARKER}\nnew\n{END_MARKER}\n\nAfter.\n"
    assert replace_table(updated, "new") == updated


@pytest.mark.parametrize(
    "text",
    [
        "no markers",
        f"{START_MARKER} only start",
        f"{END_MARKER} only end",
        f"{END_MARKER}\n{START_MARKER}",
        f"{START_MARKER}{START_MARKER}{END_MARKER}",
    ],
)
def test_replace_rejects_bad_markers(text: str) -> None:
    with pytest.raises(ValueError, match="marker"):
        replace_table(text, "table")


def paths(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    readme, results = tmp_path / "README.md", tmp_path / "results"
    readme.write_text(README, encoding="utf-8")
    results.mkdir()
    return readme, results, ["--readme", str(readme), "--results-dir", str(results)]


def test_main_writes_then_check_passes(tmp_path: Path) -> None:
    readme, results, args = paths(tmp_path)
    write_result(RECORD, results)

    assert main(args) == 0
    assert "sagawa2020_group_dro" in readme.read_text(encoding="utf-8")
    assert main([*args, "--check"]) == 0


def test_check_fails_on_stale_table_without_writing(tmp_path: Path) -> None:
    readme, _, args = paths(tmp_path)

    assert main([*args, "--check"]) == 1
    assert readme.read_text(encoding="utf-8") == README


def test_repository_readme_is_current() -> None:
    assert main(["--check"]) == 0
