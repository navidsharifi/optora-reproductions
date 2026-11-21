from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.new_paper import TEMPLATE_DIR, create_paper, main


def template_files() -> set[str]:
    return {
        path.relative_to(TEMPLATE_DIR).as_posix()
        for path in TEMPLATE_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_copies_every_template_file(tmp_path: Path) -> None:
    destination = create_paper("sagawa2020_group_dro", tmp_path)

    copied = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert destination == tmp_path / "sagawa2020_group_dro"
    assert copied == template_files()
    assert "README.md" in copied


def test_copy_runs_without_edits(tmp_path: Path) -> None:
    (tmp_path / "papers").mkdir()
    (tmp_path / "papers" / "__init__.py").touch()
    destination = create_paper("levy2020_large_scale_dro", tmp_path / "papers")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "papers"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "4 passed" in completed.stdout
    assert destination.joinpath("__init__.py").exists()


@pytest.mark.parametrize(
    "paper_id",
    ["sagawa_group_dro", "Sagawa2020_dro", "sagawa2020", "sagawa2020-dro", "2020_dro"],
)
def test_rejects_invalid_id(tmp_path: Path, paper_id: str) -> None:
    with pytest.raises(ValueError, match="paper id"):
        create_paper(paper_id, tmp_path)


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    create_paper("sagawa2020_group_dro", tmp_path)
    with pytest.raises(FileExistsError):
        create_paper("sagawa2020_group_dro", tmp_path)


def test_main_creates_folder_in_repository(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []

    def fake_create(paper_id: str) -> Path:
        calls.append(paper_id)
        return TEMPLATE_DIR.parent / paper_id

    monkeypatch.setattr("tools.new_paper.create_paper", fake_create)
    main(["duchi2019_variance_regularization"])

    assert calls == ["duchi2019_variance_regularization"]
    assert "papers" in capsys.readouterr().out
