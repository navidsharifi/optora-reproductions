from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import math
from pathlib import Path

import pytest
import torch

from optora_reproductions.results import (
    Claim,
    ClaimResult,
    Environment,
    capture_environment,
    make_result,
    read_results,
    write_result,
)

CLAIM = Claim(
    claim_id="c1",
    location="Table 1",
    metric="accuracy",
    paper_value=0.9,
    tolerance=0.01,
)
ENVIRONMENT = Environment("3.13.0", "2.14.0", "0.2.0", "Linux", "cpu")


def result(
    values: tuple[float, ...] = (0.89, 0.91),
    paper_id: str = "author2020_topic",
    claim: Claim = CLAIM,
) -> ClaimResult:
    seeds = tuple(range(len(values)))
    return ClaimResult(paper_id, claim, seeds, values, ENVIRONMENT, "2026-11-14")


class TestClaim:
    @pytest.mark.parametrize("claim_id", ["", "C1", "c-1", "c 1", "c.1"])
    def test_rejects_invalid_id(self, claim_id: str) -> None:
        with pytest.raises(ValueError, match="claim_id"):
            dataclasses.replace(CLAIM, claim_id=claim_id)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_nonfinite_paper_value(self, value: float) -> None:
        with pytest.raises(ValueError, match="paper_value"):
            dataclasses.replace(CLAIM, paper_value=value)

    @pytest.mark.parametrize("tolerance", [-1e-12, math.nan])
    def test_rejects_invalid_tolerance(self, tolerance: float) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            dataclasses.replace(CLAIM, tolerance=tolerance)

    def test_accepts_zero_tolerance(self) -> None:
        assert dataclasses.replace(CLAIM, tolerance=0.0).tolerance == 0.0


class TestClaimResult:
    def test_mean_and_sample_std(self) -> None:
        record = result((1.0, 2.0, 3.0))

        assert record.mean == 2.0
        assert record.std == 1.0

    def test_single_seed_has_zero_std(self) -> None:
        assert result((0.5,)).std == 0.0

    @pytest.mark.parametrize(
        ("values", "expected"),
        [((0.9,), True), ((0.905,), True), ((0.92,), False), ((0.88,), False)],
    )
    def test_within_tolerance(self, values: tuple[float, ...], expected: bool) -> None:
        assert result(values).within_tolerance is expected

    def test_within_tolerance_is_inclusive(self) -> None:
        claim = dataclasses.replace(CLAIM, paper_value=1.0, tolerance=0.5)
        assert result((1.5,), claim=claim).within_tolerance

    def test_filename(self) -> None:
        assert result().filename == "author2020_topic.c1.json"

    def test_rejects_invalid_paper_id(self) -> None:
        with pytest.raises(ValueError, match="paper_id"):
            result(paper_id="Author-2020")

    def test_rejects_empty_seeds(self) -> None:
        with pytest.raises(ValueError, match="at least one seed"):
            result(())

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="2 seeds but 1 values"):
            ClaimResult("p2020_x", CLAIM, (0, 1), (0.5,), ENVIRONMENT, "2026-11-14")

    def test_rejects_repeated_seeds(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            ClaimResult("p2020_x", CLAIM, (0, 0), (0.5, 0.5), ENVIRONMENT, "2026-11-14")

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_rejects_nonfinite_values(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            result((0.5, value))

    def test_rejects_invalid_date(self) -> None:
        with pytest.raises(ValueError):
            ClaimResult("p2020_x", CLAIM, (0,), (0.5,), ENVIRONMENT, "14/11/2026")

    def test_json_round_trip(self) -> None:
        record = result((0.1, 0.2, 0.3))
        assert ClaimResult.from_json(record.to_json()) == record

    def test_json_is_strict_and_ends_with_newline(self) -> None:
        text = result().to_json()
        assert text.endswith("}\n")
        assert "NaN" not in text


class TestEnvironment:
    def test_captures_cpu_environment(self) -> None:
        environment = capture_environment("cpu")

        assert environment.torch == torch.__version__
        assert environment.optora == importlib.metadata.version("optora")
        assert environment.python.count(".") == 2
        assert environment.platform
        assert environment.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_captures_cuda_device_name(self) -> None:
        environment = capture_environment(torch.device("cuda"))
        assert environment.device == torch.cuda.get_device_name()


class TestMakeResult:
    def test_from_tensor(self) -> None:
        values = torch.tensor([0.25, 0.75], dtype=torch.float32, requires_grad=True)

        record = make_result("p2020_x", CLAIM, [3, 4], values * 1, "cpu")

        assert record.values == (0.25, 0.75)
        assert record.seeds == (3, 4)
        assert record.date == datetime.date.today().isoformat()

    def test_from_sequence(self) -> None:
        record = make_result("p2020_x", CLAIM, (0,), [1], "cpu")

        assert record.values == (1.0,)
        assert type(record.values[0]) is float

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_from_cuda_tensor(self) -> None:
        values = torch.tensor([0.5, 1.5], device="cuda")
        record = make_result("p2020_x", CLAIM, (0, 1), values, values.device)
        assert record.values == (0.5, 1.5)


class TestStorage:
    def test_write_then_read(self, tmp_path: Path) -> None:
        record = result()

        path = write_result(record, tmp_path / "new")

        assert path == tmp_path / "new" / record.filename
        assert read_results(tmp_path / "new") == [record]

    def test_write_replaces_earlier_record(self, tmp_path: Path) -> None:
        write_result(result((0.1,)), tmp_path)
        write_result(result((0.2,)), tmp_path)

        assert [record.values for record in read_results(tmp_path)] == [(0.2,)]

    def test_read_sorts_by_paper_then_claim(self, tmp_path: Path) -> None:
        records = [
            result(paper_id="b2020_x"),
            result(paper_id="a2020_x", claim=dataclasses.replace(CLAIM, claim_id="c2")),
            result(paper_id="a2020_x"),
        ]
        for record in records:
            write_result(record, tmp_path)

        keys = [(r.paper_id, r.claim.claim_id) for r in read_results(tmp_path)]
        assert keys == [("a2020_x", "c1"), ("a2020_x", "c2"), ("b2020_x", "c1")]

    def test_read_empty_folder(self, tmp_path: Path) -> None:
        assert read_results(tmp_path) == []

    def test_read_missing_folder(self, tmp_path: Path) -> None:
        assert read_results(tmp_path / "missing") == []

    def test_read_rejects_misnamed_file(self, tmp_path: Path) -> None:
        (tmp_path / "wrong.json").write_text(result().to_json(), encoding="utf-8")
        with pytest.raises(ValueError, match="holds the record"):
            read_results(tmp_path)
