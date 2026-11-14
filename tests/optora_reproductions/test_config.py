from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from optora_reproductions.config import load_config


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    learning_rate: float
    name: str
    use_gpu: bool
    num_steps: int = 10


@dataclass(frozen=True)
class Unsupported:
    value: int | None


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


VALID = 'seeds = [0, 1]\nlearning_rate = 0.1\nname = "erm"\nuse_gpu = false\n'


def test_loads_every_supported_type(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID + "num_steps = 3\n"), Settings)

    assert config == Settings((0, 1), 0.1, "erm", False, 3)
    assert isinstance(config.seeds, tuple)


def test_uses_field_default_when_key_is_absent(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, VALID), Settings).num_steps == 10


def test_accepts_path_as_string(tmp_path: Path) -> None:
    assert load_config(str(write(tmp_path, VALID)), Settings).name == "erm"


def test_widens_integer_to_float(tmp_path: Path) -> None:
    text = VALID.replace("learning_rate = 0.1", "learning_rate = 1")
    config = load_config(write(tmp_path, text), Settings)

    assert config.learning_rate == 1.0
    assert type(config.learning_rate) is float


def test_accepts_nonfinite_float(tmp_path: Path) -> None:
    text = VALID.replace("learning_rate = 0.1", "learning_rate = inf")
    assert math.isinf(load_config(write(tmp_path, text), Settings).learning_rate)


def test_accepts_empty_array(tmp_path: Path) -> None:
    text = VALID.replace("seeds = [0, 1]", "seeds = []")
    assert load_config(write(tmp_path, text), Settings).seeds == ()


def test_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys \\['typo'\\]"):
        load_config(write(tmp_path, VALID + "typo = 1\n"), Settings)


def test_rejects_missing_key(tmp_path: Path) -> None:
    text = VALID.replace('name = "erm"\n', "")
    with pytest.raises(ValueError, match="missing keys \\['name'\\]"):
        load_config(write(tmp_path, text), Settings)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("learning_rate = 0.1", 'learning_rate = "0.1"'),
        ("learning_rate = 0.1", "learning_rate = true"),
        ('name = "erm"', "name = 1"),
        ("use_gpu = false", "use_gpu = 0"),
        ("seeds = [0, 1]", "seeds = 0"),
        ("seeds = [0, 1]", "seeds = [0, 1.5]"),
        ("seeds = [0, 1]", "seeds = [true]"),
    ],
)
def test_rejects_value_of_wrong_type(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ValueError, match="expected"):
        load_config(write(tmp_path, VALID.replace(old, new)), Settings)


def test_rejects_integer_given_as_bool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected int"):
        load_config(write(tmp_path, VALID + "num_steps = true\n"), Settings)


def test_rejects_unsupported_annotation(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unsupported annotation"):
        load_config(write(tmp_path, "value = 1\n"), Unsupported)


def test_rejects_fixed_length_tuple(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Pair:
        pair: tuple[int, int]

    with pytest.raises(TypeError, match="only tuple"):
        load_config(write(tmp_path, "pair = [1, 2]\n"), Pair)


def test_rejects_non_dataclass(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="dataclass type"):
        load_config(write(tmp_path, VALID), dict)  # type: ignore[type-var]


def test_rejects_dataclass_instance(tmp_path: Path) -> None:
    instance = Settings((0,), 0.1, "erm", False)
    with pytest.raises(TypeError, match="dataclass type"):
        load_config(write(tmp_path, VALID), instance)  # type: ignore[arg-type]


def test_propagates_post_init_validation(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class Positive:
        value: int

        def __post_init__(self) -> None:
            if self.value <= 0:
                raise ValueError("value must be positive")

    with pytest.raises(ValueError, match="positive"):
        load_config(write(tmp_path, "value = 0\n"), Positive)
