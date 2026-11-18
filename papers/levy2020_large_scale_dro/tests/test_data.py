from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from .. import data
from ..data import (
    NUM_CLASSES,
    download_archive,
    load_digits,
    sha256,
    subsample,
    synthetic_digits,
)


def test_subsample_matches_the_legacy_global_generator() -> None:
    np.random.seed(124)
    expected = np.random.choice(2160, 600, replace=False)

    assert np.array_equal(subsample(2160, 600, 124).numpy(), expected)


def test_synthetic_digits_layout() -> None:
    digits = synthetic_digits(sizes=(30, 5), dimension=4)

    for split in (digits.train, digits.test):
        assert split.features.shape == (35, 4)
        assert split.features.dtype == torch.float32
        assert torch.equal(split.groups[:30], split.targets[:30])
        assert torch.equal(split.groups[30:], NUM_CLASSES + split.targets[30:])


def test_synthetic_digits_is_reproducible() -> None:
    first, second = synthetic_digits(seed=3), synthetic_digits(seed=3)
    assert torch.equal(first.train.features, second.train.features)
    other = synthetic_digits(seed=4)
    assert not torch.equal(first.train.features, other.train.features)


def test_sha256_of_a_file(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"abc")
    assert sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_a_cached_archive_with_the_wrong_digest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "digits_data.zip").write_bytes(b"not the archive")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        download_archive(tmp_path)


def _feature_file(count: int, offset: float) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "features": torch.full((count, 3), offset) + torch.arange(count)[:, None],
            "targets": torch.arange(count) % NUM_CLASSES,
            "predictions": torch.zeros(count, dtype=torch.long),
        },
        buffer,
    )
    return buffer.getvalue()


def test_load_digits_builds_the_splits_of_the_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "digits_data.zip"
    sizes = {
        "data/MNIST/lenet_features/train.pt": (40, 0.0),
        "data/MNIST/lenet_features/val.pt": (20, 100.0),
        "data/typed_digits/lenet_features/train.pt": (30, 200.0),
        "data/typed_digits/lenet_features/val.pt": (10, 300.0),
    }
    with zipfile.ZipFile(archive, "w") as handle:
        for member, (count, offset) in sizes.items():
            handle.writestr(member, _feature_file(count, offset))
    monkeypatch.setattr(data, "DIGITS_SHA256", sha256(archive))

    digits = load_digits(typed_train=6, data_dir=tmp_path)

    assert digits.train.features.shape == (46, 3)
    assert digits.test.features.shape == (30, 3)
    chosen = subsample(30, 6, 124)
    assert torch.equal(digits.train.features[40:, 0], 200.0 + chosen.float())
    assert torch.equal(digits.train.groups[40:], NUM_CLASSES + chosen % NUM_CLASSES)
    assert torch.equal(digits.test.groups[:20], torch.arange(20) % NUM_CLASSES)
    assert torch.all(digits.test.groups[20:] >= NUM_CLASSES)
