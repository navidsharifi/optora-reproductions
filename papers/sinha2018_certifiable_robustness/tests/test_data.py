from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from .. import data
from ..data import (
    IMAGE_SHAPE,
    INNER_RADIUS,
    NUM_CLASSES,
    OUTER_RADIUS,
    download_mnist,
    load_mnist,
    random_images,
    sha256,
    two_dimensional,
)


def _archive(path: Path) -> None:
    generator = np.random.default_rng(0)
    np.savez(
        path,
        x_train=generator.integers(0, 256, (6, 28, 28), dtype=np.uint8),
        y_train=np.array([0, 1, 2, 3, 4, 9], dtype=np.uint8),
        x_test=np.full((2, 28, 28), 255, dtype=np.uint8),
        y_test=np.array([7, 8], dtype=np.uint8),
    )


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"mnist" * 1000)
    assert sha256(path) == hashlib.sha256(b"mnist" * 1000).hexdigest()


def test_download_verifies_and_caches_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.npz"
    _archive(source)
    monkeypatch.setattr(data, "URL", source.as_uri())
    monkeypatch.setattr(data, "SHA256", sha256(source))
    cache = tmp_path / "cache"

    path = download_mnist(cache)

    assert path == cache / "mnist.npz"
    assert path.read_bytes() == source.read_bytes()
    assert download_mnist(cache) == path
    assert sorted(p.name for p in cache.iterdir()) == ["mnist.npz"]


def test_download_rejects_a_remote_file_with_the_wrong_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.npz"
    source.write_bytes(b"tampered")
    monkeypatch.setattr(data, "URL", source.as_uri())
    cache = tmp_path / "cache"

    with pytest.raises(ValueError, match="wrong SHA-256"):
        download_mnist(cache)
    assert list(cache.iterdir()) == []


def test_download_rejects_a_cached_file_with_the_wrong_digest(tmp_path: Path) -> None:
    (tmp_path / "mnist.npz").write_bytes(b"not mnist")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        download_mnist(tmp_path)


def test_load_mnist_scales_pixels_and_adds_a_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(tmp_path / "mnist.npz")
    monkeypatch.setattr(data, "SHA256", sha256(tmp_path / "mnist.npz"))

    mnist = load_mnist(tmp_path)

    assert mnist.train.inputs.shape == (6, *IMAGE_SHAPE)
    assert mnist.train.inputs.dtype == torch.float32
    assert (
        0.0 <= float(mnist.train.inputs.min()) <= float(mnist.train.inputs.max()) <= 1
    )
    assert torch.equal(mnist.test.inputs, torch.ones(2, *IMAGE_SHAPE))
    assert mnist.train.labels.dtype == torch.int64
    assert mnist.train.labels.tolist() == [0, 1, 2, 3, 4, 9]
    assert mnist.test.labels.tolist() == [7, 8]


def test_random_images_have_the_mnist_layout() -> None:
    images = random_images(train_size=10, test_size=4, seed=3)

    assert images.train.inputs.shape == (10, *IMAGE_SHAPE)
    assert images.test.inputs.shape == (4, *IMAGE_SHAPE)
    assert int(images.train.labels.max()) < NUM_CLASSES
    assert torch.equal(images.train.inputs, random_images(10, 4, seed=3).train.inputs)


def test_dataset_moves_both_splits() -> None:
    moved = random_images(4, 2).to(torch.device("cpu"))
    assert moved.train.inputs.device.type == moved.test.labels.device.type == "cpu"


def test_two_dimensional_data_leaves_a_margin_around_the_boundary() -> None:
    split = two_dimensional(5000, torch.Generator().manual_seed(0))
    norms = split.inputs.norm(dim=1)

    assert split.inputs.shape == (5000, 2)
    assert bool(((norms <= INNER_RADIUS) | (norms >= OUTER_RADIUS)).all())
    assert torch.equal(split.labels, (norms > math.sqrt(2)).long())
    inner = float((split.labels == 0).float().mean())
    assert 0.6 < inner < 0.8


def test_two_dimensional_data_is_reproducible() -> None:
    first = two_dimensional(100, torch.Generator().manual_seed(4))
    second = two_dimensional(100, torch.Generator().manual_seed(4))
    assert torch.equal(first.inputs, second.inputs)
