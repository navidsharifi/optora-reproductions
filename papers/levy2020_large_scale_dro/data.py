"""Digits features of Levy et al. (2020), Appendix F.1, and a synthetic stand-in.

The authors released the LeNet features of their Digits experiment as one
archive (fast-dro README, "Features for the Digits experiments"). It holds
500-dimensional features of the MNIST training and test images and of the
typed digits of de Campos, Babu and Varma (2009). `load_digits` downloads it
once into `data/`, checks its SHA-256 digest, reads the four feature files
with the safe `torch.load(weights_only=True)`, and builds the training and
test sets exactly as the authors' `MNISTandTypedFeatures` does.

`synthetic_digits` draws a small dataset with the same layout, two
populations of ten classes each, for the smoke run, so that continuous
integration never downloads the archive.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DIGITS_URL = "https://www.dropbox.com/s/e0puwp86vyh4dkg/digits_data.zip?dl=1"
DIGITS_SHA256 = "c4aa57a23eb77bda88ce5d42f2c988362c8caba746e4914ec6660d82adad908c"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "levy2020_large_scale_dro"
NUM_CLASSES = 10
POPULATIONS = ("mnist", "typed")
SUBSAMPLE_SEED = 123
_FEATURE_FILES = {
    ("mnist", "train"): "data/MNIST/lenet_features/train.pt",
    ("mnist", "test"): "data/MNIST/lenet_features/val.pt",
    ("typed", "train"): "data/typed_digits/lenet_features/train.pt",
    ("typed", "test"): "data/typed_digits/lenet_features/val.pt",
}
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Split:
    """Features, labels and subgroups of one split.

    Attributes:
        features: Float32 features of shape `(n, d)`.
        targets: Class labels in `[0, 10)` of shape `(n,)`.
        groups: Subgroup `10 * population + class` in `[0, 20)`, where
            population 0 is MNIST and 1 is typed, of shape `(n,)`.
    """

    features: torch.Tensor
    targets: torch.Tensor
    groups: torch.Tensor


@dataclass(frozen=True)
class Digits:
    """Training and test splits of the Digits experiment."""

    train: Split
    test: Split


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of the file at `path`."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(data_dir: Path = DATA_DIR, url: str = DIGITS_URL) -> Path:
    """Download the feature archive once and verify its digest.

    Args:
        data_dir: Cache folder. The archive is kept there as
            `digits_data.zip`.
        url: Address of the archive.

    Returns:
        Path of the verified archive.

    Raises:
        ValueError: If the downloaded or cached file has the wrong digest.
    """
    archive = data_dir / "digits_data.zip"
    if archive.exists():
        if sha256(archive) != DIGITS_SHA256:
            raise ValueError(f"{archive} has the wrong SHA-256 digest, delete it")
        return archive
    data_dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as partial,
        urllib.request.urlopen(url) as response,
    ):
        shutil.copyfileobj(response, partial, _CHUNK)
    partial_path = Path(partial.name)
    if sha256(partial_path) != DIGITS_SHA256:
        partial_path.unlink()
        raise ValueError(f"the archive at {url} has the wrong SHA-256 digest")
    partial_path.replace(archive)
    return archive


def _read_split(
    archive: zipfile.ZipFile, member: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the features and targets of one feature file in the archive."""
    with archive.open(member) as stream:
        data = torch.load(stream, map_location="cpu", weights_only=True)
    return data["features"].float(), data["targets"].long()


def _split(parts: list[tuple[torch.Tensor, torch.Tensor]]) -> Split:
    """Concatenate the populations of one split and label their subgroups."""
    features = torch.cat([features for features, _ in parts])
    targets = torch.cat([targets for _, targets in parts])
    groups = torch.cat(
        [NUM_CLASSES * index + targets for index, (_, targets) in enumerate(parts)]
    )
    return Split(features, targets, groups)


def subsample(count: int, size: int, seed: int) -> torch.Tensor:
    """Return the indices the authors' `utils.subsample_arrays` draws.

    It seeds the legacy global `numpy` generator and calls
    `numpy.random.choice` without replacement, which a `RandomState` with
    the same seed reproduces.
    """
    indices = np.random.RandomState(seed).choice(count, size, replace=False)
    return torch.from_numpy(indices)


def load_digits(typed_train: int = 600, data_dir: Path = DATA_DIR) -> Digits:
    """Load the Digits training and test sets of Appendix F.1.

    The authors' `MNISTandTypedFeatures` subsamples population `i` with seed
    `123 + i`, and `train.py` subsamples only the typed training digits, so
    those are drawn with seed 124. Every other part is used whole.

    Args:
        typed_train: Number of typed digits in the training set.
        data_dir: Cache folder of the archive.

    Returns:
        The training set (60,000 MNIST and `typed_train` typed digits) and the
        test set (10,000 MNIST and 8,000 typed digits).
    """
    with zipfile.ZipFile(download_archive(data_dir)) as archive:
        parts = {
            key: _read_split(archive, member) for key, member in _FEATURE_FILES.items()
        }
    features, targets = parts[("typed", "train")]
    chosen = subsample(len(targets), typed_train, SUBSAMPLE_SEED + 1)
    parts[("typed", "train")] = (features[chosen], targets[chosen])
    return Digits(
        train=_split([parts[(population, "train")] for population in POPULATIONS]),
        test=_split([parts[(population, "test")] for population in POPULATIONS]),
    )


def synthetic_digits(
    sizes: tuple[int, int] = (400, 40), dimension: int = 16, seed: int = 0
) -> Digits:
    """Draw a small dataset with the layout of the Digits experiment.

    Each of the two populations draws its class means from its own normal
    distribution, so the minority population is a distinct subgroup, as the
    typed digits are.

    Args:
        sizes: Training samples of the majority and minority populations.
            Each test split has the same sizes.
        dimension: Feature dimension.
        seed: Seed of the draw.

    Returns:
        Training and test splits with float32 features.
    """
    generator = torch.Generator().manual_seed(seed)
    means = torch.randn(len(sizes), NUM_CLASSES, dimension, generator=generator)

    def draw() -> Split:
        parts = []
        for population, size in enumerate(sizes):
            targets = torch.randint(NUM_CLASSES, (size,), generator=generator)
            noise = torch.randn(size, dimension, generator=generator)
            parts.append((means[population, targets] + noise, targets))
        return _split(parts)

    return Digits(train=draw(), test=draw())
