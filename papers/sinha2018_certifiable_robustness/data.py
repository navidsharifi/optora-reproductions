r"""MNIST as the authors loaded it, and the two-dimensional data of Section 4.1.

The authors read MNIST through `keras.datasets.mnist.load_data`, which
downloads one NumPy archive from Google's storage. The same file is
downloaded here and checked against the SHA-256 digest that Keras itself
pins for it. Pixels are scaled to $[0, 1]$ as in their `data_mnist`.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import PAPER_ID

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / PAPER_ID
NUM_CLASSES = 10
IMAGE_SHAPE = (1, 28, 28)
INNER_RADIUS = math.sqrt(2.0) / 1.3
OUTER_RADIUS = 1.3 * math.sqrt(2.0)


@dataclass(frozen=True)
class Split:
    """Inputs and integer labels of one split.

    Attributes:
        inputs: Float tensor with the examples along the first dimension.
        labels: Integer class of each example, of shape `(n,)`.
    """

    inputs: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> Split:
        """Return the split on `device`."""
        return Split(self.inputs.to(device), self.labels.to(device))


@dataclass(frozen=True)
class Dataset:
    """Training and test splits.

    Attributes:
        train: Training split.
        test: Test split.
    """

    train: Split
    test: Split

    def to(self, device: torch.device) -> Dataset:
        """Return both splits on `device`."""
        return Dataset(self.train.to(device), self.test.to(device))


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_mnist(data_dir: Path = DATA_DIR) -> Path:
    """Download the MNIST archive once and verify its digest.

    Args:
        data_dir: Folder of the cached archive.

    Returns:
        Path of the verified archive.

    Raises:
        ValueError: If the cached or downloaded file has the wrong digest.
    """
    path = data_dir / "mnist.npz"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as file:
            temporary = Path(file.name)
        try:
            urllib.request.urlretrieve(URL, temporary)
            if sha256(temporary) != SHA256:
                raise ValueError(f"{URL} has the wrong SHA-256 digest")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) != SHA256:
        raise ValueError(f"{path} has the wrong SHA-256 digest")
    return path


def _split(images: np.ndarray, labels: np.ndarray) -> Split:
    inputs = torch.from_numpy(images).float().div(255.0).unsqueeze(1)
    return Split(inputs, torch.from_numpy(labels).long())


def load_mnist(data_dir: Path = DATA_DIR) -> Dataset:
    """Return MNIST with pixels in `[0, 1]` and shape `(n, 1, 28, 28)`."""
    with np.load(download_mnist(data_dir), allow_pickle=False) as archive:
        train = _split(archive["x_train"], archive["y_train"])
        test = _split(archive["x_test"], archive["y_test"])
    return Dataset(train, test)


def random_images(train_size: int = 64, test_size: int = 32, seed: int = 0) -> Dataset:
    """Return random images of the MNIST shape, the stand-in of the smoke run."""
    generator = torch.Generator().manual_seed(seed)

    def split(size: int) -> Split:
        inputs = torch.rand((size, *IMAGE_SHAPE), generator=generator)
        labels = torch.randint(0, NUM_CLASSES, (size,), generator=generator)
        return Split(inputs, labels)

    return Dataset(split(train_size), split(test_size))


def two_dimensional(size: int, generator: torch.Generator) -> Split:
    r"""Sample the synthetic data of Section 4.1.

    Each input is drawn from $\mathcal N(0, I_2)$, inputs with norm in
    $(\sqrt 2 / 1.3, 1.3 \sqrt 2)$ are rejected to leave a margin, and the
    label is 1 when $\lVert x \rVert_2 > \sqrt 2$ and 0 otherwise, the two
    classes of $\mathrm{sign}(\lVert x \rVert_2 - \sqrt 2)$.

    Args:
        size: Number of examples.
        generator: Source of randomness on the CPU.

    Returns:
        A split with inputs of shape `(size, 2)`.
    """
    accepted: list[torch.Tensor] = []
    count = 0
    while count < size:
        candidates = torch.randn(2 * size, 2, generator=generator)
        norms = candidates.norm(dim=1)
        keep = candidates[(norms <= INNER_RADIUS) | (norms >= OUTER_RADIUS)]
        accepted.append(keep)
        count += len(keep)
    inputs = torch.cat(accepted)[:size]
    return Split(inputs, (inputs.norm(dim=1) > math.sqrt(2.0)).long())
