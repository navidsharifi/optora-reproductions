"""Waterbirds of Sagawa et al. (2020), Appendix C.1, and a synthetic stand-in.

The authors release Waterbirds as one tarball on the Stanford NLP server,
which did not answer when this reproduction was written, and the WILDS copy
on CodaLab returned no content. `load_waterbirds` therefore reads the three
Parquet files of the Hugging Face dataset `grodino/waterbirds` at a fixed
revision. Its upload script (`scripts/waterbirds.py` in that repository)
downloads the authors' tarball, splits it by the `split` column of the
authors' `metadata.csv` and stores every original JPEG file unchanged with
its file name, its label `y` and its background `place`. The group counts of
all three splits equal those of the paper.

Each image is decoded once, resized to 256 by 256 pixels and center-cropped
to 224, as the authors' `get_transform_cub` does without augmentation. The
`uint8` result is cached next to the downloads, so training never decodes a
JPEG. Normalization with the ImageNet statistics happens on the device, in
`model.normalize`.

`synthetic_waterbirds` draws small images with the same layout, a label
spuriously correlated with a background, for the smoke run, so that
continuous integration never downloads the dataset.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REVISION = "e9856c710d0da2e4029d116cdd9d5fce7cc2bc80"
URL = (
    "https://huggingface.co/datasets/grodino/waterbirds/resolve/"
    "{revision}/data/{split}-00000-of-00001.parquet"
)
SHA256 = {
    "train": "575b462a17ae00e6a03b4f1833e3a5952e28492e278d3246186586c27a46367a",
    "validation": "9ab73e8625fb5856e132ecbbe509f20829e5d59a9bd087b6cda86f5c478350fb",
    "test": "fdb8696ce4db78c0ad48dd0120aaae40ce4330416d37fc41c9ea57faadc007f7",
}
SPLITS = ("train", "validation", "test")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sagawa2020_group_dro"
NUM_CLASSES = 2
NUM_GROUPS = 4
GROUP_NAMES = (
    "landbird on land",
    "landbird on water",
    "waterbird on land",
    "waterbird on water",
)
RESIZE = 256
CROP = 224
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Split:
    """Images, labels and groups of one split.

    Attributes:
        images: `uint8` RGB images of shape `(n, 3, h, w)`.
        targets: Labels of shape `(n,)`, 0 for a landbird and 1 for a
            waterbird.
        groups: Group `2 * label + place` of shape `(n,)`, where place 0 is
            land and 1 is water, as in the authors' `CUBDataset`.
    """

    images: torch.Tensor
    targets: torch.Tensor
    groups: torch.Tensor


@dataclass(frozen=True)
class Waterbirds:
    """Training, validation and test splits."""

    train: Split
    validation: Split
    test: Split


def group_counts(groups: torch.Tensor) -> torch.Tensor:
    """Return the number of examples of each group, of shape `(NUM_GROUPS,)`."""
    return torch.zeros(
        NUM_GROUPS, dtype=torch.int64, device=groups.device
    ).scatter_add_(0, groups, torch.ones_like(groups))


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of the file at `path`."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_split(split: str, data_dir: Path = DATA_DIR) -> Path:
    """Download the Parquet file of one split once and verify its digest.

    Args:
        split: One of `SPLITS`.
        data_dir: Cache folder. The file is kept there as `<split>.parquet`.

    Returns:
        Path of the verified file.

    Raises:
        ValueError: If `split` is unknown or a file has the wrong digest.
    """
    if split not in SHA256:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = data_dir / f"{split}.parquet"
    if path.exists():
        if sha256(path) != SHA256[split]:
            raise ValueError(f"{path} has the wrong SHA-256 digest, delete it")
        return path
    data_dir.mkdir(parents=True, exist_ok=True)
    url = URL.format(revision=REVISION, split=split)
    with (
        tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as partial,
        urllib.request.urlopen(url) as response,
    ):
        shutil.copyfileobj(response, partial, _CHUNK)
    partial_path = Path(partial.name)
    if sha256(partial_path) != SHA256[split]:
        partial_path.unlink()
        raise ValueError(f"the file at {url} has the wrong SHA-256 digest")
    partial_path.replace(path)
    return path


def preprocess(jpeg: bytes) -> torch.Tensor:
    """Decode one image, resize it to 256 by 256 and crop the center 224.

    Args:
        jpeg: Encoded image file.

    Returns:
        `uint8` tensor of shape `(3, 224, 224)`.
    """
    from PIL import Image
    from torchvision.transforms import v2

    with Image.open(io.BytesIO(jpeg)) as image:
        rgb = image.convert("RGB")
    transform = v2.Compose([v2.Resize((RESIZE, RESIZE)), v2.CenterCrop(CROP)])
    cropped = transform(rgb)
    return torch.from_numpy(np.array(cropped, dtype=np.uint8)).permute(2, 0, 1).clone()


def _decode(path: Path) -> Split:
    """Read one Parquet file and preprocess every image in it."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["image", "label", "place"])
    images = torch.stack(
        [preprocess(item["bytes"]) for item in table.column("image").to_pylist()]
    )
    targets = torch.tensor(table.column("label").to_pylist(), dtype=torch.int64)
    places = torch.tensor(table.column("place").to_pylist(), dtype=torch.int64)
    return Split(images, targets, 2 * targets + places)


def load_split(split: str, data_dir: Path = DATA_DIR) -> Split:
    """Load one split, downloading and decoding it on first use.

    The decoded tensors are cached as `<split>.pt` and read back with the
    safe `torch.load(weights_only=True)`.

    Args:
        split: One of `SPLITS`.
        data_dir: Cache folder.

    Returns:
        The split.
    """
    cache = data_dir / f"{split}.pt"
    if cache.exists():
        stored = torch.load(cache, map_location="cpu", weights_only=True)
        return Split(stored["images"], stored["targets"], stored["groups"])
    decoded = _decode(download_split(split, data_dir))
    torch.save(
        {
            "images": decoded.images,
            "targets": decoded.targets,
            "groups": decoded.groups,
        },
        cache,
    )
    return decoded


def load_waterbirds(data_dir: Path = DATA_DIR) -> Waterbirds:
    """Load the three splits of Waterbirds."""
    return Waterbirds(*(load_split(split, data_dir) for split in SPLITS))


def _synthetic_split(
    counts: tuple[int, ...], size: int, generator: torch.Generator
) -> Split:
    """Draw noisy images whose label and background shift two channels."""
    groups = torch.repeat_interleave(
        torch.arange(NUM_GROUPS), torch.tensor(counts, dtype=torch.int64)
    )
    targets = groups // 2
    places = groups % 2
    noise = torch.randint(0, 96, (len(groups), 3, size, size), generator=generator)
    shift = torch.zeros(len(groups), 3, 1, 1, dtype=torch.int64)
    shift[:, 0, 0, 0] = 64 * targets
    shift[:, 1, 0, 0] = 128 * places
    return Split((noise + shift).to(torch.uint8), targets, groups)


def synthetic_waterbirds(size: int = 32, seed: int = 0) -> Waterbirds:
    """Draw a small dataset with the group structure of Waterbirds.

    The training set is skewed toward the groups where label and background
    agree, and the validation and test sets are balanced across
    backgrounds, as in Appendix C.1.

    Args:
        size: Height and width of every image.
        seed: Seed of the draws.

    Returns:
        Training, validation and test splits.
    """
    generator = torch.Generator().manual_seed(seed)
    return Waterbirds(
        train=_synthetic_split((40, 4, 4, 16), size, generator),
        validation=_synthetic_split((6, 6, 4, 4), size, generator),
        test=_synthetic_split((6, 6, 4, 4), size, generator),
    )
