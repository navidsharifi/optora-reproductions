from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from ..data import (
    CROP,
    NUM_GROUPS,
    SHA256,
    _decode,
    download_split,
    group_counts,
    load_split,
    preprocess,
    sha256,
    synthetic_waterbirds,
)


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    from PIL import Image

    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(
        0, 256, (height, width, 3), generator=generator, dtype=torch.uint8
    )
    buffer = io.BytesIO()
    Image.fromarray(pixels.numpy()).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.parametrize(("width", "height"), [(500, 375), (224, 224), (180, 400)])
def test_preprocess_matches_the_authors_transform(width: int, height: int) -> None:
    from PIL import Image
    from torchvision import transforms

    jpeg = _jpeg(width, height)
    scale = 256.0 / 224.0
    authors = transforms.Compose(
        [
            transforms.Resize((int(224 * scale), int(224 * scale))),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )
    with Image.open(io.BytesIO(jpeg)) as image:
        expected = authors(image.convert("RGB"))

    result = preprocess(jpeg)

    assert result.dtype == torch.uint8
    assert result.shape == (3, CROP, CROP)
    torch.testing.assert_close(result.float() / 255.0, expected)


def test_preprocess_converts_grayscale_to_rgb() -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (300, 300), color=90).save(buffer, format="JPEG")

    image = preprocess(buffer.getvalue())

    assert image.shape == (3, CROP, CROP)
    assert torch.equal(image[0], image[1])


def test_group_counts_count_every_group() -> None:
    groups = torch.tensor([0, 3, 3, 1, 3])
    assert group_counts(groups).tolist() == [1, 1, 0, 3]


def test_synthetic_data_has_the_layout_of_waterbirds() -> None:
    data = synthetic_waterbirds(size=8)

    for split in (data.train, data.validation, data.test):
        assert split.images.dtype == torch.uint8
        assert split.images.shape[1:] == (3, 8, 8)
        assert torch.equal(split.targets, split.groups // 2)
    train = group_counts(data.train.groups)
    assert int(train[0]) > int(train[1]) and int(train[3]) > int(train[2])
    test = group_counts(data.test.groups)
    assert int(test[0]) == int(test[1]) and int(test[2]) == int(test[3])


def test_synthetic_data_is_reproducible() -> None:
    first, second = synthetic_waterbirds(seed=3), synthetic_waterbirds(seed=3)
    assert torch.equal(first.train.images, second.train.images)


def test_decode_reads_labels_places_and_images(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels, places = [0, 1, 1, 0], [0, 0, 1, 1]
    jpegs = [_jpeg(260, 250, seed) for seed in range(4)]
    images = [{"bytes": jpeg, "path": f"{seed}.jpg"} for seed, jpeg in enumerate(jpegs)]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"image": images, "label": labels, "place": places}), path)

    split = _decode(path)

    assert split.targets.tolist() == labels
    assert split.groups.tolist() == [0, 2, 3, 1]
    assert split.images.shape == (4, 3, CROP, CROP)
    assert torch.equal(split.images[2], preprocess(jpegs[2]))


def test_load_split_reads_the_decoded_cache(tmp_path: Path) -> None:
    images = torch.randint(0, 256, (3, 3, 4, 4)).to(torch.uint8)
    targets = torch.tensor([0, 1, 1])
    groups = torch.tensor([0, 2, 3])
    torch.save(
        {"images": images, "targets": targets, "groups": groups}, tmp_path / "test.pt"
    )

    split = load_split("test", tmp_path)

    assert torch.equal(split.images, images)
    assert torch.equal(split.groups, groups)
    assert int(group_counts(split.groups).sum()) == 3 <= NUM_GROUPS


def _parquet(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    images = [
        {"bytes": _jpeg(230, 240, seed), "path": f"{seed}.jpg"} for seed in range(2)
    ]
    pq.write_table(pa.table({"image": images, "label": [1, 0], "place": [0, 1]}), path)


def test_download_verifies_and_caches_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .. import data

    source = tmp_path / "source" / "test-00000-of-00001.parquet"
    source.parent.mkdir()
    _parquet(source)
    monkeypatch.setattr(
        data, "URL", source.parent.as_uri() + "/{split}-00000-of-00001.parquet"
    )
    monkeypatch.setitem(data.SHA256, "test", sha256(source))
    cache = tmp_path / "cache"

    path = download_split("test", cache)

    assert path == cache / "test.parquet"
    assert path.read_bytes() == source.read_bytes()
    assert download_split("test", cache) == path


def test_download_rejects_a_remote_file_with_the_wrong_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .. import data

    source = tmp_path / "test-00000-of-00001.parquet"
    source.write_bytes(b"tampered")
    monkeypatch.setattr(
        data, "URL", tmp_path.as_uri() + "/{split}-00000-of-00001.parquet"
    )
    cache = tmp_path / "cache"

    with pytest.raises(ValueError, match="wrong SHA-256"):
        download_split("test", cache)
    assert list(cache.iterdir()) == []


def test_load_waterbirds_decodes_once_and_then_reads_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .. import data

    def fake_download(split: str, data_dir: Path) -> Path:
        path = data_dir / f"{split}.parquet"
        _parquet(path)
        return path

    monkeypatch.setattr(data, "download_split", fake_download)
    first = data.load_waterbirds(tmp_path)

    def fail(split: str, data_dir: Path) -> Path:
        raise AssertionError("the cache was not used")

    monkeypatch.setattr(data, "download_split", fail)
    second = data.load_waterbirds(tmp_path)

    assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
        "test.pt",
        "train.pt",
        "validation.pt",
    ]
    assert first.test.groups.tolist() == [2, 1]
    assert torch.equal(first.train.images, second.train.images)


def test_download_rejects_an_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        download_split("dev", tmp_path)


def test_download_rejects_a_cached_file_with_the_wrong_digest(tmp_path: Path) -> None:
    (tmp_path / "train.parquet").write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        download_split("train", tmp_path)


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "file"
    path.write_bytes(b"waterbirds" * 1000)
    assert sha256(path) == hashlib.sha256(b"waterbirds" * 1000).hexdigest()
    assert set(SHA256) == {"train", "validation", "test"}
