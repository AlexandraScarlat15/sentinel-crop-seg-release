import csv

import numpy as np
import pytest

from sentinel_crop_seg.core.training.dataset_manifest import ManifestSegmentationDataset


def create_manifest_dataset(tmp_path, *, include_acquisition_date=False):
    data_root = tmp_path / "data"
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for sample_idx, split in enumerate(["train", "train", "val", "test"]):
        sample_id = f"sample_{sample_idx:06d}"

        image = np.stack(
            [
                np.full((8, 8), (channel_idx + 1) * 1000 + sample_idx, dtype=np.float32)
                for channel_idx in range(10)
            ],
            axis=0,
        )

        mask = np.full((8, 8), sample_idx % 11, dtype=np.uint8)

        img_path = images_dir / f"{sample_id}.npy"
        mask_path = masks_dir / f"{sample_id}.npy"

        np.save(img_path, image)
        np.save(mask_path, mask)

        row = {
            "dataset": "tiny_manifest_dataset",
            "sample_id": sample_id,
            "img_path": str(img_path),
            "mask_path": str(mask_path),
            "split": split,
        }
        if include_acquisition_date:
            row["acquisition_date"] = "2021-04-01"
        rows.append(row)

    manifest_csv = tmp_path / "manifest.csv"

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return manifest_csv


def test_manifest_dataset_loads_selected_split_with_band_selection_and_scaling(tmp_path):
    manifest_csv = create_manifest_dataset(tmp_path)

    dataset = ManifestSegmentationDataset(
        manifest_csv=manifest_csv,
        split="train",
        limit=1,
        shuffle=False,
        band_indices=[0, 3, 9],
        normalize_mode="scale_10000",
        scale_divisor=10000.0,
    )

    assert len(dataset) == 1
    assert dataset.describe() == {
        "num_samples": 1,
        "num_channels": 3,
    }

    image, mask = dataset[0]

    assert image.shape == (3, 8, 8)
    assert mask.shape == (8, 8)

    assert image.dtype == np.float32
    assert mask.dtype == np.int64

    assert np.isclose(image[0, 0, 0], 0.1)
    assert np.isclose(image[1, 0, 0], 0.4)
    assert np.isclose(image[2, 0, 0], 1.0)

    assert np.all(mask == 0)


def test_manifest_dataset_rejects_invalid_split(tmp_path):
    manifest_csv = create_manifest_dataset(tmp_path)

    with pytest.raises(ValueError, match="split must be one of"):
        ManifestSegmentationDataset(
            manifest_csv=manifest_csv,
            split="invalid",
            shuffle=False,
        )


def test_manifest_dataset_appends_cyclical_calendar_channels(tmp_path):
    manifest_csv = create_manifest_dataset(tmp_path, include_acquisition_date=True)
    dataset = ManifestSegmentationDataset(
        manifest_csv=manifest_csv,
        split="train",
        shuffle=False,
        band_indices=list(range(10)),
        normalize_mode="scale_10000",
        calendar_features="doy_sin_cos",
    )
    image, _ = dataset[0]
    phase = 2.0 * np.pi * 91.0 / 365.25
    assert image.shape == (12, 8, 8)
    assert dataset.describe()["num_channels"] == 12
    assert np.allclose(image[10], np.sin(phase))
    assert np.allclose(image[11], np.cos(phase))


def test_manifest_dataset_identifies_zero_byte_mask_in_load_error(tmp_path):
    manifest_csv = create_manifest_dataset(tmp_path)
    first_mask = tmp_path / "data" / "masks" / "sample_000000.npy"
    first_mask.write_bytes(b"")
    dataset = ManifestSegmentationDataset(manifest_csv=manifest_csv, split="train", shuffle=False)

    with pytest.raises(
        RuntimeError,
        match=r"historical mask.*sample_id='sample_000000'.*resolved_mask_path=",
    ) as error:
        _ = dataset[0]

    assert isinstance(error.value.__cause__, EOFError)
