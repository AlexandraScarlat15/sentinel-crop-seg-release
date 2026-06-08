import csv

import numpy as np
import pytest

from src.training.dataset_manifest import ManifestSegmentationDataset


def create_manifest_dataset(tmp_path):
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

        rows.append(
            {
                "dataset": "tiny_manifest_dataset",
                "sample_id": sample_id,
                "img_path": str(img_path),
                "mask_path": str(mask_path),
                "split": split,
            }
        )

    manifest_csv = tmp_path / "manifest.csv"

    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "sample_id", "img_path", "mask_path", "split"],
        )
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
