import numpy as np
import pytest

from sentinel_crop_seg.core.training.dataset import NpySegmentationDataset, SegmentationPatchDataset


def create_tiny_npy_dataset(root, num_samples=3, channels=10, height=8, width=8):
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(num_samples):
        image = np.stack(
            [
                np.full(
                    (height, width),
                    (channel_idx + 1) * 1000 + sample_idx,
                    dtype=np.float32,
                )
                for channel_idx in range(channels)
            ],
            axis=0,
        )

        mask = np.full((height, width), sample_idx % 11, dtype=np.uint8)

        np.save(images_dir / f"sample_{sample_idx:06d}.npy", image)
        np.save(masks_dir / f"sample_{sample_idx:06d}.npy", mask)


def test_npy_segmentation_dataset_loads_samples_with_band_selection_and_scaling(tmp_path):
    dataset_root = tmp_path / "tiny_dataset"
    create_tiny_npy_dataset(dataset_root)

    dataset = NpySegmentationDataset(
        dataset_root,
        limit=2,
        shuffle=False,
        band_indices=[0, 3, 9],
        normalize_mode="scale_10000",
        scale_divisor=10000.0,
    )

    assert len(dataset) == 2
    assert dataset.describe() == {
        "num_samples": 2,
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


def test_segmentation_patch_dataset_filters_allowed_sample_names(tmp_path):
    dataset_root = tmp_path / "tiny_dataset"
    create_tiny_npy_dataset(dataset_root)

    dataset = SegmentationPatchDataset(
        dataset_root,
        shuffle=False,
        allowed_sample_names=["sample_000001.npy"],
    )

    assert len(dataset) == 1
    assert dataset.samples[0].image_path.name == "sample_000001.npy"

    image, mask = dataset[0]

    assert image.shape == (10, 8, 8)
    assert mask.shape == (8, 8)
    assert np.all(mask == 1)


def test_segmentation_patch_dataset_filters_allowed_sample_stems(tmp_path):
    dataset_root = tmp_path / "tiny_dataset"
    create_tiny_npy_dataset(dataset_root)

    dataset = SegmentationPatchDataset(
        dataset_root,
        shuffle=False,
        allowed_sample_names=["sample_000001"],
    )

    assert len(dataset) == 1
    assert dataset.samples[0].image_path.name == "sample_000001.npy"


def test_segmentation_patch_dataset_filters_allowed_nested_sample_names(tmp_path):
    root = tmp_path / "root"
    dataset_root = root / "AT_2021_beet_v2"

    create_tiny_npy_dataset(dataset_root)

    dataset = SegmentationPatchDataset(
        root,
        shuffle=False,
        allowed_sample_names=["AT_2021_beet_v2/sample_000001.npy"],
    )

    assert len(dataset) == 1
    assert dataset.samples[0].image_path.name == "sample_000001.npy"


def test_segmentation_patch_dataset_raises_when_allowed_samples_do_not_match(tmp_path):
    dataset_root = tmp_path / "tiny_dataset"
    create_tiny_npy_dataset(dataset_root)

    with pytest.raises(FileNotFoundError):
        SegmentationPatchDataset(
            dataset_root,
            shuffle=False,
            allowed_sample_names=["sample_999999.npy"],
        )