from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from sentinel_crop_seg.core.geospatial.geotiff_writer import save_mask_geotiff, save_multiband_geotiff
from sentinel_crop_seg.core.training.dataset import SegmentationPatchDataset


def test_segmentation_patch_dataset_loads_equivalent_npy_and_geotiff_samples(tmp_path: Path):
    dataset_root = tmp_path / "dataset"

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_tif_dir = dataset_root / "images_tif"
    masks_tif_dir = dataset_root / "masks_tif"

    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    images_tif_dir.mkdir(parents=True)
    masks_tif_dir.mkdir(parents=True)

    sample_name = "sample_000000"

    image = np.arange(10 * 16 * 16, dtype=np.uint16).reshape(10, 16, 16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 5:13] = 3

    np.save(images_dir / f"{sample_name}.npy", image)
    np.save(masks_dir / f"{sample_name}.npy", mask)

    transform = from_origin(500000, 4600000, 10, 10)
    crs = "EPSG:32633"
    band_names = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]

    save_multiband_geotiff(
        output_path=images_tif_dir / f"{sample_name}.tif",
        image=image,
        transform=transform,
        crs=crs,
        band_names=band_names,
    )

    save_mask_geotiff(
        output_path=masks_tif_dir / f"{sample_name}.tif",
        mask=mask,
        transform=transform,
        crs=crs,
    )

    npy_dataset = SegmentationPatchDataset(
        dataset_root,
        sample_format="npy",
        shuffle=False,
        normalize_mode="none",
    )

    geotiff_dataset = SegmentationPatchDataset(
        dataset_root,
        sample_format="geotiff",
        shuffle=False,
        normalize_mode="none",
    )

    assert npy_dataset.describe() == {
        "num_samples": 1,
        "num_channels": 10,
    }

    assert geotiff_dataset.describe() == {
        "num_samples": 1,
        "num_channels": 10,
    }

    assert npy_dataset.describe_storage() == {
        "sample_format": "npy",
        "image_suffix": ".npy",
        "mask_suffix": ".npy",
    }

    assert geotiff_dataset.describe_storage() == {
        "sample_format": "geotiff",
        "image_suffix": ".tif",
        "mask_suffix": ".tif",
    }

    x_npy, y_npy = npy_dataset[0]
    x_tif, y_tif = geotiff_dataset[0]

    assert x_npy.shape == (10, 16, 16)
    assert y_npy.shape == (16, 16)

    assert x_npy.dtype == np.float32
    assert x_tif.dtype == np.float32
    assert y_npy.dtype == np.int64
    assert y_tif.dtype == np.int64

    assert np.array_equal(x_npy, x_tif)
    assert np.array_equal(y_npy, y_tif)