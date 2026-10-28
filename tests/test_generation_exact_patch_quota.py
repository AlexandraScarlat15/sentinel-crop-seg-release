"""Offline regression coverage for exact accepted-patch quotas.

Uses a tiny in-memory Sentinel raster and EuroCrops-like geometry.
Ensures NPY, GeoTIFF and patches_log.csv v3 all contain exactly the accepted
patches, without touching the historical dataset.
"""
from __future__ import annotations

import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from sentinel_crop_seg.core.data_engine import DataEngine, PATCH_LOG_COLUMNS


class FakeSentinel:
    def __init__(self):
        self.transform = from_origin(500_000, 4_600_000, 10, 10)

    def get_spatial_metadata(self):
        return {
            "crs": "EPSG:32633",
            "transform": self.transform,
            "width": 16,
            "height": 8,
        }

    def read_window(self, window, bands):
        col, row, width, height = map(int, window)
        yy, xx = np.indices((height, width))
        return np.stack([
            (100 + xx + col + (yy + row) * 16 + channel).astype(np.uint16)
            for channel in range(len(bands))
        ])

    def read_scl(self, window):
        return np.full((int(window[3]), int(window[2])), 4, dtype=np.uint8)


class FakeEuroCrops:
    crop_column = "crop"

    def __init__(self):
        roi = box(500_000, 4_599_920, 500_160, 4_600_000)
        self.gdf = gpd.GeoDataFrame(
            {"crop": ["maize"]}, geometry=[roi], crs="EPSG:32633"
        )

    def map_crop_to_class_id(self, crop):
        return 1 if crop == "maize" else 0


ROI = (500_000, 4_599_920, 500_160, 4_600_000)
BANDS = ["B02", "B03"]
SOURCE_PRODUCT = "S2B_MSIL2A_20210810T095029_N0500_R079_T33TWN_20230211T233939"


def engine(*, roi_cap=6):
    return DataEngine(
        config={
            "dataset": {
                "patch": {"size": 4, "stride": 4},
                "max_patches_per_roi": roi_cap,
                "min_foreground_ratio": 0,
                "max_nodata_fraction": 0,
                "max_cloud_fraction": 0,
                "max_cloud_on_foreground_fraction": 0,
                "cloud_dilate_radius": 0,
                "use_b02_brightness_filter": False,
            }
        },
        sentinel_loader=FakeSentinel(),
        mask_loader=FakeEuroCrops(),
    )


def generate(tmp_path, *, cap, rois=None, roi_cap=6, target=1):
    return engine(roi_cap=roi_cap).save_dataset(
        output_dir=str(tmp_path / "output"),
        rois=rois if rois is not None else [ROI],
        bands=BANDS,
        target_class_id=target,
        min_target_pixels=1,
        source_product_id=SOURCE_PRODUCT,
        source_safe_id=SOURCE_PRODUCT + ".SAFE",
        max_new_patches=cap,
    )


def assert_storage_integrity(out: Path, expected: int):
    images = sorted((out / "images").glob("sample_*.npy"))
    masks = sorted((out / "masks").glob("sample_*.npy"))
    images_tif = sorted((out / "images_tif").glob("sample_*.tif"))
    masks_tif = sorted((out / "masks_tif").glob("sample_*.tif"))
    assert len(images) == len(masks) == len(images_tif) == len(masks_tif) == expected
    with (out / "patches_log.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PATCH_LOG_COLUMNS
        rows = list(reader)
    assert len(rows) == expected
    ids = [row["sample_id"] for row in rows]
    assert ids == [f"sample_{n:06d}" for n in range(expected)]
    assert all(row["log_schema_version"] == "3" for row in rows)
    assert all(row["product_id"] == SOURCE_PRODUCT for row in rows)
    for row in rows:
        sid = row["sample_id"]
        image = np.load(out / "images" / f"{sid}.npy")
        mask = np.load(out / "masks" / f"{sid}.npy")
        with rasterio.open(out / "images_tif" / f"{sid}.tif") as tif:
            np.testing.assert_array_equal(tif.read(), image)
            assert tif.crs.to_epsg() == 32633
        with rasterio.open(out / "masks_tif" / f"{sid}.tif") as tif:
            np.testing.assert_array_equal(tif.read(1), mask)
            assert tif.crs.to_epsg() == 32633
    return rows


def test_exact_global_cap_stops_at_final_accepted_patch(tmp_path):
    assert generate(tmp_path, cap=3) == 3
    rows = assert_storage_integrity(tmp_path / "output", 3)
    assert rows[0]["sample_id"] == "sample_000000"
    assert rows[-1]["sample_id"] == "sample_000002"


def test_remaining_cap_across_multiple_rois_preserves_per_roi_limit(tmp_path):
    assert generate(tmp_path, cap=7, rois=[ROI, ROI], roi_cap=5) == 7
    rows = assert_storage_integrity(tmp_path / "output", 7)
    assert [row["roi_idx"] for row in rows] == ["1"] * 5 + ["2"] * 2


def test_legacy_none_cap_still_obeys_per_roi_limit(tmp_path):
    assert generate(tmp_path, cap=None, roi_cap=4) == 4
    assert_storage_integrity(tmp_path / "output", 4)


def test_appending_only_counts_newly_written_patches(tmp_path):
    assert generate(tmp_path, cap=3) == 3
    assert generate(tmp_path, cap=2) == 2
    assert_storage_integrity(tmp_path / "output", 5)


def test_zero_cap_skips_all_writes(tmp_path):
    assert generate(tmp_path, cap=0) == 0
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "3"])
def test_rejects_invalid_cap_before_writing(tmp_path, invalid):
    with pytest.raises(ValueError, match="max_new_patches"):
        generate(tmp_path, cap=invalid)
    assert not (tmp_path / "output").exists()


def test_only_accepted_patches_consume_quota(tmp_path):
    # Rejected masks must not be counted toward the cap.
    assert generate(tmp_path, cap=3, target=2) == 0
    assert generate(tmp_path, cap=3, target=1) == 3
    assert_storage_integrity(tmp_path / "output", 3)
