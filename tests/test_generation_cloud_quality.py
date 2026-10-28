"""End-to-end synthetic checks for patch cloud/foreground quality filtering.

These tests exercise DataEngine.save_dataset and the v3 patch log, without
Sentinel downloads or access to the historical crop-segmentation dataset.
"""
from __future__ import annotations

import csv

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from sentinel_crop_seg.core.data_engine import DataEngine, _dilate_bool_mask


class CloudSentinel:
    def __init__(self, *, bad_code=9, bad_col=1, bad_row=1,
                 with_cloud=True, with_nodata=False, with_bright=False):
        self.transform = from_origin(500000, 4600000, 10, 10)
        self.scl = np.full((8, 16), 4, dtype=np.uint8)
        if with_cloud:
            self.scl[bad_row, bad_col] = bad_code
        self.with_nodata = with_nodata
        self.with_bright = with_bright

    def get_spatial_metadata(self):
        return {
            "crs": "EPSG:32633", "transform": self.transform,
            "height": 8, "width": 16,
        }

    def read_window(self, window, bands):
        col, row, width, height = map(int, window)
        yy, xx = np.indices((height, width))
        data = np.stack([
            (300 + xx + col + (yy + row) * 16 + channel).astype(np.uint16)
            for channel in range(len(bands))
        ])
        if self.with_nodata and row == 0 and col == 0:
            data[:, 1, 1] = 0
        if self.with_bright and row == 0 and col == 0:
            data[0, 1, 1] = 5000
        return data

    def read_scl(self, window):
        col, row, width, height = map(int, window)
        return self.scl[row:row+height, col:col+width]


class CloudEuroCrops:
    crop_column = "crop"

    def __init__(self, *, half_mask=True):
        # The first 4x4 sample has foreground on its left half. This lets
        # us test clouds outside vs. inside the target crop independently.
        xmax = 500020 if half_mask else 500160
        self.gdf = gpd.GeoDataFrame(
            {"crop": ["maize"]},
            geometry=[box(500000, 4599920, xmax, 4600000)],
            crs="EPSG:32633",
        )

    def map_crop_to_class_id(self, value):
        return 1 if value == "maize" else 0


ROI = (500000, 4599920, 500160, 4600000)


def run_case(tmp_path, *, sentinel=None, settings=None, half_mask=True, cap=1):
    config = {
        "dataset": {
            "patch": {"size": 4, "stride": 4},
            "max_patches_per_roi": 10,
            "min_foreground_ratio": 0.0,
            "max_nodata_fraction": 0.0,
            "max_cloud_fraction": 0.10,
            "max_cloud_on_foreground_fraction": 0.05,
            "cloud_dilate_radius": 0,
            "use_b02_brightness_filter": False,
        }
    }
    config["dataset"].update(settings or {})
    engine = DataEngine(
        config=config, sentinel_loader=sentinel or CloudSentinel(),
        mask_loader=CloudEuroCrops(half_mask=half_mask),
    )
    output = tmp_path / "out"
    accepted = engine.save_dataset(
        output_dir=str(output), rois=[ROI], bands=["B02", "B03"],
        max_new_patches=cap,
    )
    with (output / "patches_log.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert accepted == len(rows)
    assert accepted == len(list((output / "images").glob("*.npy")))
    assert accepted == len(list((output / "masks").glob("*.npy")))
    assert accepted == len(list((output / "images_tif").glob("*.tif")))
    assert accepted == len(list((output / "masks_tif").glob("*.tif")))
    return rows


@pytest.mark.parametrize("code", [0, 1, 2, 3, 7, 8, 9, 10, 11])
def test_default_scl_bad_classes_reject_first_sample(tmp_path, code):
    rows = run_case(tmp_path, sentinel=CloudSentinel(bad_code=code))
    assert len(rows) == 1
    assert rows[0]["col"] == "4"
    assert float(rows[0]["cloud_ratio"]) == pytest.approx(0)


@pytest.mark.parametrize("code", [4, 5, 6])
def test_healthy_scl_classes_are_not_treated_as_clouds(tmp_path, code):
    rows = run_case(tmp_path, sentinel=CloudSentinel(bad_code=code))
    assert rows[0]["col"] == "0"
    assert float(rows[0]["cloud_ratio"]) == pytest.approx(0)


def test_cloud_outside_crop_foreground_is_accepted(tmp_path):
    rows = run_case(
        tmp_path, sentinel=CloudSentinel(bad_col=3),
        settings={"max_cloud_fraction": 0.10,
                  "max_cloud_on_foreground_fraction": 0.01},
    )
    assert rows[0]["col"] == "0"
    assert float(rows[0]["cloud_ratio"]) == pytest.approx(1/16, abs=0.0001)
    assert float(rows[0]["cloud_on_fg_ratio"]) == pytest.approx(0)


def test_cloud_inside_crop_foreground_fails_target_specific_threshold(tmp_path):
    rows = run_case(
        tmp_path, sentinel=CloudSentinel(bad_col=1),
        settings={"max_cloud_fraction": 0.10,
                  "max_cloud_on_foreground_fraction": 0.01},
    )
    assert rows[0]["col"] == "4"


def test_dilation_expands_cloud_rejection_zone(tmp_path):
    cloud = np.zeros((4, 4), dtype=bool)
    cloud[1, 1] = True
    assert _dilate_bool_mask(cloud, 0).sum() == 1
    assert _dilate_bool_mask(cloud, 1).sum() == 9
    rows = run_case(
        tmp_path, sentinel=CloudSentinel(),
        settings={"max_cloud_fraction": 0.10,
                  "max_cloud_on_foreground_fraction": 1.0,
                  "cloud_dilate_radius": 1},
    )
    assert rows[0]["col"] == "4"


def test_same_cloud_without_dilation_passes_threshold(tmp_path):
    rows = run_case(
        tmp_path, sentinel=CloudSentinel(),
        settings={"max_cloud_fraction": 0.10,
                  "max_cloud_on_foreground_fraction": 1.0,
                  "cloud_dilate_radius": 0},
    )
    assert rows[0]["col"] == "0"


def test_b02_brightness_filter_rejects_when_explicitly_enabled(tmp_path):
    bright = CloudSentinel(with_cloud=False, with_bright=True)
    rows = run_case(
        tmp_path, sentinel=bright,
        settings={"use_b02_brightness_filter": True,
                  "b02_bright_threshold": 2000,
                  "max_b02_bright_fraction": 0.01,
                  "max_b02_bright_on_foreground_fraction": 1.0},
    )
    assert rows[0]["col"] == "4"


def test_b02_brightness_filter_disabled_keeps_bright_sample(tmp_path):
    bright = CloudSentinel(with_cloud=False, with_bright=True)
    rows = run_case(
        tmp_path, sentinel=bright,
        settings={"use_b02_brightness_filter": False},
    )
    assert rows[0]["col"] == "0"


def test_nodata_rejects_before_cloud_evaluation(tmp_path):
    rows = run_case(
        tmp_path,
        sentinel=CloudSentinel(with_cloud=False, with_nodata=True),
    )
    assert rows[0]["col"] == "4"


def test_custom_scl_class_policy_is_respected(tmp_path):
    rows = run_case(
        tmp_path,
        sentinel=CloudSentinel(bad_code=9),
        settings={"bad_scl_classes": [0, 1]},
    )
    assert rows[0]["col"] == "0"


def test_misaligned_scl_is_rejected_before_any_patch_is_written(tmp_path):
    sentinel = CloudSentinel(with_cloud=False)
    sentinel.read_scl = lambda window: np.zeros((2, 2), dtype=np.uint8)
    engine = DataEngine(
        config={"dataset": {
            "patch": {"size": 4, "stride": 4},
            "max_patches_per_roi": 10,
            "min_foreground_ratio": 0.0,
            "max_nodata_fraction": 0.0,
            "max_cloud_fraction": 1.0,
            "max_cloud_on_foreground_fraction": 1.0,
            "cloud_dilate_radius": 0,
            "use_b02_brightness_filter": False,
        }},
        sentinel_loader=sentinel,
        mask_loader=CloudEuroCrops(),
    )
    output = tmp_path / "misaligned"
    with pytest.raises(ValueError, match="SCL patch shape does not match mask"):
        engine.save_dataset(
            output_dir=str(output), rois=[ROI], bands=["B02", "B03"],
            max_new_patches=1,
        )
    assert list((output / "images").glob("sample_*.npy")) == []
    assert list((output / "masks").glob("sample_*.npy")) == []
