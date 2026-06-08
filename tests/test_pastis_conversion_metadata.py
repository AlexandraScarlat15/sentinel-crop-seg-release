from __future__ import annotations

import csv
from pathlib import Path

from src.external.pastis_conversion import (
    save_metadata_csv,
    scene_date_from_yyyymmdd,
)


def test_scene_date_from_yyyymmdd():
    assert scene_date_from_yyyymmdd(20180912) == "2018-09-12"
    assert scene_date_from_yyyymmdd("20210405") == "2021-04-05"


def test_save_metadata_csv_writes_scene_date_column(tmp_path: Path):
    metadata_csv = tmp_path / "metadata.csv"

    row = {
        "sample_name": "sample_000001.npy",
        "patch_id": "123",
        "fold": "1",
        "tile": "T31TCJ",
        "acquisition_index": 4,
        "acquisition_date": 20180912,
        "scene_date": "2018-09-12",
        "year": 2018,
        "month": 9,
        "day": 12,
        "source_s2": "S2_123.npy",
        "source_target": "TARGET_123.npy",
        "image_path": "images/sample_000001.npy",
        "mask_path": "masks/sample_000001.npy",
        "height": 128,
        "width": 128,
        "num_channels": 10,
        "band_reorder": "null",
        "input_band_order": "[]",
        "output_band_order": "[]",
        "total_pixels": 16384,
        "valid_pixels": 16000,
        "background_pixels": 1000,
        "nonbackground_valid_pixels": 15000,
        "ignore_pixels": 384,
        "valid_ratio_total": 0.9765625,
        "ignore_ratio_total": 0.0234375,
        "interest_ratio_total": 0.5,
        "interest_ratio_valid": 0.6,
        "bright_ratio": 0.0,
        "very_bright_ratio": 0.0,
        "white_cloud_ratio": 0.0,
        "dark_ratio": 0.0,
        "rgb_std": 0.1,
        "rgb_mean": 0.2,
        "ndvi_mean": 0.3,
        "vegetation_ratio": 0.4,
        "cloud_score": 0.0,
        "present_project_classes": "[]",
        "project_value_counts": "{}",
        "present_pastis_semantic_classes": "[]",
        "pastis_semantic_value_counts": "{}",
        "n_parcel": 0,
        "parcel_cover": 0,
    }

    save_metadata_csv(metadata_csv, [row])

    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["sample_name"] == "sample_000001.npy"
    assert rows[0]["acquisition_date"] == "20180912"
    assert rows[0]["scene_date"] == "2018-09-12"