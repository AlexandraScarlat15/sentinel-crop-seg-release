"""Offline tests for the DataEngine v3 log -> geospatial split-manifest contract."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sentinel_crop_seg.core.data_engine import PATCH_LOG_COLUMNS
from sentinel_crop_seg.core.data.patch_log_manifest import adapt_patch_log, adapt_patch_log_file
from sentinel_crop_seg.core.data.patch_io import load_image_patch, load_mask_patch
from sentinel_crop_seg.core.geospatial.geotiff_writer import save_mask_geotiff, save_multiband_geotiff
from sentinel_crop_seg.core.splitting.spatial_split import run_from_files
from sentinel_crop_seg.core.training.dataset_manifest import ManifestSegmentationDataset


def synthetic_log(tmp_path, n=12, *, with_geotiff=False):
    """Create a full v3 log, 8x8 pixel assets, and separated spatial footprints."""
    from rasterio.transform import from_origin

    root = tmp_path / "generated"
    for name in ("images", "masks", "images_tif", "masks_tif"):
        (root / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        name = f"sample_{index:06d}"
        x, y = 4_000_000 + index * 80_000, 3_000_000
        image = np.full((10, 8, 8), index + 1, dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1, 2] = 3
        np.save(root / "images" / f"{name}.npy", image)
        np.save(root / "masks" / f"{name}.npy", mask)
        if with_geotiff:
            transform = from_origin(x, y + 80, 10, 10)
            save_multiband_geotiff(
                root / "images_tif" / f"{name}.tif",
                image, transform, "EPSG:3035",
            )
            save_mask_geotiff(
                root / "masks_tif" / f"{name}.tif",
                mask, transform, "EPSG:3035",
            )
        record = {column: "" for column in PATCH_LOG_COLUMNS}
        record.update({
            "log_schema_version": "3",
            "sample_id": name,
            "product_id": f"S2A_MSIL2A_20210401T000000_N0300_R001_T33TWN_20210402T000000_{index}",
            "safe_id": f"SAFE_TEST_{index}",
            "scene_date": "2021-04-01",
            "tile_id": "T33TWN",
            "roi_idx": "1",
            "col": str(index),
            "row": "0",
            "global_col": str(index * 8),
            "global_row": "0",
            "size": "8",
            "image_npy_path": f"images/{name}.npy",
            "mask_npy_path": f"masks/{name}.npy",
            "image_tif_path": f"images_tif/{name}.tif",
            "mask_tif_path": f"masks_tif/{name}.tif",
            "image_formats": "npy;geotiff",
            "mask_formats": "npy;geotiff",
            "crs": "EPSG:3035",
            "transform_a": "10",
            "transform_b": "0",
            "transform_c": str(x),
            "transform_d": "0",
            "transform_e": "-10",
            "transform_f": str(y + 80),
            "min_x": str(x),
            "min_y": str(y),
            "max_x": str(x + 80),
            "max_y": str(y + 80),
            "min_lng": "13.0",
            "min_lat": "47.0",
            "max_lng": "13.01",
            "max_lat": "47.01",
            "band_order": "B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12",
            "class_pixel_counts": '{"0":63,"3":1}',
            "foreground_ratio": "0.0156",
            "nodata_ratio": "0.0",
            "cloud_ratio": "0.0",
        })
        rows.append(record)
    log = root / "patches_log.csv"
    pd.DataFrame(rows, columns=PATCH_LOG_COLUMNS).to_csv(log, index=False)
    return log


def load_raw(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_adapter_preserves_every_v3_column_and_adds_spatial_metadata(tmp_path):
    source = synthetic_log(tmp_path, n=2)
    raw = load_raw(source)
    adapted = adapt_patch_log(
        raw, dataset_id="AT_2021", country="AT", cohort="AT_2021",
        label_source="eurocrops", label_source_ref="AT_EC21_vector_snapshot",
    )
    assert set(raw.columns).issubset(adapted.columns)
    for column in raw.columns:
        assert raw[column].tolist() == adapted[column].tolist(), column
    assert adapted["dataset"].tolist() == ["AT_2021"] * 2
    assert adapted["epsg"].tolist() == [3035] * 2
    assert adapted.loc[0, "bbox_minx"] == 4_000_000
    assert adapted.loc[0, "bbox_maxy"] == 3_000_080
    assert adapted["acquisition_date"].tolist() == ["2021-04-01"] * 2
    assert adapted["label_source_ref"].tolist() == ["AT_EC21_vector_snapshot"] * 2
    assert adapted["p0"].tolist() == [63, 63]
    assert adapted["p3"].tolist() == [1, 1]
    assert adapted["img_path"].iloc[0] == raw["image_npy_path"].iloc[0]


def test_no_invented_source_or_acquisition_date(tmp_path):
    raw = load_raw(synthetic_log(tmp_path, n=2))
    raw.loc[0, ["product_id", "safe_id", "scene_date", "tile_id"]] = ""
    frame = adapt_patch_log(raw, dataset_id="AT_2021")
    assert frame.loc[0, "acquisition_date"] == ""
    assert frame.loc[0, "product_id"] == ""
    assert "label_source" not in frame
    assert "label_source_ref" not in frame
    assert "country" not in frame
    assert "cohort" not in frame


@pytest.mark.parametrize(("field", "value", "error"), [
    ("log_schema_version", "2", "schema v3"),
    ("crs", "EPSG:4326", "projected EPSG"),
    ("crs", "NOT_A_CRS", "unrecognized CRS"),
    ("max_x", "-123", "invalid projected bounds"),
    ("scene_date", "unknown", "scene_date"),
    ("class_pixel_counts", '{"0":-3}', "invalid class id/count"),
    ("class_pixel_counts", "not-json", "invalid class_pixel_counts"),
    ("image_npy_path", "", "Missing image path"),
])
def test_rejects_malformed_v3_rows(tmp_path, field, value, error):
    raw = load_raw(synthetic_log(tmp_path, n=1))
    raw.loc[0, field] = value
    with pytest.raises(ValueError, match=error):
        adapt_patch_log(raw, dataset_id="AT_2021")


def test_rejects_missing_fields_and_duplicate_sample_ids(tmp_path):
    raw = load_raw(synthetic_log(tmp_path, n=2))
    with pytest.raises(ValueError, match="missing v3 columns"):
        adapt_patch_log(raw.drop(columns="safe_id"), dataset_id="AT_2021")
    raw.loc[1, "sample_id"] = raw.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        adapt_patch_log(raw, dataset_id="AT_2021")


def test_rejects_unsupported_or_unannounced_asset_format(tmp_path):
    raw = load_raw(synthetic_log(tmp_path, n=1))
    with pytest.raises(ValueError, match="asset_format"):
        adapt_patch_log(raw, dataset_id="AT_2021", asset_format="jpg")
    raw.loc[0, "image_formats"] = "npy"
    with pytest.raises(ValueError, match="does not advertise geotiff"):
        adapt_patch_log(raw, dataset_id="AT_2021", asset_format="geotiff")


def test_label_source_reference_must_be_explicit(tmp_path):
    raw = load_raw(synthetic_log(tmp_path, n=1))
    with pytest.raises(ValueError, match="label_source_ref"):
        adapt_patch_log(raw, dataset_id="AT_2021", label_source_ref="guessed")
    with pytest.raises(ValueError, match="dataset_id"):
        adapt_patch_log(raw, dataset_id="")


def test_geotiff_selection_points_to_georeferenced_equal_arrays(tmp_path):
    source = synthetic_log(tmp_path, n=1, with_geotiff=True)
    raw = load_raw(source)
    tif = adapt_patch_log(raw, dataset_id="AT_2021", asset_format="geotiff")
    assert tif.loc[0, "img_path"] == raw.loc[0, "image_tif_path"]
    assert tif.loc[0, "mask_path"] == raw.loc[0, "mask_tif_path"]
    image = load_image_patch(source.parent / tif.loc[0, "img_path"])
    mask = load_mask_patch(source.parent / tif.loc[0, "mask_path"])
    assert np.array_equal(image, np.load(source.parent / raw.loc[0, "image_npy_path"]))
    assert np.array_equal(mask, np.load(source.parent / raw.loc[0, "mask_npy_path"]))


def test_adapter_file_preserves_source_hash_and_never_overwrites(tmp_path):
    source = synthetic_log(tmp_path)
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "manifests" / "adapted.csv"
    preview = adapt_patch_log_file(
        source, target, dataset_id="AT_2021", asset_format="npy",
        country="AT", cohort="AT_2021", verify_assets=True, validate_only=True,
    )
    assert preview["input_log_sha256"] == fingerprint
    assert preview["verified_asset_paths"]
    assert not target.exists()
    summary = adapt_patch_log_file(
        source, target, dataset_id="AT_2021", asset_format="npy",
        country="AT", cohort="AT_2021", verify_assets=True,
    )
    assert target.exists()
    assert summary["samples"] == 12
    audit = json.loads(target.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert audit["original_columns_preserved"] == PATCH_LOG_COLUMNS
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fingerprint
    with pytest.raises(FileExistsError, match="already exists"):
        adapt_patch_log_file(source, target, dataset_id="AT_2021")


def test_adapter_to_spatial_split_to_real_dataloader(tmp_path):
    source = synthetic_log(tmp_path)
    adapter_output = tmp_path / "converted" / "from_log.csv"
    adapt_patch_log_file(
        source, adapter_output, dataset_id="AT_2021",
        country="AT", cohort="AT_2021", verify_assets=True,
    )
    adapted = load_raw(adapter_output)
    original = load_raw(source)
    assert adapted["product_id"].tolist() == original["product_id"].tolist()
    assert adapted["safe_id"].tolist() == original["safe_id"].tolist()
    assert adapted["tile_id"].tolist() == original["tile_id"].tolist()

    config = tmp_path / "spatial.yaml"
    config.write_text("splitting:\n  allocation:\n    candidate_seeds: 30\n", encoding="utf-8")
    output = tmp_path / "spatial_outputs" / "split_at_2021"
    summary = run_from_files(config, adapter_output, output)
    assert summary["leakage_audit_clean"]
    final = load_raw(output / "split_manifest.csv")
    for column in original.columns:
        assert final[column].tolist() == original[column].tolist(), column
    for partition in ("train", "val", "test"):
        loader = ManifestSegmentationDataset(
            manifest_csv=output / "split_manifest.csv",
            split=partition, shuffle=False, limit=1,
        )
        image, mask = loader[0]
        assert image.shape == (10, 8, 8)
        assert mask.shape == (8, 8)
    assert (output / "class_distribution.csv").is_file()


def test_verify_assets_fails_before_output_if_missing(tmp_path):
    source = synthetic_log(tmp_path, n=1)
    image = source.parent / "images" / "sample_000000.npy"
    image.unlink()
    output = tmp_path / "adapted.csv"
    with pytest.raises(FileNotFoundError, match="Unresolved img_path"):
        adapt_patch_log_file(
            source, output, dataset_id="AT_2021", verify_assets=True,
        )
    assert not output.exists()
