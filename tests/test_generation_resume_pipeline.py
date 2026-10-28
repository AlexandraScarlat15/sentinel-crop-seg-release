"""Pipeline integration for strict guarded generation using synthetic source files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

from sentinel_crop_seg.core.pipeline import dataset_generation as pipeline
from sentinel_crop_seg.core.pipeline.generation_run_guard import LEDGER_NAME
from sentinel_crop_seg.core.selection.roi_selectors.base import RoiSelectionResult


def setup_fake_pipeline(tmp_path, monkeypatch, *, product_raises=False):
    source = tmp_path / "source"
    source.mkdir()
    shp = source / "BE_2021.shp"
    for ext in (".shp", ".shx", ".dbf", ".prj"):
        shp.with_suffix(ext).write_bytes(b"fixture " + ext.encode())

    output = tmp_path / "out"
    cache_root = tmp_path / "cache"
    config = {
        "eurocrops": {"shapefile_path": str(shp)},
        "classes": {"maize": {"id": 1, "include": ["maize"]}},
        "generation": {"satellite": "sentinel2", "label_source": "eurocrops"},
        "paths": {
            "sentinel_data_root": str(cache_root),
            "dataset_output_dir": str(output),
        },
        "dataset": {
            "roi": {"strategy": "grid"},
            "target": {"class_id": None, "total_patches": 2},
            "bands": {"B02": {"resolution": 10}},
            "min_foreground_ratio": 0.0,
        },
        "catalog": {"stac": {
            "start_date": "2021-04-01", "end_date": "2021-09-30",
            "max_cloud_cover": 10, "limit": 5,
        }},
    }

    class FakeMaskLoader:
        def __init__(self, cfg):
            self.crop_column = "crop"
            self.gdf = gpd.GeoDataFrame(
                {"crop": ["maize"]},
                geometry=[box(500000, 4599900, 500400, 4600000)],
                crs="EPSG:32633",
            )

        def map_crop_to_class_id(self, name):
            return 1

    class FakeSelector:
        def select(self, ctx):
            return RoiSelectionResult(rois=[
                (500000, 4599900, 500100, 4600000),
                (500100, 4599900, 500200, 4600000),
                (500200, 4599900, 500300, 4600000),
            ], strategy="grid_coverage")

    class FakeCatalog:
        def search(self, **kwargs):
            return [SimpleNamespace(product_id="FAKE_PRODUCT")]

    class FakeCache:
        def __init__(self, cfg, processed_state_path):
            self.state = Path(processed_state_path)
            if self.state.exists():
                self.processed = json.loads(self.state.read_text())
            else:
                self.processed = {}

        def is_processed(self, key):
            return key in self.processed

        def ensure_safe_available(self, product):
            return "FAKE_PRODUCT.SAFE"

        def mark_processed(self, key):
            self.processed[key] = True
            self.state.parent.mkdir(parents=True, exist_ok=True)
            self.state.write_text(json.dumps(self.processed), encoding="utf-8")

        def cleanup_if_needed(self):
            pass

    calls = []

    class FakeEngine:
        def __init__(self, config, sentinel_loader, mask_loader):
            pass

        def save_dataset(self, **kwargs):
            cap = kwargs["max_new_patches"]
            calls.append(cap)
            existing = sorted((output / "images").glob("sample_*.npy"))
            index = len(existing)
            for directory, suffix in (("images", ".npy"), ("masks", ".npy"),
                                      ("images_tif", ".tif"), ("masks_tif", ".tif")):
                d = output / directory
                d.mkdir(exist_ok=True)
                (d / f"sample_{index:06d}{suffix}").write_bytes(b"synthetic patch data")
            with (output / "patches_log.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id"])
                if index == 0:
                    writer.writeheader()
                writer.writerow({"sample_id": f"sample_{index:06d}"})
            if product_raises:
                raise RuntimeError("synthetic crash after writing patch")
            return 1

    monkeypatch.setattr(pipeline, "load_generation_config", lambda *a, **k: config)
    monkeypatch.setattr(pipeline, "EuroCropsMaskLoader", FakeMaskLoader)
    monkeypatch.setattr(pipeline, "GridCoverageSelector", FakeSelector)
    monkeypatch.setattr(pipeline, "SentinelCacheManager", FakeCache)
    monkeypatch.setattr(pipeline, "STACSentinelCatalogSearcher", FakeCatalog)
    monkeypatch.setattr(pipeline, "SentinelSafeDataLoader", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "DataEngine", FakeEngine)
    monkeypatch.setattr(pipeline, "tile_intersects_eurocrops", lambda *args: True)
    return config, output, calls


def run_strict(tmp_path):
    return pipeline.generate_dataset(
        config_dir=tmp_path, run_tag="teststrict",
        config_mode="isolated", resume_policy="strict", sleep_seconds=0,
    )


def test_strict_pipeline_checks_cumulative_target_and_skips_old_products(tmp_path, monkeypatch):
    config, output, calls = setup_fake_pipeline(tmp_path, monkeypatch)
    first = run_strict(tmp_path)
    assert first.total_samples == 2
    assert calls == [2, 1]
    ledger = json.loads((output / LEDGER_NAME).read_text())
    assert ledger["snapshot"]["sample_count"] == 2
    assert ledger["status"] == "clean"
    resumed = run_strict(tmp_path)
    assert resumed.total_samples == 0
    assert resumed.processed_products == 0
    assert calls == [2, 1]
    assert len(list((output / "images").glob("*.npy"))) == 2


def test_strict_pipeline_refuses_recipe_change_before_loading_mask(tmp_path, monkeypatch):
    config, output, calls = setup_fake_pipeline(tmp_path, monkeypatch)
    run_strict(tmp_path)
    config["dataset"]["min_foreground_ratio"] = 0.75
    with pytest.raises(RuntimeError, match="identity changed"):
        run_strict(tmp_path)
    assert calls == [2, 1]


def test_strict_pipeline_keeps_failed_partial_product_unresumable(tmp_path, monkeypatch):
    config, output, calls = setup_fake_pipeline(tmp_path, monkeypatch, product_raises=True)
    with pytest.raises(RuntimeError, match="Strict generation aborted"):
        run_strict(tmp_path)
    assert json.loads((output / LEDGER_NAME).read_text())["status"] == "running"
    with pytest.raises(RuntimeError, match="Incomplete strict generation"):
        run_strict(tmp_path)
    assert calls == [2]
