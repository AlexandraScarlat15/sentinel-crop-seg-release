"""Pipeline-level quota tests; all STAC, SAFE and EuroCrops IO is mocked."""
from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

from sentinel_crop_seg.core.pipeline import dataset_generation as pipeline
from sentinel_crop_seg.core.selection.roi_selectors.base import RoiSelectionResult


def exercise_pipeline(tmp_path, monkeypatch, target_total):
    seen_limits = []
    completed = []

    class FakeMaskLoader:
        def __init__(self, config):
            self.crop_column = "crop"
            self.gdf = gpd.GeoDataFrame(
                {"crop": ["maize"]},
                geometry=[box(500000, 4599900, 500100, 4600000)],
                crs="EPSG:32633",
            )

        def map_crop_to_class_id(self, crop):
            return 1 if crop == "maize" else 0

    class FakeSelector:
        def select(self, ctx):
            rois = [
                (500000, 4599900, 500100, 4600000),
                (500100, 4599900, 500200, 4600000),
                (500200, 4599900, 500300, 4600000),
            ]
            return RoiSelectionResult(rois=rois, strategy="grid_coverage")

    class FakeCache:
        def __init__(self, config, processed_state_path):
            pass

        def is_processed(self, key):
            return False

        def ensure_safe_available(self, product):
            return product.product_id + ".SAFE"

        def mark_processed(self, key):
            completed.append(key)

        def cleanup_if_needed(self):
            pass

    class FakeCatalog:
        def search(self, **kwargs):
            return [SimpleNamespace(product_id="S2B_MSIL2A_20210810T095029_N0500_R079_T33TWN_20230211T233939")]

    class FakeEngine:
        def __init__(self, config, sentinel_loader, mask_loader):
            pass

        def save_dataset(self, **kwargs):
            remaining = kwargs["max_new_patches"]
            seen_limits.append(remaining)
            # This fake has three acceptable samples per ROI/product.
            return 3 if remaining is None else min(3, remaining)

    config = {
        "generation": {"satellite": "sentinel2", "label_source": "eurocrops"},
        "paths": {
            "sentinel_data_root": str(tmp_path / "unused_sentinel"),
            "dataset_output_dir": str(tmp_path / "unused_dataset"),
        },
        "dataset": {
            "roi": {"strategy": "grid"},
            "target": {"class_id": None, "total_patches": target_total},
            "bands": {"B02": {"resolution": 10}},
        },
        "catalog": {"stac": {
            "start_date": "2021-04-01", "end_date": "2021-09-30",
            "max_cloud_cover": 10, "limit": 5,
        }},
    }
    monkeypatch.setattr(pipeline, "load_generation_config", lambda *a, **k: config)
    monkeypatch.setattr(pipeline, "EuroCropsMaskLoader", FakeMaskLoader)
    monkeypatch.setattr(pipeline, "GridCoverageSelector", FakeSelector)
    monkeypatch.setattr(pipeline, "SentinelCacheManager", FakeCache)
    monkeypatch.setattr(pipeline, "STACSentinelCatalogSearcher", FakeCatalog)
    monkeypatch.setattr(pipeline, "SentinelSafeDataLoader", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "DataEngine", FakeEngine)
    monkeypatch.setattr(pipeline, "tile_intersects_eurocrops", lambda *args: True)

    result = pipeline.generate_dataset(
        config_dir=tmp_path, run_tag="quota_smoke",
        config_mode="isolated", sleep_seconds=0,
    )
    return result, seen_limits, completed


@pytest.mark.parametrize(
    ("target", "expected_total", "expected_limits"),
    [
        (5, 5, [5, 2]),
        (2, 2, [2]),
        (6, 6, [6, 3]),
        (0, 9, [None, None, None]),
    ],
)
def test_pipeline_passes_exact_remaining_quota(tmp_path, monkeypatch, target,
                                              expected_total, expected_limits):
    result, limits, completed = exercise_pipeline(tmp_path, monkeypatch, target)
    assert result.total_samples == expected_total
    assert result.processed_products == len(expected_limits)
    assert limits == expected_limits
    assert len(completed) == len(expected_limits)
