"""Calendar-selection tests are offline and use synthetic STAC metadata."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

from sentinel_crop_seg.core.pipeline.acquisition_calendar import AcquisitionCalendar
from sentinel_crop_seg.core.pipeline.generation_config import load_generation_config, generation_config_report
from sentinel_crop_seg.core.pipeline import dataset_generation as pipeline
from sentinel_crop_seg.core.selection.roi_selectors.base import RoiSelectionResult


def configured(tmp_path: Path, *, crops=("Wheat",), year=2021, country="AT"):
    csv_file = tmp_path / "calendar.csv"
    csv_file.write_text(
        "country,crop,start,end\n"
        "AT,Wheat,04-01,08-15\n"
        "AT,Maize,05-10,09-30\n"
        "FR,Wheat,04-01,07-31\n", encoding="utf-8"
    )
    config = {"catalog": {"calendar": {
        "mode": "selected_crop_union", "file": str(csv_file),
        "country": country, "year": year, "crops": list(crops),
    }}}
    return config, csv_file


def test_calendar_boundaries_and_union(tmp_path):
    config, _ = configured(tmp_path)
    cal = AcquisitionCalendar.from_config(config)
    assert cal.accepts("2021-04-01")
    assert cal.accepts("2021-08-15")
    assert not cal.accepts("2021-08-16")
    assert not cal.accepts("2020-07-01")
    assert cal.search_intervals("2021-04-01T00:00:00Z", "2021-09-30T23:59:59Z") == [
        ("2021-04-01T00:00:00Z", "2021-08-15T23:59:59.999999Z")
    ]
    config["catalog"]["calendar"]["crops"] = ["Wheat", "Maize"]
    combo = AcquisitionCalendar.from_config(config)
    assert combo.accepts("2021-09-20")
    assert len(combo.search_intervals("2021-01-01", "2021-12-31")) == 1


def test_calendar_clipping_and_no_overlap(tmp_path):
    config, _ = configured(tmp_path)
    cal = AcquisitionCalendar.from_config(config)
    assert cal.search_intervals("2021-06-01", "2021-06-15") == [
        ("2021-06-01T00:00:00Z", "2021-06-15T23:59:59.999999Z")
    ]
    with pytest.raises(ValueError, match="no overlap"):
        cal.search_intervals("2021-10-01", "2021-12-31")
    with pytest.raises(ValueError, match="UTC offset"):
        cal.search_intervals("2021-04-01T00:00:00", "2021-09-30T23:59:59Z")


@pytest.mark.parametrize("change,error", [
    ({"country": "at"}, "ISO-2"),
    ({"year": "2021"}, "integer"),
    ({"crops": []}, "nonempty"),
    ({"crops": ["Wheat", "wheat"]}, "Duplicate"),
    ({"crops": ["Potato"]}, "Missing calendar windows"),
    ({"mode": "anything"}, "selected_crop_union"),
])
def test_invalid_calendar_settings_fail_closed(tmp_path, change, error):
    config, _ = configured(tmp_path)
    config["catalog"]["calendar"].update(change)
    with pytest.raises(ValueError, match=error):
        AcquisitionCalendar.from_config(config)


@pytest.mark.parametrize("body,error", [
    ("country,crop,start,end\nAT,Wheat,04-31,08-15\n", "Invalid calendar"),
    ("country,crop,start,end\nAT,Wheat,08-15,04-01\n", "crosses the year"),
    ("country,crop,start,end\nAT,Wheat,04-01,08-15\nAT,Wheat,04-01,08-15\n", "Duplicate"),
    ("country,crop,start\nAT,Wheat,04-01\n", "exactly"),
])
def test_invalid_calendar_table_rejected(tmp_path, body, error):
    config, path = configured(tmp_path)
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        AcquisitionCalendar.from_config(config)


def test_calendar_source_edits_change_sha(tmp_path):
    config, source = configured(tmp_path)
    first = AcquisitionCalendar.from_config(config)
    source.write_text(source.read_text().replace("08-15", "08-10"), encoding="utf-8")
    second = AcquisitionCalendar.from_config(config)
    assert first.csv_sha256 != second.csv_sha256
    assert first.provenance()["windows"] != second.provenance()["windows"]


def test_real_2018_2021_example_recipes_preflight_offline():
    examples = [
        ("config/experiments/examples/at_2021_calendar.yaml", "AT", 2021),
        ("config/experiments/examples/fr_2018_calendar.yaml", "FR", 2018),
    ]
    for recipe, country, year in examples:
        report = generation_config_report("config", experiment_recipe=recipe)
        meta = report["acquisition_calendar"]
        assert meta["country"] == country and meta["year"] == year
        assert meta["crops"] == ["Wheat"]
        assert len(meta["calendar_sha256"]) == 64
        assert meta["semantics"].endswith("not_parcel_filter")


def test_calendar_is_enforced_before_downloading(tmp_path, monkeypatch):
    config, _ = configured(tmp_path)
    config.update({
        "generation": {"satellite": "sentinel2", "label_source": "eurocrops"},
        "paths": {"sentinel_data_root": str(tmp_path / "cache"),
                  "dataset_output_dir": str(tmp_path / "output")},
        "dataset": {"roi": {"strategy": "grid"},
                    "target": {"class_id": None, "total_patches": 0},
                    "bands": {"B02": {"resolution": 10}}},
    })
    config["catalog"]["stac"] = {
        "start_date": "2021-04-01T00:00:00Z",
        "end_date": "2021-09-30T23:59:59Z",
        "max_cloud_cover": 10, "limit": 50,
    }
    config["acquisition_calendar_provenance"] = AcquisitionCalendar.from_config(config).provenance()
    touched, ranges, saved = [], [], []

    class FakeMask:
        def __init__(self, resolved):
            self.crop_column = "crop"
            self.gdf = gpd.GeoDataFrame(
                {"crop": ["wheat"]},
                geometry=[box(500000, 4599900, 500100, 4600000)],
                crs="EPSG:32633",
            )
        def map_crop_to_class_id(self, crop):
            return 3

    class FakeSelector:
        def select(self, context):
            return RoiSelectionResult(rois=[(500000, 4599900, 500100, 4600000)],
                                      strategy="grid")

    class FakeCatalog:
        def search(self, **kwargs):
            ranges.append((kwargs["start_date"], kwargs["end_date"]))
            return [
                SimpleNamespace(product_id="inside", acquisition_date="2021-06-15"),
                SimpleNamespace(product_id="outside", acquisition_date="2021-09-15"),
            ]

    class FakeCache:
        def __init__(self, config, processed_state_path):
            pass
        def is_processed(self, key):
            return False
        def ensure_safe_available(self, product):
            touched.append(product.product_id)
            return product.product_id
        def mark_processed(self, key):
            pass
        def cleanup_if_needed(self):
            pass

    class FakeEngine:
        def __init__(self, **kwargs):
            pass
        def save_dataset(self, **kwargs):
            saved.append(kwargs["source_product_id"])
            return 1

    monkeypatch.setattr(pipeline, "load_generation_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(pipeline, "EuroCropsMaskLoader", FakeMask)
    monkeypatch.setattr(pipeline, "GridCoverageSelector", FakeSelector)
    monkeypatch.setattr(pipeline, "STACSentinelCatalogSearcher", FakeCatalog)
    monkeypatch.setattr(pipeline, "SentinelCacheManager", FakeCache)
    monkeypatch.setattr(pipeline, "SentinelSafeDataLoader", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "DataEngine", FakeEngine)
    monkeypatch.setattr(pipeline, "tile_intersects_eurocrops", lambda *args: True)

    result = pipeline.generate_dataset(config_mode="isolated", sleep_seconds=0)
    assert result.total_samples == 1
    assert saved == touched == ["inside"]
    assert ranges == [("2021-04-01T00:00:00Z", "2021-08-15T23:59:59.999999Z")]
