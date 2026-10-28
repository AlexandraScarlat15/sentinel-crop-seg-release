"""Synthetic ROI scoring regressions; no EuroCrops or Sentinel products required."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from sentinel_crop_seg.core.selection.roi_area import covered_area, require_projected_metre_crs
from sentinel_crop_seg.core.selection.roi_selectors.base import RoiSelectionContext
from sentinel_crop_seg.core.selection.roi_selectors.grid_coverage_selector import (
    GridCoverageSelector, _compute_roi_stats,
)
from sentinel_crop_seg.core.selection.roi_selectors.hotcell_selector import HotcellSelector


def polygons(crs="EPSG:32633"):
    # Two 100m square polygons fully overlap. Their sum is 200% of the ROI,
    # whereas the geometric union covers exactly 100%.
    base = gpd.GeoDataFrame(
        {"class_id": [8, 8], "crop": ["soybean", "soybean"]},
        geometry=[box(500000, 4600000, 500100, 4600100)] * 2,
        crs="EPSG:32633",
    )
    # The US foot-based CRS is intentionally a unit-test fixture, not an
    # attempt to transform Central European geometry into New York.
    if crs == "EPSG:2263":
        return base.set_crs(crs, allow_override=True)
    return base.to_crs(crs)


ROI = (500000, 4600000, 500100, 4600100)


def grid_ctx(area_mode):
    return RoiSelectionContext(gdf=polygons(), config={
        "dataset": {"roi": {
            "strategy": "grid", "size_km": 0.1,
            "min_polygon_count": 1, "min_total_area_m2": 0,
            "min_coverage": 0, "area_mode": area_mode,
        }}
    })


def hotcell_ctx(area_mode, *, min_area=0):
    return RoiSelectionContext(gdf=polygons(), config={
        "dataset": {
            "target": {"class_id": 8},
            "roi": {"strategy": "hotcell", "size_km": 0.1,
                    "hotcell": {"cell_km": 0.1, "top_k_cells": 4,
                                "rois_from_cells": 4,
                                "min_target_area_m2": min_area,
                                "area_mode": area_mode}},
        }}
    )


def test_grid_legacy_sum_retains_existing_double_count_semantics():
    result = _compute_roi_stats(polygons(), [ROI])
    assert result[0].polygon_count == 2
    assert result[0].total_crop_area_m2 == pytest.approx(20_000)
    assert result[0].coverage == pytest.approx(2.0)


def test_grid_union_mode_counts_overlaps_once_and_bounds_coverage():
    result = _compute_roi_stats(polygons(), [ROI], area_mode="union")
    assert result[0].polygon_count == 2
    assert result[0].total_crop_area_m2 == pytest.approx(10_000)
    assert result[0].coverage == pytest.approx(1.0)


def test_grid_union_mode_respects_true_min_coverage_threshold():
    selected = GridCoverageSelector().select(grid_ctx("union"))
    assert selected.rois
    assert selected.meta["area_mode"] == "union"
    assert selected.meta["selected"] == len(selected.rois)


def test_grid_union_requires_metric_projection():
    gdf = polygons("EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        _compute_roi_stats(gdf, [(13.0, 47.0, 13.1, 47.1)], area_mode="union")


@pytest.mark.parametrize("crs", ["EPSG:2263", "EPSG:4326"])
def test_grid_union_rejects_nonmetric_or_geographic_crs(crs):
    gdf = polygons(crs)
    with pytest.raises(ValueError, match="metre|projected"):
        _compute_roi_stats(gdf, [tuple(gdf.total_bounds)], area_mode="union")


def test_grid_legacy_still_allows_historic_geographic_fallback():
    gdf = polygons("EPSG:4326")
    result = GridCoverageSelector().select(RoiSelectionContext(gdf=gdf, config={
        "dataset": {"roi": {"size_km": 30, "strategy": "grid",
                            "min_polygon_count": 1, "area_mode": "legacy_sum"}}
    }))
    assert result.meta["area_mode"] == "legacy_sum"


def test_hotcell_legacy_scoring_preserves_polygon_area_sum():
    result = HotcellSelector().select(hotcell_ctx("legacy_sum", min_area=15000))
    assert result.rois
    assert result.meta["area_mode"] == "legacy_sum"


def test_hotcell_union_rejects_duplicated_area_above_threshold():
    result = HotcellSelector().select(hotcell_ctx("union", min_area=15000))
    assert not result.rois
    assert result.meta["reason"] == "no_cells_passed_threshold"


def test_hotcell_union_selects_valid_area_and_reports_mode():
    result = HotcellSelector().select(hotcell_ctx("union", min_area=9000))
    assert result.rois
    assert result.meta["area_mode"] == "union"


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:2263"])
def test_hotcell_rejects_nonmetric_crs_before_allocating_large_grid(crs):
    ctx = hotcell_ctx("union")
    foreign_ctx = RoiSelectionContext(gdf=polygons(crs), config=ctx.config)
    with pytest.raises(ValueError, match="metre|projected"):
        HotcellSelector().select(foreign_ctx)


@pytest.mark.parametrize("mode", ["ignore_overlap", "", None])
def test_roi_rejects_unknown_area_mode(mode):
    with pytest.raises(ValueError, match="area_mode"):
        _compute_roi_stats(polygons(), [ROI], area_mode=mode)
