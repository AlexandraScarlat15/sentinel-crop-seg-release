from types import SimpleNamespace

import geopandas as gpd
from shapely.geometry import box

from src.selection.roi_grid_builder import RoiGridBuilder, RoiGridConfig
from src.utils.spatial_filters import tile_intersects_eurocrops


def test_roi_grid_builder_returns_empty_list_for_empty_geodataframe():
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    builder = RoiGridBuilder(
        RoiGridConfig(
            roi_size=1000.0,
            min_polygon_count=1,
        )
    )

    assert builder.build_rois(gdf) == []


def test_roi_grid_builder_keeps_cells_with_enough_polygons():
    gdf = gpd.GeoDataFrame(
        geometry=[
            box(0.1, 0.1, 0.2, 0.2),
            box(0.3, 0.3, 0.4, 0.4),
        ],
        crs="EPSG:3857",
    )

    builder = RoiGridBuilder(
        RoiGridConfig(
            roi_size=1.0,
            min_polygon_count=2,
        )
    )

    rois = builder.build_rois(gdf)

    assert len(rois) == 1

    minx, miny, maxx, maxy = rois[0]
    assert minx <= 0.1
    assert miny <= 0.1
    assert maxx >= 1.1
    assert maxy >= 1.1


def test_tile_intersects_eurocrops_returns_true_for_overlapping_footprint():
    eurocrops_gdf = gpd.GeoDataFrame(
        geometry=[box(0.0, 0.0, 1.0, 1.0)],
        crs="EPSG:4326",
    )

    product = SimpleNamespace(
        footprint=box(0.5, 0.5, 1.5, 1.5).__geo_interface__
    )

    assert tile_intersects_eurocrops(product, eurocrops_gdf) is True


def test_tile_intersects_eurocrops_returns_false_for_non_overlapping_footprint():
    eurocrops_gdf = gpd.GeoDataFrame(
        geometry=[box(0.0, 0.0, 1.0, 1.0)],
        crs="EPSG:4326",
    )

    product = SimpleNamespace(
        footprint=box(2.0, 2.0, 3.0, 3.0).__geo_interface__
    )

    assert tile_intersects_eurocrops(product, eurocrops_gdf) is False
