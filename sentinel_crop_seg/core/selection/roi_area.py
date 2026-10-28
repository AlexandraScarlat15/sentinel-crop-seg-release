"""Geometric ROI area rules with explicit scientific/legacy semantics.

Legacy workflows sum clipped polygon areas, including overlapping parcels.
The optional 'union' mode measures actual footprint coverage and requires a
projected CRS with metre units. Coordinates are never silently reprojected.
"""
from __future__ import annotations

import math

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

ROI_AREA_MODES = ("legacy_sum", "union")


def validate_roi_area_mode(mode: str) -> str:
    if mode not in ROI_AREA_MODES:
        raise ValueError(f"ROI area_mode must be one of {ROI_AREA_MODES}, got {mode!r}")
    return mode


def require_projected_metre_crs(gdf: gpd.GeoDataFrame, *, strategy: str) -> None:
    crs = gdf.crs
    if crs is None or not crs.is_projected:
        raise ValueError(
            f"{strategy} requires a projected CRS with metre units; "
            "reproject EuroCrops before ROI selection."
        )
    axes = crs.axis_info[:2]
    if len(axes) != 2 or any(
        not math.isclose(float(axis.unit_conversion_factor), 1.0, rel_tol=0, abs_tol=1e-8)
        for axis in axes
    ):
        raise ValueError(
            f"{strategy} requires metre-based projected coordinates "
            f"(received CRS {crs}); reproject EuroCrops first."
        )


def covered_area(
    clipped_geometries: gpd.GeoSeries,
    *,
    mode: str,
) -> float:
    """Clipped area in native CRS units; union mode avoids double counting."""
    validate_roi_area_mode(mode)
    nonempty: list[BaseGeometry] = [
        geometry for geometry in clipped_geometries
        if geometry is not None and not geometry.is_empty
    ]
    if not nonempty:
        return 0.0
    if mode == "legacy_sum":
        return float(sum(geometry.area for geometry in nonempty))
    return float(unary_union(nonempty).area)
