from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from sentinel_crop_seg.core.selection.roi_grid_builder import RoiGridBuilder, RoiGridConfig
from sentinel_crop_seg.core.selection.roi_area import (
    covered_area, require_projected_metre_crs, validate_roi_area_mode,
)
from sentinel_crop_seg.core.selection.roi_selectors.base import Roi, RoiSelector, RoiSelectionContext, RoiSelectionResult


@dataclass(frozen=True)
class RoiStats:
    roi: Roi
    polygon_count: int
    total_crop_area_m2: float
    roi_area_m2: float
    coverage: float
    score: float


def _compute_roi_stats(
    gdf: gpd.GeoDataFrame,
    rois: List[Roi],
    *,
    clip_area: bool = True,
    score_mode: str = "coverage",  # "coverage" or "coverage_logcount"
    area_mode: str = "legacy_sum",
) -> List[RoiStats]:
    validate_roi_area_mode(area_mode)
    if area_mode == "union":
        require_projected_metre_crs(gdf, strategy="GridCoverageSelector")
    if gdf.empty or not rois:
        return []

    if not gdf.crs or not gdf.crs.is_projected:
        print("[ROI] WARNING: CRS is not projected; area metrics may be meaningless.")

    sindex = gdf.sindex
    out: List[RoiStats] = []

    for roi in rois:
        roi_geom = box(*roi)
        roi_area = float(roi_geom.area)

        cand_idx = list(sindex.intersection(roi_geom.bounds))
        if not cand_idx:
            out.append(RoiStats(roi, 0, 0.0, roi_area, 0.0, 0.0))
            continue

        cand = gdf.iloc[cand_idx]
        intersects = cand[cand.intersects(roi_geom)]
        poly_count = int(len(intersects))

        if poly_count == 0:
            out.append(RoiStats(roi, 0, 0.0, roi_area, 0.0, 0.0))
            continue

        if clip_area:
            clipped = intersects.geometry.intersection(roi_geom)
            area_sum = covered_area(clipped, mode=area_mode)
        else:
            area_sum = covered_area(intersects.geometry, mode=area_mode)

        coverage = (area_sum / roi_area) if roi_area > 0 else 0.0
        if area_mode == "union":
            # Small floating-point discrepancies should not exceed full cover.
            coverage = min(1.0, max(0.0, coverage))

        if score_mode == "coverage_logcount":
            score = float(coverage * np.log1p(poly_count))
        else:
            score = float(coverage)

        out.append(RoiStats(roi, poly_count, area_sum, roi_area, coverage, score))

    return out


def _rank_and_filter(
    stats: List[RoiStats],
    *,
    min_polygon_count: int,
    min_total_area_m2: float,
    min_coverage: Optional[float],
    top_n_rois: Optional[int],
) -> List[RoiStats]:
    filtered = [
        s for s in stats
        if s.polygon_count >= int(min_polygon_count)
        and s.total_crop_area_m2 >= float(min_total_area_m2)
    ]

    if min_coverage is not None:
        filtered = [s for s in filtered if s.coverage >= float(min_coverage)]

    filtered.sort(key=lambda s: s.score, reverse=True)

    if top_n_rois is not None:
        filtered = filtered[: int(top_n_rois)]

    return filtered


class GridCoverageSelector(RoiSelector):
    """
    Strategy A:
    - Build ROI candidates with RoiGridBuilder (grid)
    - Compute density stats (coverage)
    - Filter + sort densest first
    """

    def select(self, ctx: RoiSelectionContext) -> RoiSelectionResult:
        gdf = ctx.gdf
        config = ctx.config

        roi_cfg = config["dataset"]["roi"]
        roi_size_km = float(roi_cfg["size_km"])

        min_polygon_count = int(roi_cfg.get("min_polygon_count", 1))
        min_total_area_m2 = float(roi_cfg.get("min_total_area_m2", 0.0))

        min_coverage = roi_cfg.get("min_coverage", None)
        min_coverage = float(min_coverage) if min_coverage is not None else None

        top_n_rois = roi_cfg.get("top_n_rois", None)
        top_n_rois = int(top_n_rois) if top_n_rois is not None else None

        score_mode = roi_cfg.get("score_mode", "coverage")
        area_mode = validate_roi_area_mode(roi_cfg.get("area_mode", "legacy_sum"))
        if area_mode == "union":
            require_projected_metre_crs(gdf, strategy="GridCoverageSelector")

        # Convert ROI size to CRS units
        if gdf.crs and gdf.crs.is_projected:
            roi_step = roi_size_km * 1000.0
            unit = "meters"
        else:
            roi_step = roi_size_km / 111.0
            unit = "degrees"

        print(f"[ROI] Strategy=grid_coverage")
        print(f"[ROI] ROI size: {roi_size_km} km ({roi_step:.2f} {unit})")
        print(
            f"[ROI] Filters: min_polygon_count={min_polygon_count}, "
            f"min_total_area_m2={min_total_area_m2}, min_coverage={min_coverage}, "
            f"top_n_rois={top_n_rois}, score_mode={score_mode}"
        )

        builder = RoiGridBuilder(
            RoiGridConfig(
                roi_size=roi_step,
                min_polygon_count=min_polygon_count,
            )
        )
        rois = builder.build_rois(gdf)
        print(f"[ROI] Grid candidates: {len(rois)}")

        if not rois:
            return RoiSelectionResult(rois=[], strategy="grid_coverage", meta={"reason": "no_rois_from_grid"})

        stats = _compute_roi_stats(
            gdf, rois, clip_area=True, score_mode=score_mode, area_mode=area_mode,
        )

        coverages = np.array([s.coverage for s in stats], dtype=float)
        areas = np.array([s.total_crop_area_m2 for s in stats], dtype=float)
        counts = np.array([s.polygon_count for s in stats], dtype=float)

        print(
            f"[ROI] coverage min={coverages.min():.4f}, p50={np.percentile(coverages, 50):.4f}, "
            f"p90={np.percentile(coverages, 90):.4f}, max={coverages.max():.4f}"
        )
        print(
            f"[ROI] total_area_m2 min={areas.min():.0f}, p50={np.percentile(areas, 50):.0f}, "
            f"p90={np.percentile(areas, 90):.0f}, max={areas.max():.0f}"
        )
        print(
            f"[ROI] poly_count min={counts.min():.0f}, p50={np.percentile(counts, 50):.0f}, "
            f"p90={np.percentile(counts, 90):.0f}, max={counts.max():.0f}"
        )

        selected = _rank_and_filter(
            stats,
            min_polygon_count=min_polygon_count,
            min_total_area_m2=min_total_area_m2,
            min_coverage=min_coverage,
            top_n_rois=top_n_rois,
        )

        rois_out = [s.roi for s in selected]
        print(f"[ROI] Selected ROIs: {len(rois_out)}")

        meta: Dict[str, Any] = {
            "strategy": "grid_coverage",
            "roi_size_km": roi_size_km,
            "roi_step": roi_step,
            "min_polygon_count": min_polygon_count,
            "min_total_area_m2": min_total_area_m2,
            "min_coverage": min_coverage,
            "top_n_rois": top_n_rois,
            "score_mode": score_mode,
            "area_mode": area_mode,
            "grid_candidates": len(rois),
            "selected": len(rois_out),
        }

        return RoiSelectionResult(rois=rois_out, strategy="grid_coverage", meta=meta)
