from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import geopandas as gpd
from shapely.geometry import box

from sentinel_crop_seg.core.selection.roi_area import (
    covered_area, require_projected_metre_crs, validate_roi_area_mode,
)
from sentinel_crop_seg.core.selection.roi_selectors.base import (
    RoiSelector,
    RoiSelectionContext,
    RoiSelectionResult,
)

Roi = Tuple[float, float, float, float]


@dataclass(frozen=True)
class HotCell:
    cell_bbox: Roi
    target_area_m2: float
    score: float


def _ensure_projected_meters(gdf: gpd.GeoDataFrame) -> None:
    """Reject projected-but-nonmetric CRSs: cell_km and m2 require metres."""
    require_projected_metre_crs(gdf, strategy="HotcellSelector")


def _roi_box_from_center(cx: float, cy: float, half_size_m: float) -> Roi:
    return (cx - half_size_m, cy - half_size_m, cx + half_size_m, cy + half_size_m)


def _unique_rois(rois: List[Roi], decimals: int = 1) -> List[Roi]:
    seen = set()
    out: List[Roi] = []
    for r in rois:
        key = tuple(round(float(x), decimals) for x in r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


class HotcellSelector(RoiSelector):
    """Select high-coverage regions around cells containing a target crop.

    Candidate grid cells are ranked by clipped target-polygon area; top cells
    become centered output ROIs with duplicate bounds removed.
    """

    def select(self, ctx: RoiSelectionContext) -> RoiSelectionResult:
        """Rank target-class grid cells and return bounded ROI candidates.

        Args:
            ctx: Spatial polygon frame and ROI/target configuration.

        Returns:
            Selected ROI bounding boxes and selection provenance metadata.

        Raises:
            ValueError: If the CRS, class mapping or target configuration
                is incompatible with hotcell sampling.
        """
        gdf = ctx.gdf
        config = ctx.config

        _ensure_projected_meters(gdf)

        dataset_cfg: Dict[str, Any] = config.get("dataset", {}) or {}
        roi_cfg: Dict[str, Any] = dataset_cfg.get("roi", {}) or {}
        hotcell_cfg: Dict[str, Any] = roi_cfg.get("hotcell", {}) or {}

        target_cfg = dataset_cfg.get("target", {}) or {}
        target_class_id = target_cfg.get("class_id", None)
        if target_class_id is None:
            raise ValueError("HotcellSelector requires dataset.target.class_id to be set (not null).")
        target_class_id = int(target_class_id)

        roi_size_km = float(roi_cfg.get("size_km", 30.0))
        roi_size_m = roi_size_km * 1000.0
        roi_half_m = roi_size_m / 2.0

        cell_km = float(hotcell_cfg.get("cell_km", 5.0))
        cell_size_m = cell_km * 1000.0

        top_k_cells = int(hotcell_cfg.get("top_k_cells", 2000))
        rois_from_cells = int(hotcell_cfg.get("rois_from_cells", 300))
        min_target_area_m2 = float(hotcell_cfg.get("min_target_area_m2", 0.0))
        class_id_col = str(hotcell_cfg.get("class_id_column", "class_id"))
        area_mode = validate_roi_area_mode(hotcell_cfg.get("area_mode", "legacy_sum"))

        if class_id_col not in gdf.columns:
            raise ValueError(
                f"HotcellSelector expects column '{class_id_col}' in gdf. "
                "You must add it before selection (recommended)."
            )

        gdf_t = gdf[gdf[class_id_col].astype(int) == target_class_id].copy()
        if gdf_t.empty:
            return RoiSelectionResult(
                rois=[],
                strategy="hotcell",
                meta={"reason": "no_target_polygons", "target_class_id": target_class_id},
            )

        print("[ROI] Strategy=hotcell")
        print(
            f"[ROI] target_class_id={target_class_id}, roi_size_km={roi_size_km}, "
            f"cell_km={cell_km}, top_k_cells={top_k_cells}, rois_from_cells={rois_from_cells}, "
            f"min_target_area_m2={min_target_area_m2}"
        )

        minx, miny, maxx, maxy = gdf_t.total_bounds

        gx0 = np.floor(minx / cell_size_m) * cell_size_m
        gy0 = np.floor(miny / cell_size_m) * cell_size_m
        gx1 = np.ceil(maxx / cell_size_m) * cell_size_m
        gy1 = np.ceil(maxy / cell_size_m) * cell_size_m

        xs = np.arange(gx0, gx1, cell_size_m, dtype=float)
        ys = np.arange(gy0, gy1, cell_size_m, dtype=float)

        sindex = gdf_t.sindex
        cells: List[HotCell] = []

        for x in xs:
            for y in ys:
                cell_geom = box(x, y, x + cell_size_m, y + cell_size_m)
                cand_idx = list(sindex.intersection(cell_geom.bounds))
                if not cand_idx:
                    continue
                cand = gdf_t.iloc[cand_idx]
                inter = cand[cand.intersects(cell_geom)]
                if inter.empty:
                    continue

                clipped = inter.geometry.intersection(cell_geom)
                area_sum = covered_area(clipped, mode=area_mode)
                if area_sum < min_target_area_m2:
                    continue

                cells.append(
                    HotCell(
                        cell_bbox=(x, y, x + cell_size_m, y + cell_size_m),
                        target_area_m2=area_sum,
                        score=area_sum,
                    )
                )

        if not cells:
            return RoiSelectionResult(
                rois=[],
                strategy="hotcell",
                meta={
                    "reason": "no_cells_passed_threshold",
                    "target_class_id": target_class_id,
                    "min_target_area_m2": min_target_area_m2,
                },
            )

        cells.sort(key=lambda c: c.score, reverse=True)
        cells = cells[:top_k_cells]

        rois: List[Roi] = []
        for c in cells[:rois_from_cells]:
            x0, y0, x1, y1 = c.cell_bbox
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            rois.append(_roi_box_from_center(cx, cy, roi_half_m))

        rois = _unique_rois(rois, decimals=1)

        meta: Dict[str, Any] = {
            "target_class_id": target_class_id,
            "roi_size_km": roi_size_km,
            "cell_km": cell_km,
            "top_k_cells": top_k_cells,
            "rois_from_cells": rois_from_cells,
            "min_target_area_m2": min_target_area_m2,
            "target_polygons": int(len(gdf_t)),
            "cells_scored": int(len(cells)),
            "rois_returned": int(len(rois)),
            "class_id_column": class_id_col,
            "area_mode": area_mode,
        }

        return RoiSelectionResult(rois=rois, strategy="hotcell", meta=meta)
