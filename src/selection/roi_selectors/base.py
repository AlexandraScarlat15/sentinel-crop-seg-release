from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import geopandas as gpd

Roi = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RoiSelectionContext:
    """
    Inputs that any ROI selection strategy may need.
    - gdf: EuroCrops polygons already filtered to valid classes.
    - config: full config dict.
    """
    gdf: gpd.GeoDataFrame
    config: Dict[str, Any]


@dataclass(frozen=True)
class RoiSelectionResult:
    """
    Unified output from any strategy.
    - rois: list of ROI boxes in gdf.crs (same CRS as gdf).
    - strategy: identifier string.
    - meta: optional debug info (counts, thresholds used, stats).
    """
    rois: List[Roi]
    strategy: str
    meta: Optional[Dict[str, Any]] = None


class RoiSelector:
    """
    Contract: any ROI selection strategy must implement select().
    """
    def select(self, ctx: RoiSelectionContext) -> RoiSelectionResult:
        raise NotImplementedError
