from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from shapely.geometry import box
import geopandas as gpd

Roi = Tuple[float, float, float, float]

@dataclass(frozen=True)
class RoiGridConfig:
    roi_size: float          # meters if CRS is projected
    min_polygon_count: int

class RoiGridBuilder:
    def __init__(self, config: RoiGridConfig):
        self.config = config

    def build_rois(self, gdf: gpd.GeoDataFrame) -> List[Roi]:
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
        if gdf.empty:
            return []

        minx, miny, maxx, maxy = gdf.total_bounds
        step = self.config.roi_size

        xs = np.arange(minx, maxx, step)
        ys = np.arange(miny, maxy, step)

        rois = []
        sindex = gdf.sindex

        print(f"[ROI] Building grid: {len(xs)} x-rows")

        for i, x0 in enumerate(xs, start=1):
            if i % 5 == 0 or i == 1:
                print(f"[ROI] Processing row {i}/{len(xs)} (x={x0:.2f})")

            for y0 in ys:
                cell = box(x0, y0, x0 + step, y0 + step)
                idx = list(sindex.intersection(cell.bounds))
                if len(idx) < self.config.min_polygon_count:
                    continue
                rois.append(cell.bounds)

        print(f"[ROI] Finished. Kept {len(rois)} ROIs")
        return rois
