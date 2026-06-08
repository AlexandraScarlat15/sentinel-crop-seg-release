from typing import Callable
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize


class RoiMaskCache:
    def __init__(self, mask: np.ndarray):
        self.mask = mask

    @staticmethod
    def build_from_image(
        image_cache,
        gdf_roi_sentinel: gpd.GeoDataFrame,
        class_id_fn: Callable[[str], int],
        crop_column: str,
    ) -> "RoiMaskCache":

        _, H, W = image_cache.image.shape
        transform = image_cache.transform

        # Use the exact world bbox that was actually cached (after clamping)
        minx, miny, maxx, maxy = image_cache.world_bbox
        roi_box = box(minx, miny, maxx, maxy)

        print(f"[MASK] Effective cached ROI world box: {roi_box}")

        gdf = gdf_roi_sentinel[gdf_roi_sentinel.geometry.intersects(roi_box)].copy()

        if gdf.empty:
            return RoiMaskCache(np.zeros((H, W), dtype=np.uint8))

        gdf["class_id"] = gdf[crop_column].apply(class_id_fn)
        gdf = gdf[gdf["class_id"] > 0]

        if gdf.empty:
            return RoiMaskCache(np.zeros((H, W), dtype=np.uint8))

        shapes = zip(gdf.geometry, gdf["class_id"])

        mask = rasterize(
            shapes=shapes,
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )

        return RoiMaskCache(mask)

    def get_patch(self, window):
        col, row, w, h = window
        return self.mask[row:row + h, col:col + w]
