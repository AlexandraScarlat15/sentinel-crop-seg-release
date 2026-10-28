from typing import Tuple, List, Optional
import numpy as np
import rasterio
from rasterio.windows import Window

Roi = Tuple[float, float, float, float]
PatchWindow = Tuple[int, int, int, int]


class RoiImageCache:
    """
    Holds a cached Sentinel image subset (bands, H, W) and its affine transform.

    Important:
    - The ROI passed into build() must be in the same CRS as spatial_metadata["crs"].
    - The ROI may be partially outside the tile; we clamp to tile bounds.
    - We expose the effective world bbox actually cached (after clamping).
    """

    def __init__(self, image: np.ndarray, transform, world_bbox: Roi, window: Window):
        self.image = image
        self.transform = transform
        self.world_bbox = world_bbox  # (minx, miny, maxx, maxy) in Sentinel CRS
        self.window = window          # rasterio Window in tile pixel coordinates

    @staticmethod
    def build(sentinel_loader, spatial_metadata, roi: Roi, bands: List[str]) -> "RoiImageCache":
        transform = spatial_metadata["transform"]
        inv = ~transform

        minx, miny, maxx, maxy = roi

        # World -> pixel. Use (minx, maxy) and (maxx, miny) for north-up imagery.
        col0, row0 = inv * (minx, maxy)
        col1, row1 = inv * (maxx, miny)

        col_min = int(np.floor(min(col0, col1)))
        row_min = int(np.floor(min(row0, row1)))
        col_max = int(np.ceil(max(col0, col1)))
        row_max = int(np.ceil(max(row0, row1)))

        img_w = int(spatial_metadata["width"])
        img_h = int(spatial_metadata["height"])

        # Clamp to tile bounds
        col_min = max(0, min(col_min, img_w - 1))
        row_min = max(0, min(row_min, img_h - 1))
        col_max = max(1, min(col_max, img_w))
        row_max = max(1, min(row_max, img_h))

        width = col_max - col_min
        height = row_max - row_min

        if width <= 1 or height <= 1:
            raise ValueError(
                f"Degenerate ROI after clamping. roi={roi}, "
                f"pixel_window={(col_min, row_min, width, height)}, "
                f"tile_size={(img_w, img_h)}"
            )

        window = Window(col_min, row_min, width, height)

        image = sentinel_loader.read_window(
            window=(col_min, row_min, width, height),
            bands=bands,
        )

        roi_transform = rasterio.windows.transform(window, transform)

        # Effective world bbox actually cached (after clamping)
        x0, y0 = roi_transform * (0, 0)
        x1, y1 = roi_transform * (width, height)
        world_bbox = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

        return RoiImageCache(
            image=image,
            transform=roi_transform,
            world_bbox=world_bbox,
            window=window,
        )

    def get_patch(self, window: PatchWindow) -> np.ndarray:
        col, row, w, h = window
        return self.image[:, row:row + h, col:col + w]
