"""
EuroCrops Mask Loader
=====================

Concrete implementation of MaskDataLoader using EuroCrops vector data.
Checks config 'classes' to map crop names to IDs and generates aligned masks.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import geopandas as gpd
import rasterio.windows
from shapely.geometry import box
from rasterio.features import rasterize

from src.data.mask_loader import MaskDataLoader


class EuroCropsMaskLoader(MaskDataLoader):
    """
    Mask loader implementation based on EuroCrops vector datasets.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        euro_cfg = config.get("eurocrops", {})
        shapefile_path = euro_cfg.get("shapefile_path")

        if shapefile_path is None:
            raise ValueError("EuroCrops shapefile_path not defined in config.")

        self.crop_column = euro_cfg.get("crop_column", "EC_trans_n")

        print(f"[LOADER] Loading EuroCrops shapefile: {shapefile_path}")
        self.gdf = gpd.read_file(shapefile_path)

        if self.gdf.crs is None:
            raise ValueError("EuroCrops shapefile has no CRS.")
        
        if not self.gdf.has_sindex:
            self.gdf.sindex

    def get_class_mapping(self) -> Dict[int, str]:
        """
        Return the project class mapping as {class_id: class_name}.
        """
        classes_cfg = self.config.get("classes", {})

        mapping: Dict[int, str] = {}

        for class_name, class_def in classes_cfg.items():
            class_id = int(class_def.get("id", 0))
            mapping[class_id] = str(class_name)

        return dict(sorted(mapping.items()))

    def map_crop_to_class_id(self, crop_name: str) -> int:
        """
        Map a EuroCrops crop name to a class ID using config rules (case-insensitive substring match).
        """
        if not isinstance(crop_name, str):
            return 0

        crop_name_lower = crop_name.lower()
        classes_cfg = self.config.get("classes", {})

        for class_def in classes_cfg.values():
            class_id = class_def.get("id", 0)
            includes = [s.lower() for s in class_def.get("include", [])]
            excludes = [s.lower() for s in class_def.get("exclude", [])]

            if any(excl in crop_name_lower for excl in excludes):
                continue

            if any(inc in crop_name_lower for inc in includes):
                return class_id

        return 0 

    def generate_mask(
        self,
        window: Tuple[int, int, int, int],
        spatial_metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate a ground truth mask aligned with a Sentinel image window.
        
        CRITICAL: This method ensures pixel-perfect alignment between the vector
        data and the raster window.
        """
        col, row, width, height = window
        sentinel_transform = spatial_metadata["transform"]
        sentinel_crs = spatial_metadata["crs"]

        window_transform = rasterio.windows.transform(window, sentinel_transform)

        x_min, y_max = window_transform * (0, 0)
       
        x_max, y_min = window_transform * (width, height)
        
        window_bbox = box(x_min, y_min, x_max, y_max)

        if self.gdf.crs.to_string() != sentinel_crs:
            # print(f"[MASK] Reprojecting GDF slice to {sentinel_crs}...")
            gdf_projected = self.gdf.to_crs(sentinel_crs)
        else:
            gdf_projected = self.gdf

        try:
             gdf_window = gdf_projected.cx[x_min:x_max, y_min:y_max].copy()
        except Exception as e:
             print(f"[MASK WARNING] Spatial index error, falling back to slow intersection: {e}")
             gdf_window = gdf_projected[gdf_projected.geometry.intersects(window_bbox)].copy()

        if gdf_window.empty:
            return np.zeros((height, width), dtype=np.uint8)

        gdf_window["class_id"] = gdf_window[self.crop_column].apply(self.map_crop_to_class_id)
        
        gdf_window = gdf_window[gdf_window["class_id"] > 0]

        if gdf_window.empty:
            return np.zeros((height, width), dtype=np.uint8)

        shapes = ((geom, cid) for geom, cid in zip(gdf_window.geometry, gdf_window["class_id"]))

        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=window_transform, 
            fill=0,
            dtype="uint8",
            all_touched=False 
        )
        
        return mask
    
    
    def get_regions_of_interest(self) -> List[Tuple[float, float, float, float]]:
        raise NotImplementedError("ROIs are built externally using RoiGridBuilder.")

    def get_gdf_in_crs(self, target_crs: str):
        if self.gdf.crs.to_string() != target_crs:
            return self.gdf.to_crs(target_crs)
        return self.gdf