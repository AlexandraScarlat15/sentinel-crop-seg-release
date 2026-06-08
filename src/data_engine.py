from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio.windows import (
    Window,
    bounds as window_bounds,
    transform as window_transform,
)
from shapely.geometry import box

from src.data.roi_image_cache import RoiImageCache
from src.data.roi_mask_cache import RoiMaskCache
from src.data.sentinel_product_metadata import (
    infer_sentinel_scene_date,
    infer_sentinel_tile_id,
)
from src.geospatial.geotiff_writer import (
    save_mask_geotiff,
    save_multiband_geotiff,
)


Roi = Tuple[float, float, float, float]


PATCH_LOG_COLUMNS = [
    "log_schema_version",
    "sample_id",
    "product_id",
    "safe_id",
    "scene_date",
    "tile_id",
    "roi_idx",
    "col",
    "row",
    "global_col",
    "global_row",
    "size",
    "image_npy_path",
    "mask_npy_path",
    "image_tif_path",
    "mask_tif_path",
    "image_formats",
    "mask_formats",
    "crs",
    "transform_a",
    "transform_b",
    "transform_c",
    "transform_d",
    "transform_e",
    "transform_f",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
    "min_lng",
    "min_lat",
    "max_lng",
    "max_lat",
    "band_order",
    "class_pixel_counts",
    "foreground_ratio",
    "nodata_ratio",
    "cloud_ratio",
    "cloud_on_fg_ratio",
    "bright_ratio",
    "bright_on_fg_ratio",
    "target_class_id",
    "target_pixels",
    "target_ratio",
]


PATCH_LOG_SCHEMA_VERSION = "3"


def _dilate_bool_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Simple 2D dilation using only NumPy."""
    if radius <= 0:
        return mask

    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    padded = np.pad(
        mask,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=False,
    )

    out = np.zeros_like(mask, dtype=bool)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= padded[
                radius + dy : radius + dy + mask.shape[0],
                radius + dx : radius + dx + mask.shape[1],
            ]

    return out


def _relative_posix(path: Path, root: Path) -> str:
    """Return a stable POSIX-style relative path for metadata files."""
    return path.relative_to(root).as_posix()


def _class_pixel_counts(mask: np.ndarray) -> str:
    """Return compact JSON with class-id to pixel-count mapping."""
    values, counts = np.unique(mask, return_counts=True)
    payload = {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _validate_existing_patch_log(log_path: Path) -> bool:
    """
    Validate an existing patch log.

    Returns True if the file exists and already has a valid header.
    Returns False if the file does not exist or is empty.
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return False

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)

    if existing_header != PATCH_LOG_COLUMNS:
        raise ValueError(
            "Existing patches_log.csv uses an older or incompatible schema.\n"
            f"Path: {log_path}\n"
            "Please remove this file or write to a fresh output directory before "
            "generating new patches."
        )

    return True


class DataEngine:
    def __init__(self, config, sentinel_loader, mask_loader):
        self.config = config
        self.sentinel = sentinel_loader
        self.mask = mask_loader
        self.sentinel_crs = self.sentinel.get_spatial_metadata()["crs"]

    def _get_next_index(self, *sample_dirs: Path) -> int:
        """
        Find the next sample index across one or more output directories.

        This prevents duplicate sample IDs when both NumPy and GeoTIFF outputs exist.
        """
        max_index = -1

        for sample_dir in sample_dirs:
            if not sample_dir.exists():
                continue

            for pattern in ("sample_*.npy", "sample_*.tif", "sample_*.tiff"):
                for path in sample_dir.glob(pattern):
                    try:
                        current_index = int(path.stem.split("_")[-1])
                    except ValueError:
                        continue

                    max_index = max(max_index, current_index)

        return max_index + 1

    def _reproject_roi_to_sentinel(self, roi_ec: Roi) -> Roi:
        roi_box = box(*roi_ec)

        roi_s2 = (
            gpd.GeoDataFrame(geometry=[roi_box], crs=self.mask.gdf.crs)
            .to_crs(self.sentinel_crs)
            .geometry.iloc[0]
        )

        minx, miny, maxx, maxy = roi_s2.bounds

        if minx >= maxx or miny >= maxy:
            raise ValueError("Degenerate ROI after reprojection")

        return float(minx), float(miny), float(maxx), float(maxy)

    def save_dataset(
        self,
        output_dir: str,
        rois: List[Roi],
        bands: List[str],
        target_class_id: int | None = None,
        min_target_ratio: float = 0.0,
        min_target_pixels: int = 0,
        source_product_id: str | None = None,
        source_safe_id: str | None = None,
        source_scene_date: str | None = None,
        source_tile_id: str | None = None,
    ) -> int:
        """
        Extract image-mask patches and save them to disk.

        Each accepted patch is saved in two formats:

        1. NumPy training cache:
           - images/sample_XXXXXX.npy
           - masks/sample_XXXXXX.npy

        2. Georeferenced GeoTIFF:
           - images_tif/sample_XXXXXX.tif
           - masks_tif/sample_XXXXXX.tif

        The GeoTIFF image and mask share the same CRS, affine transform, width,
        and height. The CSV patch log stores relative paths, CRS, affine transform
        coefficients, WGS84 bounds, source-grid bounds, source Sentinel metadata,
        quality indicators, and class composition.
        """
        output_dir = Path(output_dir)

        img_dir = output_dir / "images"
        mask_dir = output_dir / "masks"

        img_tif_dir = output_dir / "images_tif"
        mask_tif_dir = output_dir / "masks_tif"

        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        img_tif_dir.mkdir(parents=True, exist_ok=True)
        mask_tif_dir.mkdir(parents=True, exist_ok=True)

        patch_size = int(self.config["dataset"]["patch"]["size"])
        stride = int(self.config["dataset"]["patch"]["stride"])
        max_patches_per_roi = int(
            self.config["dataset"].get("max_patches_per_roi", 1000)
        )

        min_foreground_ratio = float(
            self.config["dataset"].get("min_foreground_ratio", 0.30)
        )
        max_nodata_ratio = float(
            self.config["dataset"].get("max_nodata_fraction", 0.05)
        )

        max_cloud_fraction = float(
            self.config["dataset"].get("max_cloud_fraction", 0.01)
        )
        max_cloud_on_fg_fraction = float(
            self.config["dataset"].get("max_cloud_on_foreground_fraction", 0.005)
        )

        default_bad_scl = [0, 1, 2, 3, 7, 8, 9, 10, 11]
        bad_scl_classes = self.config["dataset"].get(
            "bad_scl_classes",
            default_bad_scl,
        )
        bad_scl_classes = [int(x) for x in bad_scl_classes]

        cloud_dilate_radius = int(
            self.config["dataset"].get("cloud_dilate_radius", 2)
        )

        use_b02_brightness_filter = bool(
            self.config["dataset"].get("use_b02_brightness_filter", True)
        )
        b02_bright_threshold = float(
            self.config["dataset"].get("b02_bright_threshold", 2000)
        )
        max_bright_fraction = float(
            self.config["dataset"].get("max_b02_bright_fraction", 0.02)
        )
        max_bright_on_fg_fraction = float(
            self.config["dataset"].get("max_b02_bright_on_foreground_fraction", 0.01)
        )

        if "B02" not in bands:
            use_b02_brightness_filter = False
            b02_index = -1
        else:
            b02_index = bands.index("B02")

        spatial_metadata = self.sentinel.get_spatial_metadata()
        ref_transform = spatial_metadata["transform"]
        ref_px_x = abs(ref_transform.a)

        to_wgs84 = Transformer.from_crs(
            self.sentinel_crs,
            "EPSG:4326",
            always_xy=True,
        )

        log_path = output_dir / "patches_log.csv"
        log_exists = _validate_existing_patch_log(log_path)

        saved_global = self._get_next_index(
            img_dir,
            mask_dir,
            img_tif_dir,
            mask_tif_dir,
        )
        start_index = saved_global

        band_order = ",".join(str(band) for band in bands)

        product_id_for_log = "" if source_product_id is None else str(source_product_id)
        safe_id_for_log = "" if source_safe_id is None else str(source_safe_id)

        scene_date_for_log = (
            str(source_scene_date)
            if source_scene_date is not None
            else infer_sentinel_scene_date(product_id_for_log)
        )

        tile_id_for_log = (
            str(source_tile_id)
            if source_tile_id is not None
            else (
                infer_sentinel_tile_id(product_id_for_log)
                or infer_sentinel_tile_id(safe_id_for_log)
            )
        )

        with log_path.open("a", newline="", encoding="utf-8") as log_f:
            writer = csv.writer(log_f)

            if not log_exists:
                writer.writerow(PATCH_LOG_COLUMNS)

            for roi_idx, roi_ec in enumerate(rois, start=1):
                print(f"\n[ENGINE] ROI {roi_idx}/{len(rois)}")

                roi_box = box(*roi_ec)
                gdf_roi_ec = self.mask.gdf[
                    self.mask.gdf.geometry.intersects(roi_box)
                ].copy()

                if gdf_roi_ec.empty:
                    print("[ENGINE] No EuroCrops polygons")
                    continue

                try:
                    roi_s2 = self._reproject_roi_to_sentinel(roi_ec)
                except Exception as e:
                    print(f"[ENGINE] reprojection failed: {e}")
                    continue

                gdf_roi_s2 = gdf_roi_ec.to_crs(self.sentinel_crs)

                try:
                    roi_img = RoiImageCache.build(
                        sentinel_loader=self.sentinel,
                        spatial_metadata=spatial_metadata,
                        roi=roi_s2,
                        bands=bands,
                    )
                except Exception as e:
                    print(f"[ENGINE] image cache failed: {e}")
                    continue

                _, H, W = roi_img.image.shape

                print(f"[ENGINE] ROI image size: {W}x{H}px")

                if H < patch_size or W < patch_size:
                    print("[ENGINE] ROI image smaller than patch_size -> skipping")
                    continue

                print(f"[ENGINE] Patch size: {patch_size}px (~{patch_size * ref_px_x:.0f}m)")
                print(
                    f"[ENGINE] Thresholds: min_fg={min_foreground_ratio}, "
                    f"max_cloud={max_cloud_fraction}, "
                    f"max_cloud_on_fg={max_cloud_on_fg_fraction}, "
                    f"bad_scl={bad_scl_classes}, "
                    f"dilate={cloud_dilate_radius}, "
                    f"brightness_filter={use_b02_brightness_filter}, "
                    f"target_class_id={target_class_id}, "
                    f"min_target_ratio={min_target_ratio}, "
                    f"min_target_pixels={min_target_pixels}"
                )

                roi_mask = RoiMaskCache.build_from_image(
                    image_cache=roi_img,
                    gdf_roi_sentinel=gdf_roi_s2,
                    class_id_fn=self.mask.map_crop_to_class_id,
                    crop_column=self.mask.crop_column,
                )

                attempted = 0
                roi_saved = 0

                rej_fg = 0
                rej_cloud = 0
                rej_bright = 0
                rej_nodata = 0
                rej_flat = 0
                rej_target = 0

                for row in range(0, H - patch_size + 1, stride):
                    for col in range(0, W - patch_size + 1, stride):
                        attempted += 1

                        patch_img = roi_img.get_patch(
                            (col, row, patch_size, patch_size)
                        )
                        patch_mask = roi_mask.get_patch(
                            (col, row, patch_size, patch_size)
                        )

                        nodata_mask = np.all(patch_img == 0, axis=0)
                        nodata_ratio = float(nodata_mask.mean())

                        if nodata_ratio > max_nodata_ratio:
                            rej_nodata += 1
                            continue

                        foreground = patch_mask > 0
                        foreground_ratio = float(foreground.mean())

                        if foreground_ratio < min_foreground_ratio:
                            rej_fg += 1
                            continue

                        if float(patch_img.max()) == float(patch_img.min()):
                            rej_flat += 1
                            continue

                        bright_ratio = 0.0
                        bright_on_fg_ratio = 0.0

                        if use_b02_brightness_filter and b02_index >= 0:
                            b02 = patch_img[b02_index].astype(np.float32)
                            bright = b02 > b02_bright_threshold

                            bright_ratio = float(bright.mean())

                            fg_count = int(foreground.sum())
                            bright_on_fg_ratio = float(
                                (bright & foreground).sum() / max(1, fg_count)
                            )

                            if (
                                bright_ratio > max_bright_fraction
                                or bright_on_fg_ratio > max_bright_on_fg_fraction
                            ):
                                rej_bright += 1
                                continue

                        scl_patch = self.sentinel.read_scl(
                            window=(
                                int(roi_img.window.col_off + col),
                                int(roi_img.window.row_off + row),
                                int(patch_size),
                                int(patch_size),
                            )
                        )

                        bad = np.isin(scl_patch, bad_scl_classes)
                        bad = _dilate_bool_mask(bad, cloud_dilate_radius)

                        cloud_ratio = float(bad.mean())

                        fg_count = int(foreground.sum())
                        cloud_on_fg_ratio = float(
                            (bad & foreground).sum() / max(1, fg_count)
                        )

                        if (
                            cloud_ratio > max_cloud_fraction
                            or cloud_on_fg_ratio > max_cloud_on_fg_fraction
                        ):
                            rej_cloud += 1
                            continue

                        target_pixels_for_log = ""
                        target_ratio_for_log = ""

                        if target_class_id is not None:
                            target = patch_mask == target_class_id
                            target_pixels = int(target.sum())
                            target_ratio = float(target_pixels / target.size)

                            if (
                                target_ratio < min_target_ratio
                                or target_pixels < min_target_pixels
                            ):
                                rej_target += 1
                                continue

                            target_pixels_for_log = target_pixels
                            target_ratio_for_log = f"{target_ratio:.6f}"

                        global_window = Window(
                            col_off=float(roi_img.window.col_off + col),
                            row_off=float(roi_img.window.row_off + row),
                            width=float(patch_size),
                            height=float(patch_size),
                        )

                        patch_transform = window_transform(global_window, ref_transform)

                        left, bottom, right, top = window_bounds(
                            global_window,
                            ref_transform,
                        )

                        lng_left, lat_bottom = to_wgs84.transform(left, bottom)
                        lng_right, lat_top = to_wgs84.transform(right, top)

                        min_lng = min(lng_left, lng_right)
                        max_lng = max(lng_left, lng_right)
                        min_lat = min(lat_bottom, lat_top)
                        max_lat = max(lat_bottom, lat_top)

                        min_x = min(left, right)
                        max_x = max(left, right)
                        min_y = min(bottom, top)
                        max_y = max(bottom, top)

                        sample_id = f"sample_{saved_global:06d}"

                        image_npy_path = img_dir / f"{sample_id}.npy"
                        mask_npy_path = mask_dir / f"{sample_id}.npy"

                        image_tif_path = img_tif_dir / f"{sample_id}.tif"
                        mask_tif_path = mask_tif_dir / f"{sample_id}.tif"

                        np.save(image_npy_path, patch_img)
                        np.save(mask_npy_path, patch_mask)

                        save_multiband_geotiff(
                            output_path=image_tif_path,
                            image=patch_img,
                            transform=patch_transform,
                            crs=self.sentinel_crs,
                            band_names=bands,
                        )

                        save_mask_geotiff(
                            output_path=mask_tif_path,
                            mask=patch_mask,
                            transform=patch_transform,
                            crs=self.sentinel_crs,
                        )

                        writer.writerow(
                            [
                                PATCH_LOG_SCHEMA_VERSION,
                                sample_id,
                                product_id_for_log,
                                safe_id_for_log,
                                scene_date_for_log,
                                tile_id_for_log,
                                roi_idx,
                                int(col),
                                int(row),
                                int(roi_img.window.col_off + col),
                                int(roi_img.window.row_off + row),
                                int(patch_size),
                                _relative_posix(image_npy_path, output_dir),
                                _relative_posix(mask_npy_path, output_dir),
                                _relative_posix(image_tif_path, output_dir),
                                _relative_posix(mask_tif_path, output_dir),
                                "npy;geotiff",
                                "npy;geotiff",
                                str(self.sentinel_crs),
                                f"{patch_transform.a:.12g}",
                                f"{patch_transform.b:.12g}",
                                f"{patch_transform.c:.12g}",
                                f"{patch_transform.d:.12g}",
                                f"{patch_transform.e:.12g}",
                                f"{patch_transform.f:.12g}",
                                f"{min_x:.3f}",
                                f"{min_y:.3f}",
                                f"{max_x:.3f}",
                                f"{max_y:.3f}",
                                f"{min_lng:.8f}",
                                f"{min_lat:.8f}",
                                f"{max_lng:.8f}",
                                f"{max_lat:.8f}",
                                band_order,
                                _class_pixel_counts(patch_mask),
                                f"{foreground_ratio:.4f}",
                                f"{nodata_ratio:.4f}",
                                f"{cloud_ratio:.4f}",
                                f"{cloud_on_fg_ratio:.4f}",
                                f"{bright_ratio:.4f}",
                                f"{bright_on_fg_ratio:.4f}",
                                "" if target_class_id is None else int(target_class_id),
                                target_pixels_for_log,
                                target_ratio_for_log,
                            ]
                        )

                        saved_global += 1
                        roi_saved += 1

                        if roi_saved >= max_patches_per_roi:
                            break

                    if roi_saved >= max_patches_per_roi:
                        break

                print(
                    f"[ENGINE] ROI summary: tried={attempted}, saved={roi_saved}, "
                    f"fg_rej={rej_fg}, cloud_rej={rej_cloud}, bright_rej={rej_bright}, "
                    f"nodata_rej={rej_nodata}, flat_rej={rej_flat}, "
                    f"target_rej={rej_target}"
                )

        new_saved = saved_global - start_index

        print(f"\n[ENGINE] TOTAL new patches: {new_saved}")

        return new_saved