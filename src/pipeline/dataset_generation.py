from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
from shapely.geometry import box

from src.catalog.stac_sentinel_catalog_searcher import STACSentinelCatalogSearcher
from src.config_loader import load_config
from src.data.eurocrops_mask_loader import EuroCropsMaskLoader
from src.data.sentinel_cache_manager import SentinelCacheManager
from src.data.sentinel_safe_loader import SentinelSafeDataLoader
from src.data_engine import DataEngine
from src.selection.roi_selectors.base import RoiSelectionContext
from src.selection.roi_selectors.grid_coverage_selector import GridCoverageSelector
from src.selection.roi_selectors.hotcell_selector import HotcellSelector
from src.utils.spatial_filters import tile_intersects_eurocrops


Roi = Tuple[float, float, float, float]


@dataclass(frozen=True)
class DatasetGenerationResult:
    output_dir: Path
    total_samples: int
    selected_rois: int
    processed_products: int
    strategy: str
    run_tag: str
    processed_state_path: Path
    satellite: str = "sentinel2"
    label_source: str = "eurocrops"


def roi_to_wgs84(roi: Roi, src_crs: str):
    gdf = gpd.GeoDataFrame(geometry=[box(*roi)], crs=src_crs)
    return gdf.to_crs("EPSG:4326").geometry.iloc[0].bounds


def make_roi_key(roi: Roi, decimals: int = 1) -> str:
    minx, miny, maxx, maxy = roi
    return (
        f"{round(float(minx), decimals)}_"
        f"{round(float(miny), decimals)}_"
        f"{round(float(maxx), decimals)}_"
        f"{round(float(maxy), decimals)}"
    )


def _read_target_config(config: dict) -> tuple[Optional[int], float, int, int]:
    target_cfg = (config.get("dataset", {}).get("target", {}) or {})

    class_id = target_cfg.get("class_id", None)
    class_id = int(class_id) if class_id is not None else None

    min_ratio = float(target_cfg.get("min_target_ratio", 0.0))
    min_pixels = int(target_cfg.get("min_target_pixels", 0))
    total_patches = int(target_cfg.get("total_patches", 0) or 0)

    return class_id, min_ratio, min_pixels, total_patches


def _read_generation_config(config: dict) -> tuple[str, str]:
    generation_cfg = config.get("generation", {}) or {}

    satellite = str(generation_cfg.get("satellite", "sentinel2")).lower().strip()
    label_source = str(generation_cfg.get("label_source", "eurocrops")).lower().strip()

    if satellite != "sentinel2":
        raise ValueError(
            f"Unsupported generation.satellite={satellite!r}. "
            "Currently only 'sentinel2' is supported."
        )

    if label_source != "eurocrops":
        raise ValueError(
            f"Unsupported generation.label_source={label_source!r}. "
            "Currently only 'eurocrops' is supported."
        )

    return satellite, label_source


def _empty_result(
    *,
    output_dir: Path,
    selected_rois: int,
    strategy: str,
    run_tag: str,
    processed_state_path: Path,
    satellite: str = "sentinel2",
    label_source: str = "eurocrops",
) -> DatasetGenerationResult:
    return DatasetGenerationResult(
        output_dir=output_dir,
        total_samples=0,
        selected_rois=selected_rois,
        processed_products=0,
        strategy=strategy,
        run_tag=run_tag,
        processed_state_path=processed_state_path,
        satellite=satellite,
        label_source=label_source,
    )


def generate_dataset(
    config_dir: str | Path = "config",
    run_tag: str = "v2",
    *,
    sleep_seconds: float = 1.0,
    overrides: Sequence[str] | None = None,
) -> DatasetGenerationResult:
    """
    Generate a Sentinel-2 / EuroCrops segmentation dataset.

    This function is the programmatic API for dataset generation. It can be used
    from Python code, while the CLI script can call the same function internally.
    """
    print("[MAIN] Starting dataset generation")
    print(f"[MAIN] config_dir={config_dir}")
    print(f"[MAIN] run_tag={run_tag}")

    if overrides:
        print(f"[MAIN] config_overrides={len(overrides)}")

    config = load_config(str(config_dir), overrides=overrides)

    satellite, label_source = _read_generation_config(config)
    print(f"[MAIN] Generation source: satellite={satellite}, label_source={label_source}")

    mask_loader = EuroCropsMaskLoader(config)
    gdf = mask_loader.gdf

    print(f"[MAIN] Loaded EuroCrops polygons: {len(gdf)}")
    print(f"[MAIN] EuroCrops CRS (native): {gdf.crs}")

    roi_crs = gdf.crs.to_string()

    gdf = gdf[gdf[mask_loader.crop_column].apply(
        lambda x: mask_loader.map_crop_to_class_id(x) > 0
    )].copy()

    gdf["class_id"] = gdf[mask_loader.crop_column].apply(
        mask_loader.map_crop_to_class_id
    ).astype(int)

    print(f"[MAIN] Filtered polygons: {len(gdf)}")

    roi_cfg = config.get("dataset", {}).get("roi", {}) or {}
    strategy = str(roi_cfg.get("strategy", "grid")).lower().strip()

    sentinel_root = Path(config["paths"]["sentinel_data_root"])
    processed_path = sentinel_root / f"processed_{strategy}_{run_tag}.json"
    output_dir = Path(config["paths"]["dataset_output_dir"])

    if gdf.empty:
        print("[MAIN] No valid crops after filtering. Exiting.")
        return _empty_result(
            output_dir=output_dir,
            selected_rois=0,
            strategy=strategy,
            run_tag=run_tag,
            processed_state_path=processed_path,
            satellite=satellite,
            label_source=label_source,
        )

    target_class_id, min_target_ratio, min_target_pixels, target_total_patches = _read_target_config(config)

    if target_class_id is None:
        print("[MAIN] Target mode: OFF")
    else:
        print(
            f"[MAIN] Target mode: ON (class_id={target_class_id}, "
            f"min_target_ratio={min_target_ratio}, min_target_pixels={min_target_pixels}, "
            f"total_patches={target_total_patches})"
        )

    print(f"[MAIN] ROI strategy from config: {strategy}")

    if strategy not in {"grid", "hotcell"}:
        raise ValueError(f"Unknown ROI strategy '{strategy}'. Use 'grid' or 'hotcell'.")

    if strategy == "hotcell" and target_class_id is None:
        raise ValueError("Hotcell strategy requires dataset.target.class_id to be set (not null).")

    selector = GridCoverageSelector() if strategy == "grid" else HotcellSelector()

    ctx = RoiSelectionContext(gdf=gdf, config=config)
    result = selector.select(ctx)
    rois = result.rois

    print(f"[MAIN] ROI strategy={result.strategy}, selected_rois={len(rois)}")
    if result.meta:
        print(f"[MAIN] ROI meta: {result.meta}")

    if not rois:
        print("[MAIN] No ROIs selected. Exiting.")
        return _empty_result(
            output_dir=output_dir,
            selected_rois=0,
            strategy=strategy,
            run_tag=run_tag,
            processed_state_path=processed_path,
            satellite=satellite,
            label_source=label_source,
        )

    print(f"[MAIN] Processed state file: {processed_path}")

    cache = SentinelCacheManager(config, processed_state_path=str(processed_path))
    searcher = STACSentinelCatalogSearcher()

    total_saved = 0
    processed_products = 0

    print(f"[MAIN] Dataset output dir: {output_dir}")

    for roi_idx, roi in enumerate(rois, start=1):
        if target_total_patches > 0 and total_saved >= target_total_patches:
            print(f"[MAIN] Reached target_total_patches={target_total_patches}. Stopping.")
            break

        print(f"\n[MAIN] ROI {roi_idx}/{len(rois)}")

        roi_wgs84 = roi_to_wgs84(roi, roi_crs)

        try:
            products = searcher.search(
                roi=roi_wgs84,
                start_date=config["catalog"]["stac"]["start_date"],
                end_date=config["catalog"]["stac"]["end_date"],
                max_cloud_cover=config["catalog"]["stac"]["max_cloud_cover"],
                limit=config["catalog"]["stac"]["limit"],
            )
        except Exception as e:
            print(f"[STAC] Search failed for ROI {roi_idx}: {e}")
            continue

        print(f"[MAIN] Found {len(products)} products")
        rkey = make_roi_key(roi)

        for product in products:
            if target_total_patches > 0 and total_saved >= target_total_patches:
                print(f"[MAIN] Reached target_total_patches={target_total_patches}. Stopping.")
                break

            if not tile_intersects_eurocrops(product, mask_loader.gdf):
                print(f"[MAIN] Skipping {product.product_id} (no EuroCrops overlap)")
                continue

            roi_product_key = f"{rkey}__{product.product_id}"

            if cache.is_processed(roi_product_key):
                print(f"[MAIN] {roi_product_key} already processed (strategy={strategy}, run_tag={run_tag})")
                continue

            print(f"[MAIN] Processing {roi_product_key} (strategy={strategy}, run_tag={run_tag})")
            print(
                f"[MAIN] Target filter passed to engine: class_id={target_class_id}, "
                f"min_ratio={min_target_ratio}, min_pixels={min_target_pixels}"
            )

            try:
                safe_id = cache.ensure_safe_available(product)
            except Exception as e:
                print(f"[CACHE] Failed to prepare SAFE for {product.product_id}: {e}")
                continue

            sentinel_loader = SentinelSafeDataLoader(config=config, safe_id=safe_id)

            engine = DataEngine(
                config=config,
                sentinel_loader=sentinel_loader,
                mask_loader=mask_loader,
            )

            try:
                saved_now = engine.save_dataset(
                    output_dir=str(output_dir),
                    rois=[roi],
                    bands=list(config["dataset"]["bands"].keys()),
                    target_class_id=target_class_id,
                    min_target_ratio=min_target_ratio,
                    min_target_pixels=min_target_pixels,
                    source_product_id=product.product_id,
                    source_safe_id=safe_id,
                )
            except Exception as e:
                print(f"[ENGINE] Failed on {roi_product_key}: {e}")
                continue

            print(f"[MAIN] Saved {saved_now} samples from {roi_product_key}")
            total_saved += int(saved_now)
            processed_products += 1

            cache.mark_processed(roi_product_key)
            cache.cleanup_if_needed()

            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))

    print("\n[MAIN] Dataset generation finished")
    print(f"[MAIN] TOTAL samples saved: {total_saved}")

    return DatasetGenerationResult(
        output_dir=output_dir,
        total_samples=total_saved,
        selected_rois=len(rois),
        processed_products=processed_products,
        strategy=strategy,
        run_tag=run_tag,
        processed_state_path=processed_path,
        satellite=satellite,
        label_source=label_source,
    )