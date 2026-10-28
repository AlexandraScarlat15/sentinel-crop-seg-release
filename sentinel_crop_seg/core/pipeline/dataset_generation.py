"""Generate Sentinel-2/EuroCrops patch datasets using validated configuration."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
from shapely.geometry import box

from sentinel_crop_seg.core.catalog.stac_sentinel_catalog_searcher import STACSentinelCatalogSearcher
from sentinel_crop_seg.core.pipeline.generation_config import load_generation_config
from sentinel_crop_seg.core.pipeline.acquisition_calendar import AcquisitionCalendar
from sentinel_crop_seg.core.pipeline.generation_run_guard import GenerationRunGuard
from sentinel_crop_seg.core.data.eurocrops_mask_loader import EuroCropsMaskLoader
from sentinel_crop_seg.core.data.sentinel_cache_manager import SentinelCacheManager
from sentinel_crop_seg.core.data.sentinel_safe_loader import SentinelSafeDataLoader
from sentinel_crop_seg.core.data_engine import DataEngine
from sentinel_crop_seg.core.selection.roi_selectors.base import RoiSelectionContext
from sentinel_crop_seg.core.selection.roi_selectors.grid_coverage_selector import GridCoverageSelector
from sentinel_crop_seg.core.selection.roi_selectors.hotcell_selector import HotcellSelector
from sentinel_crop_seg.core.utils.spatial_filters import tile_intersects_eurocrops


Roi = Tuple[float, float, float, float]


@dataclass(frozen=True)
class DatasetGenerationResult:
    """Summarize a generation invocation and its resulting output locations."""

    output_dir: Path
    total_samples: int
    selected_rois: int
    processed_products: int
    strategy: str
    run_tag: str
    processed_state_path: Path
    satellite: str = "sentinel2"
    label_source: str = "eurocrops"
    # Counted separately from "this ROI had no scenes". A raised search is a
    # fault -- bad credentials, a rejected parameter, an API or network
    # problem -- and a run where every search faulted must not look like a
    # run over empty regions.
    search_failures: int = 0


def roi_to_wgs84(roi: Roi, src_crs: str):
    """Transform a projected ROI bounding box to longitude/latitude bounds.

    Args:
        roi: Source-CRS bounds as `(xmin, ymin, xmax, ymax)`.
        src_crs: Coordinate reference system of the ROI.

    Returns:
        Geographic bounding box in EPSG:4326 coordinates.
    """
    gdf = gpd.GeoDataFrame(geometry=[box(*roi)], crs=src_crs)
    return gdf.to_crs("EPSG:4326").geometry.iloc[0].bounds


def make_roi_key(roi: Roi, decimals: int = 1) -> str:
    """Create a stable, rounded identifier for a projected ROI.

    Args:
        roi: Bounding box in the source coordinate system.
        decimals: Number of decimal digits used in coordinate rounding.

    Returns:
        Underscore-separated coordinate key.
    """
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
    config_mode: str = "legacy",
    resume_policy: str = "legacy",
    experiment_recipe: str | Path | None = None,
) -> DatasetGenerationResult:
    """Generate a dataset using legacy semantics or a guarded isolated recipe.

    Args:
        config_dir: Directory containing YAML generator configuration.
        run_tag: Unique name used in generation run-state filenames.
        sleep_seconds: Pause between processed products, in seconds.
        overrides: Optional dotted-key configuration overrides.
        config_mode: `legacy` or `isolated` configuration resolution.
        resume_policy: `legacy` behavior or fail-closed `strict` resume.
        experiment_recipe: Optional versioned recipe for isolated generation.

    Returns:
        Summary of generated samples, products and output locations.

    Raises:
        ValueError: If resume settings or a source recipe are incompatible.
        RuntimeError: If a strict run fails its provenance or state checks.
    """
    if resume_policy not in ("legacy", "strict"):
        raise ValueError("resume_policy must be 'legacy' or 'strict'")
    print("[MAIN] Starting dataset generation")
    print(f"[MAIN] config_dir={config_dir}")
    print(f"[MAIN] run_tag={run_tag}")
    if overrides:
        print(f"[MAIN] config_overrides={len(overrides)}")

    config = load_generation_config(str(config_dir), mode=config_mode, overrides=overrides, experiment_recipe=experiment_recipe)
    if resume_policy == "legacy":
        return _generate_dataset_with_config(config, run_tag, sleep_seconds, None)

    if config_mode != "isolated":
        raise ValueError("Strict resume requires --config-mode isolated")
    roi_cfg = config.get("dataset", {}).get("roi", {})
    strategy = str(roi_cfg.get("strategy", "grid")).lower().strip()
    root = Path(config["paths"]["sentinel_data_root"])
    output = Path(config["paths"]["dataset_output_dir"])
    processed = root / f"processed_{strategy}_{run_tag}.json"
    guard = GenerationRunGuard(
        config, run_tag, output, processed, config_mode=config_mode
    )
    with guard:
        result = _generate_dataset_with_config(config, run_tag, sleep_seconds, guard)
        guard.finish()
        return result


def _generate_dataset_with_config(
    config: dict,
    run_tag: str,
    sleep_seconds: float,
    strict_guard: GenerationRunGuard | None,
) -> DatasetGenerationResult:
    """Internal pipeline implementation; no changes to legacy filter semantics."""
    satellite, label_source = _read_generation_config(config)
    calendar = AcquisitionCalendar.from_config(config)
    if calendar is not None and config.get("acquisition_calendar_provenance") != calendar.provenance():
        raise ValueError("Calendar provenance changed after configuration resolution")
    stac_settings = config["catalog"]["stac"]
    intervals = (
        calendar.search_intervals(stac_settings["start_date"], stac_settings["end_date"])
        if calendar else [(stac_settings["start_date"], stac_settings["end_date"])]
    )
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
    prior_saved = strict_guard.initial_samples if strict_guard is not None else 0
    processed_products = 0
    search_failures = 0

    print(f"[MAIN] Dataset output dir: {output_dir}")

    for roi_idx, roi in enumerate(rois, start=1):
        if target_total_patches > 0 and prior_saved + total_saved >= target_total_patches:
            print(f"[MAIN] Reached target_total_patches={target_total_patches}. Stopping.")
            break

        print(f"\n[MAIN] ROI {roi_idx}/{len(rois)}")

        roi_wgs84 = roi_to_wgs84(roi, roi_crs)

        products = []
        try:
            for start_date, end_date in intervals:
                products.extend(searcher.search(
                    roi=roi_wgs84,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=stac_settings["max_cloud_cover"],
                    limit=stac_settings["limit"],
                ))
        except Exception as exc:
            if calendar is not None:
                raise RuntimeError(
                    f"Calendar-controlled STAC search failed for ROI {roi_idx}"
                ) from exc
            search_failures += 1
            print(f"[STAC] Search failed for ROI {roi_idx}: {exc}")
            continue

        if calendar is not None:
            products = [
                product for product in products
                if calendar.accepts(getattr(product, "acquisition_date", None))
            ]
            products = list({product.product_id: product for product in products}.values())
        print(f"[MAIN] Found {len(products)} products")
        rkey = make_roi_key(roi)

        for product in products:
            if target_total_patches > 0 and prior_saved + total_saved >= target_total_patches:
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

            # The total target applies to *newly accepted* patches in this
            # invocation. The engine receives the remainder so its final
            # product/ROI cannot overshoot the requested total.
            remaining_patches = (
                target_total_patches - prior_saved - total_saved if target_total_patches > 0
                else None
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
                    max_new_patches=remaining_patches,
                )
            except Exception as e:
                if strict_guard is not None:
                    # The engine may already have written some patch assets.
                    # Preserve the in-flight ledger and refuse unsafe resume.
                    raise RuntimeError(
                        f"Strict generation aborted during {roi_product_key}; "
                        "manual review required before reuse"
                    ) from e
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
    if search_failures:
        print(f"[MAIN] WARNING: catalog search faulted for {search_failures} "
              f"of {len(rois)} ROIs")

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
        search_failures=search_failures,
    )