"""Lossless-column adapter from DataEngine patches_log.csv v3 to training manifests.

The generator's original columns are preserved. Derived fields map the source
CRS/bounds and choose one of its NPY/GeoTIFF image-mask pairs. No source files
are ever modified or copied by this adapter.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError

from sentinel_crop_seg.core.splitting.spatial_split import rebase_asset_paths


PATCH_LOG_VERSION = "3"
REQUIRED_COLUMNS = frozenset({
    "log_schema_version", "sample_id", "product_id", "safe_id",
    "scene_date", "tile_id", "crs", "min_x", "min_y", "max_x",
    "max_y", "image_npy_path", "mask_npy_path", "image_tif_path",
    "mask_tif_path", "image_formats", "mask_formats",
    "band_order", "class_pixel_counts",
})
SOURCE_PATH_COLUMNS = {
    "npy": ("image_npy_path", "mask_npy_path", ".npy"),
    "geotiff": ("image_tif_path", "mask_tif_path", ".tif"),
}
ADDED_FIELDS = frozenset({
    "dataset", "epsg", "bbox_minx", "bbox_miny", "bbox_maxx",
    "bbox_maxy", "img_path", "mask_path", "acquisition_date",
    "country", "cohort", "label_source", "label_source_ref",
})


def _identifier(value: str | None, field: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} must be supplied explicitly")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string when provided")
    return value.strip()


def _date_or_blank(value: str) -> str:
    if not value.strip():
        return ""
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scene_date={value!r}; expected YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"Invalid scene_date={value!r}; expected YYYY-MM-DD")
    return value


def _parse_counts(raw: str, *, row: int) -> dict[int, int]:
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row}: invalid class_pixel_counts JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"row {row}: class_pixel_counts must be a JSON object")
    parsed: dict[int, int] = {}
    for key, value in decoded.items():
        if (not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                or isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"row {row}: invalid class id/count in class_pixel_counts")
        class_id = int(key)
        if class_id in parsed:
            raise ValueError(f"row {row}: repeated class id in class_pixel_counts")
        parsed[class_id] = value
    return parsed


def adapt_patch_log(
    log: pd.DataFrame,
    *,
    dataset_id: str,
    asset_format: str = "npy",
    country: str | None = None,
    cohort: str | None = None,
    label_source: str | None = None,
    label_source_ref: str | None = None,
) -> pd.DataFrame:
    """Adapt a v3 patch log to a training manifest without losing columns.

    Original product IDs, scene metadata, raster coordinates and source
    histograms are preserved; unknown optional provenance is never invented.

    Args:
        log: DataFrame containing the generator's v3 patch-log schema.
        dataset_id: Explicit dataset identity assigned to every sample.
        asset_format: `npy` or `geotiff` image/mask assets to reference.
        country: Optional source country identifier.
        cohort: Optional generation cohort identifier.
        label_source: Optional label provider name.
        label_source_ref: Optional label source version/reference.

    Returns:
        A copied DataFrame with manifest fields and class-pixel columns.

    Raises:
        ValueError: If a required column, geometry, identity or asset entry
            fails validation.
    """
    dataset_id = _identifier(dataset_id, "dataset_id", required=True)
    country = _identifier(country, "country")
    cohort = _identifier(cohort, "cohort")
    label_source = _identifier(label_source, "label_source")
    label_source_ref = _identifier(label_source_ref, "label_source_ref")
    if label_source_ref and not label_source:
        raise ValueError("label_source_ref requires an explicit label_source")
    if asset_format not in SOURCE_PATH_COLUMNS:
        raise ValueError(f"asset_format must be one of {sorted(SOURCE_PATH_COLUMNS)}")
    if log.empty:
        raise ValueError("patches_log.csv is empty")
    missing = REQUIRED_COLUMNS - set(log.columns)
    if missing:
        raise ValueError(f"patches_log.csv missing v3 columns: {sorted(missing)}")
    reserved = ADDED_FIELDS & set(log.columns)
    if reserved or any(
        len(c) >= 2 and c[0] == "p" and c[1:].isascii() and c[1:].isdigit()
        for c in log.columns
    ):
        raise ValueError(f"patch log already has derived manifest columns: {sorted(reserved)}")
    work = log.copy().reset_index(drop=True)
    version = work["log_schema_version"].astype(str).str.strip()
    if not version.eq(PATCH_LOG_VERSION).all():
        raise ValueError("Only DataEngine patch log schema v3 is supported")
    ids = work["sample_id"].astype(str)
    if ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("sample_id must be unique and non-empty within a patch log")

    image_key, mask_key, suffix = SOURCE_PATH_COLUMNS[asset_format]
    for kind, column in (("image", image_key), ("mask", mask_key)):
        values = work[column].astype(str)
        if values.str.strip().eq("").any():
            raise ValueError(f"Missing {kind} path for {asset_format}")
        if not values.str.lower().str.endswith(suffix).all():
            raise ValueError(f"{column} must contain {suffix} paths")
        formats_key = f"{kind}_formats"
        for index, formats in enumerate(work[formats_key].astype(str)):
            options = {part.strip().lower() for part in formats.split(";")}
            if asset_format not in options:
                raise ValueError(f"row {index}: {kind}_formats does not advertise {asset_format}")

    codes = []
    for row, value in enumerate(work["crs"].astype(str)):
        try:
            crs = CRS.from_user_input(value)
        except (TypeError, ValueError, CRSError) as exc:
            raise ValueError(f"row {row}: unrecognized CRS {value!r}") from exc
        epsg = crs.to_epsg()
        if not crs.is_projected or epsg is None:
            raise ValueError(
                f"row {row}: CRS must be a projected EPSG CRS to build a spatial split"
            )
        codes.append(epsg)

    bounds = (("min_x", "bbox_minx"), ("min_y", "bbox_miny"),
              ("max_x", "bbox_maxx"), ("max_y", "bbox_maxy"))
    for original, derived in bounds:
        values = pd.to_numeric(work[original], errors="raise")
        if not pd.Series(values).map(lambda v: pd.notna(v) and abs(v) < float("inf")).all():
            raise ValueError(f"Non-finite {original} in patch log")
        work[derived] = values.astype(float)
    if not ((work["bbox_minx"] < work["bbox_maxx"]) &
            (work["bbox_miny"] < work["bbox_maxy"])).all():
        raise ValueError("Patch log contains invalid projected bounds")

    histograms = [_parse_counts(raw, row=i) for i, raw in enumerate(
        work["class_pixel_counts"].astype(str)
    )]
    for class_id in sorted({key for histogram in histograms for key in histogram}):
        work[f"p{class_id}"] = [counts.get(class_id, 0) for counts in histograms]

    work["dataset"] = dataset_id
    work["epsg"] = codes
    work["img_path"] = work[image_key]
    work["mask_path"] = work[mask_key]
    work["acquisition_date"] = [
        _date_or_blank(value) for value in work["scene_date"].astype(str)
    ]
    if country is not None:
        work["country"] = country
    if cohort is not None:
        work["cohort"] = cohort
    if label_source is not None:
        work["label_source"] = label_source
    if label_source_ref is not None:
        work["label_source_ref"] = label_source_ref
    return work


def adapt_patch_log_file(
    patch_log: str | Path,
    output_manifest: str | Path,
    *,
    dataset_id: str,
    asset_format: str = "npy",
    country: str | None = None,
    cohort: str | None = None,
    label_source: str | None = None,
    label_source_ref: str | None = None,
    verify_assets: bool = False,
    validate_only: bool = False,
) -> dict:
    """Validate a patch log and optionally write a new, path-rebased manifest.

    Args:
        patch_log: Existing version-3 patch log CSV.
        output_manifest: New CSV destination; existing files are not overwritten.
        dataset_id: Explicit identity of the generated dataset.
        asset_format: `npy` or `geotiff` asset references.
        country: Optional country provenance.
        cohort: Optional cohort provenance.
        label_source: Optional label source name.
        label_source_ref: Optional label source version/reference.
        verify_assets: Check that referenced image/mask files exist.
        validate_only: Return the audit without writing output files.

    Returns:
        A report containing conversion counts and source provenance hashes.

    Raises:
        ValueError: If source rows or asset paths fail validation.
        FileExistsError: If a requested output file already exists.
    """
    source = Path(patch_log)
    output = Path(output_manifest)
    if not validate_only and (output.exists() or output.with_suffix(".provenance.json").exists()):
        raise FileExistsError("Output manifest or its provenance sidecar already exists")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    converted = adapt_patch_log(
        raw, dataset_id=dataset_id, asset_format=asset_format,
        country=country, cohort=cohort, label_source=label_source,
        label_source_ref=label_source_ref,
    )
    # The original log's relative paths are relative to its directory.
    # Derive consumable paths relative to the *new* manifest, leaving the
    # original log columns untouched.
    converted = rebase_asset_paths(converted, source, output.parent)
    if verify_assets:
        for column in ("img_path", "mask_path"):
            for value in converted[column]:
                target = Path(value)
                resolved = target if target.is_absolute() else (output.parent / target).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f"Unresolved {column}: {resolved}")
    summary = {
        "input_log_file": source.name,
        "input_log_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_log_schema": PATCH_LOG_VERSION,
        "dataset": dataset_id,
        "asset_format": asset_format,
        "samples": len(converted),
        "classes_in_histograms": sorted(
            int(col[1:]) for col in converted.columns
            if len(col) > 1 and col[0] == "p" and col[1:].isdigit()
        ),
        "missing_product_ids": int(converted["product_id"].eq("").sum()),
        "missing_safe_ids": int(converted["safe_id"].eq("").sum()),
        "missing_scene_dates": int(converted["acquisition_date"].eq("").sum()),
        "missing_tile_ids": int(converted["tile_id"].eq("").sum()),
        "country_provided": country is not None,
        "cohort_provided": cohort is not None,
        "label_source_provided": label_source is not None,
        "label_source_ref_provided": label_source_ref is not None,
        "verified_asset_paths": bool(verify_assets),
        "original_columns_preserved": list(raw.columns),
    }
    if validate_only:
        return summary
    if output.exists() or output.with_suffix(".provenance.json").exists():
        raise FileExistsError("Output manifest or its provenance sidecar already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output, index=False)
    output.with_suffix(".provenance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
