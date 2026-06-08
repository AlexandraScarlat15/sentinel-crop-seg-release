from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_PASTIS_BAND_ORDER = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

PROJECT_BAND_ORDER = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PASTIS Sentinel-2 time series into a single-date NPY "
            "segmentation dataset compatible with this project."
        )
    )

    parser.add_argument(
        "--pastis-root",
        required=True,
        type=str,
        help="Path to extracted PASTIS root, for example .../PASTIS_redownload/extracted/PASTIS.",
    )

    parser.add_argument(
        "--mapping-yaml",
        required=True,
        type=str,
        help="Path to config/pastis/label_mapping.yaml.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output dataset directory. It will contain images/, masks/, metadata.csv.",
    )

    parser.add_argument(
        "--month-start",
        required=True,
        type=int,
        help="First acquisition month to keep, for example 4.",
    )

    parser.add_argument(
        "--month-end",
        required=True,
        type=int,
        help="Last acquisition month to keep, for example 9.",
    )

    parser.add_argument(
        "--fold",
        default=None,
        type=int,
        help="Optional PASTIS fold filter. If omitted, all folds are used.",
    )

    parser.add_argument(
        "--limit-samples",
        default=0,
        type=int,
        help="Optional maximum number of converted samples. 0 means no limit.",
    )

    parser.add_argument(
        "--limit-patches",
        default=0,
        type=int,
        help="Optional maximum number of PASTIS patches to inspect. 0 means no limit.",
    )

    parser.add_argument(
        "--min-valid-pixels",
        default=1,
        type=int,
        help="Minimum number of pixels not equal to ignore value.",
    )

    parser.add_argument(
        "--min-nonbackground-pixels",
        default=1,
        type=int,
        help="Minimum number of valid pixels with class different from background.",
    )

    parser.add_argument(
        "--min-interest-ratio-total",
        default=0.0,
        type=float,
        help=(
            "Minimum ratio of crop-interest pixels over all pixels. "
            "Interest pixels are valid pixels with class id in 1..num_classes-1."
        ),
    )

    parser.add_argument(
        "--min-interest-ratio-valid",
        default=0.0,
        type=float,
        help=(
            "Minimum ratio of crop-interest pixels over valid pixels. "
            "Valid pixels exclude the ignore value."
        ),
    )

    parser.add_argument(
        "--max-white-cloud-ratio",
        default=1.0,
        type=float,
        help="Maximum allowed ratio of white/cloud-like pixels. Default keeps all samples.",
    )

    parser.add_argument(
        "--max-very-bright-ratio",
        default=1.0,
        type=float,
        help="Maximum allowed ratio of very bright pixels. Default keeps all samples.",
    )

    parser.add_argument(
        "--max-bright-ratio",
        default=1.0,
        type=float,
        help="Maximum allowed ratio of bright pixels. Default keeps all samples.",
    )

    parser.add_argument(
        "--max-cloud-score",
        default=None,
        type=float,
        help="Maximum allowed cloud score. If omitted, cloud score is not used as a filter.",
    )

    parser.add_argument(
        "--min-vegetation-ratio",
        default=0.0,
        type=float,
        help="Minimum allowed ratio of pixels with NDVI > threshold.",
    )

    parser.add_argument(
        "--ndvi-threshold",
        default=0.35,
        type=float,
        help="NDVI threshold used for vegetation_ratio.",
    )

    parser.add_argument(
        "--select-best-per-patch",
        action="store_true",
        help=(
            "Keep only the best acquisition for each patch according to cloud_score. "
            "This is recommended for converting PASTIS time series into single-date samples."
        ),
    )

    parser.add_argument(
        "--band-reorder",
        default=None,
        type=str,
        help=(
            "Optional comma-separated band index order. "
            "Example: 0,1,2,6,3,4,5,7,8,9 converts the common PASTIS order "
            "B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 to the project order "
            "B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12."
        ),
    )

    parser.add_argument(
        "--clip-reflectance",
        action="store_true",
        help=(
            "Clip Sentinel-2 values after band reordering. "
            "Recommended for PASTIS because THEIA-corrected values may be negative "
            "or above the usual 0..10000 range."
        ),
    )

    parser.add_argument(
        "--clip-min",
        default=0.0,
        type=float,
        help="Minimum value used when --clip-reflectance is enabled.",
    )

    parser.add_argument(
        "--clip-max",
        default=10000.0,
        type=float,
        help="Maximum value used when --clip-reflectance is enabled.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )

    return parser.parse_args()


def month_is_allowed(month: int, month_start: int, month_end: int) -> bool:
    if month_start <= month_end:
        return month_start <= month <= month_end

    return month >= month_start or month <= month_end


def parse_yyyymmdd(value: int | str) -> tuple[int, int, int]:
    text = str(value)

    if len(text) != 8:
        raise ValueError(f"Expected YYYYMMDD date, got {value!r}")

    year = int(text[:4])
    month = int(text[4:6])
    day = int(text[6:8])

    return year, month, day


def scene_date_from_yyyymmdd(value: int | str) -> str:
    """
    Convert a PASTIS acquisition date from YYYYMMDD to YYYY-MM-DD.

    Example:
    20180912 -> 2018-09-12
    """
    year, month, day = parse_yyyymmdd(value)
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_band_reorder(value: str | None) -> list[int] | None:
    if value is None:
        return None

    value = value.strip()

    if not value or value.lower() == "none":
        return None

    indices = [int(item.strip()) for item in value.split(",")]

    if len(indices) != 10:
        raise ValueError(f"Expected 10 band indices, got {len(indices)} from {value!r}")

    if sorted(indices) != list(range(10)):
        raise ValueError(f"Band reorder must be a permutation of 0..9, got {indices}")

    return indices


def get_output_band_order(band_reorder: list[int] | None) -> list[str]:
    if band_reorder is None:
        return DEFAULT_PASTIS_BAND_ORDER

    return [DEFAULT_PASTIS_BAND_ORDER[index] for index in band_reorder]


def load_mapping(mapping_yaml: Path) -> tuple[dict[int, int], int]:
    if not mapping_yaml.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_yaml}")

    with mapping_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    pastis_cfg = data.get("pastis", {})
    ignore_value = int(pastis_cfg.get("ignore_value", 255))

    raw_mapping = pastis_cfg.get("remap_to_project_class_id", {})
    if not raw_mapping:
        raise ValueError("Missing pastis.remap_to_project_class_id in mapping YAML.")

    mapping: dict[int, int] = {}

    for src, dst in raw_mapping.items():
        mapping[int(src)] = int(dst)

    return mapping, ignore_value


def remap_mask(
    semantic_mask: np.ndarray,
    mapping: dict[int, int],
    ignore_value: int,
) -> np.ndarray:
    if semantic_mask.ndim != 2:
        raise ValueError(f"Expected semantic mask with shape (H, W), got {semantic_mask.shape}")

    out = np.full(semantic_mask.shape, ignore_value, dtype=np.uint8)

    for src_class, dst_class in mapping.items():
        out[semantic_mask == src_class] = dst_class

    return out


def get_present_classes(mask: np.ndarray, ignore_value: int) -> list[int]:
    values = np.unique(mask)
    values = values[values != ignore_value]
    return [int(value) for value in values.tolist()]


def get_value_counts(mask: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(mask, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"PASTIS metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        raise ValueError("No features found in metadata.geojson.")

    rows: list[dict[str, Any]] = []

    for feature in features:
        props = feature.get("properties", {})
        patch_id = int(props["ID_PATCH"])

        rows.append(
            {
                "patch_id": patch_id,
                "fold": int(props["Fold"]),
                "tile": str(props.get("TILE", "")),
                "n_parcel": int(props.get("N_Parcel", 0)),
                "parcel_cover": float(props.get("Parcel_Cover", 0.0)),
                "dates_s2": props["dates-S2"],
            }
        )

    rows.sort(key=lambda item: item["patch_id"])
    return rows


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"

    if output_dir.exists():
        existing_files = list(output_dir.glob("*"))

        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite if this is intended."
            )

        if overwrite:
            if images_dir.exists():
                shutil.rmtree(images_dir)
            if masks_dir.exists():
                shutil.rmtree(masks_dir)

            for filename in ["metadata.csv", "dataset_info.json", "quality_report.csv"]:
                path = output_dir / filename
                if path.exists():
                    path.unlink()

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    return images_dir, masks_dir


def maybe_clip_reflectance(
    image: np.ndarray,
    enabled: bool,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    if not enabled:
        return image

    if clip_max <= clip_min:
        raise ValueError(f"clip_max must be greater than clip_min, got {clip_min} and {clip_max}")

    image = np.nan_to_num(
        image,
        nan=clip_min,
        neginf=clip_min,
        posinf=clip_max,
    )

    return np.clip(image, clip_min, clip_max)


def compute_mask_stats(mask: np.ndarray, ignore_value: int) -> dict[str, float | int]:
    total_pixels = int(mask.size)
    valid_pixels = int(np.sum(mask != ignore_value))
    background_pixels = int(np.sum(mask == 0))
    ignore_pixels = int(np.sum(mask == ignore_value))
    nonbackground_valid_pixels = int(np.sum((mask != ignore_value) & (mask != 0)))

    valid_ratio_total = valid_pixels / total_pixels if total_pixels > 0 else 0.0
    ignore_ratio_total = ignore_pixels / total_pixels if total_pixels > 0 else 0.0
    interest_ratio_total = nonbackground_valid_pixels / total_pixels if total_pixels > 0 else 0.0
    interest_ratio_valid = (
        nonbackground_valid_pixels / valid_pixels if valid_pixels > 0 else 0.0
    )

    return {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "background_pixels": background_pixels,
        "ignore_pixels": ignore_pixels,
        "nonbackground_valid_pixels": nonbackground_valid_pixels,
        "valid_ratio_total": valid_ratio_total,
        "ignore_ratio_total": ignore_ratio_total,
        "interest_ratio_total": interest_ratio_total,
        "interest_ratio_valid": interest_ratio_valid,
    }


def compute_quality_metrics(
    image_chw: np.ndarray,
    ndvi_threshold: float,
) -> dict[str, float]:
    if image_chw.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {image_chw.shape}")

    if image_chw.shape[0] < 4:
        raise ValueError(f"Expected at least 4 channels, got {image_chw.shape[0]}")

    # Project band order after optional reorder:
    # 0 B02, 1 B03, 2 B04, 3 B08, 4 B05, 5 B06, 6 B07, 7 B8A, 8 B11, 9 B12.
    b02 = image_chw[0].astype(np.float32, copy=False)
    b03 = image_chw[1].astype(np.float32, copy=False)
    b04 = image_chw[2].astype(np.float32, copy=False)
    b08 = image_chw[3].astype(np.float32, copy=False)

    rgb = np.stack([b04, b03, b02], axis=0)
    rgb_mean = rgb.mean(axis=0)
    rgb_range = rgb.max(axis=0) - rgb.min(axis=0)

    bright_ratio = float(np.mean(rgb_mean > 3000.0))
    very_bright_ratio = float(np.mean(rgb_mean > 4500.0))
    white_cloud_ratio = float(np.mean((rgb_mean > 2500.0) & (rgb_range < 450.0)))
    dark_ratio = float(np.mean(rgb_mean < 300.0))
    rgb_std = float(np.std(rgb_mean))
    rgb_mean_value = float(np.mean(rgb_mean))

    ndvi = (b08 - b04) / (b08 + b04 + 1e-6)
    ndvi_mean = float(np.nanmean(ndvi))
    vegetation_ratio = float(np.mean(ndvi > ndvi_threshold))

    cloud_score = (
        1.5 * white_cloud_ratio
        + 1.0 * very_bright_ratio
        + 0.5 * bright_ratio
        - 0.2 * vegetation_ratio
    )

    return {
        "bright_ratio": bright_ratio,
        "very_bright_ratio": very_bright_ratio,
        "white_cloud_ratio": white_cloud_ratio,
        "dark_ratio": dark_ratio,
        "rgb_std": rgb_std,
        "rgb_mean": rgb_mean_value,
        "ndvi_mean": ndvi_mean,
        "vegetation_ratio": vegetation_ratio,
        "cloud_score": float(cloud_score),
    }


def passes_filters(
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if candidate["valid_pixels"] < args.min_valid_pixels:
        return False, "min_valid_pixels"

    if candidate["nonbackground_valid_pixels"] < args.min_nonbackground_pixels:
        return False, "min_nonbackground_pixels"

    if candidate["interest_ratio_total"] < args.min_interest_ratio_total:
        return False, "min_interest_ratio_total"

    if candidate["interest_ratio_valid"] < args.min_interest_ratio_valid:
        return False, "min_interest_ratio_valid"

    if candidate["white_cloud_ratio"] > args.max_white_cloud_ratio:
        return False, "max_white_cloud_ratio"

    if candidate["very_bright_ratio"] > args.max_very_bright_ratio:
        return False, "max_very_bright_ratio"

    if candidate["bright_ratio"] > args.max_bright_ratio:
        return False, "max_bright_ratio"

    if args.max_cloud_score is not None and candidate["cloud_score"] > args.max_cloud_score:
        return False, "max_cloud_score"

    if candidate["vegetation_ratio"] < args.min_vegetation_ratio:
        return False, "min_vegetation_ratio"

    return True, ""


def save_metadata_csv(metadata_csv: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_name",
        "patch_id",
        "fold",
        "tile",
        "acquisition_index",
        "acquisition_date",
        "scene_date",
        "year",
        "month",
        "day",
        "source_s2",
        "source_target",
        "image_path",
        "mask_path",
        "height",
        "width",
        "num_channels",
        "band_reorder",
        "input_band_order",
        "output_band_order",
        "total_pixels",
        "valid_pixels",
        "background_pixels",
        "nonbackground_valid_pixels",
        "ignore_pixels",
        "valid_ratio_total",
        "ignore_ratio_total",
        "interest_ratio_total",
        "interest_ratio_valid",
        "bright_ratio",
        "very_bright_ratio",
        "white_cloud_ratio",
        "dark_ratio",
        "rgb_std",
        "rgb_mean",
        "ndvi_mean",
        "vegetation_ratio",
        "cloud_score",
        "present_project_classes",
        "project_value_counts",
        "present_pastis_semantic_classes",
        "pastis_semantic_value_counts",
        "n_parcel",
        "parcel_cover",
    ]

    with metadata_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_quality_report(quality_csv: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "patch_id",
        "fold",
        "tile",
        "acquisition_index",
        "acquisition_date",
        "year",
        "month",
        "day",
        "passed_filters",
        "skip_reason",
        "interest_ratio_total",
        "interest_ratio_valid",
        "valid_ratio_total",
        "ignore_ratio_total",
        "bright_ratio",
        "very_bright_ratio",
        "white_cloud_ratio",
        "dark_ratio",
        "rgb_std",
        "rgb_mean",
        "ndvi_mean",
        "vegetation_ratio",
        "cloud_score",
        "present_project_classes",
    ]

    with quality_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_dataset_info(
    output_dir: Path,
    pastis_root: Path,
    mapping_yaml: Path,
    args: argparse.Namespace,
    ignore_value: int,
    band_reorder: list[int] | None,
    output_band_order: list[str],
    num_samples: int,
    counters: dict[str, int],
) -> None:
    info = {
        "dataset_name": "pastis_single_date",
        "pastis_root": str(pastis_root),
        "mapping_yaml": str(mapping_yaml),
        "month_start": args.month_start,
        "month_end": args.month_end,
        "fold": args.fold,
        "limit_samples": args.limit_samples,
        "limit_patches": args.limit_patches,
        "ignore_value": ignore_value,
        "input_band_order": DEFAULT_PASTIS_BAND_ORDER,
        "band_reorder": band_reorder,
        "output_band_order": output_band_order,
        "expected_project_band_order": PROJECT_BAND_ORDER,
        "clip_reflectance": args.clip_reflectance,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "select_best_per_patch": args.select_best_per_patch,
        "min_valid_pixels": args.min_valid_pixels,
        "min_nonbackground_pixels": args.min_nonbackground_pixels,
        "min_interest_ratio_total": args.min_interest_ratio_total,
        "min_interest_ratio_valid": args.min_interest_ratio_valid,
        "max_white_cloud_ratio": args.max_white_cloud_ratio,
        "max_very_bright_ratio": args.max_very_bright_ratio,
        "max_bright_ratio": args.max_bright_ratio,
        "max_cloud_score": args.max_cloud_score,
        "min_vegetation_ratio": args.min_vegetation_ratio,
        "ndvi_threshold": args.ndvi_threshold,
        "num_samples": num_samples,
        "counters": counters,
        "image_shape": ["C", "H", "W"],
        "mask_shape": ["H", "W"],
        "semantic_mask_source": "TARGET[0]",
    }

    path = output_dir / "dataset_info.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def save_candidate(
    candidate: dict[str, Any],
    sample_index: int,
    images_dir: Path,
    masks_dir: Path,
    band_reorder: list[int] | None,
    output_band_order: list[str],
) -> dict[str, Any]:
    sample_name = f"sample_{sample_index:06d}.npy"

    image_out_path = images_dir / sample_name
    mask_out_path = masks_dir / sample_name

    np.save(image_out_path, candidate["image"])
    np.save(mask_out_path, candidate["mask"])

    row = {
        "sample_name": sample_name,
        "patch_id": candidate["patch_id"],
        "fold": candidate["fold"],
        "tile": candidate["tile"],
        "acquisition_index": candidate["acquisition_index"],
        "acquisition_date": candidate["acquisition_date"],
        "scene_date": scene_date_from_yyyymmdd(candidate["acquisition_date"]),
        "year": candidate["year"],
        "month": candidate["month"],
        "day": candidate["day"],
        "source_s2": str(candidate["source_s2"]),
        "source_target": str(candidate["source_target"]),
        "image_path": str(image_out_path),
        "mask_path": str(mask_out_path),
        "height": candidate["height"],
        "width": candidate["width"],
        "num_channels": candidate["num_channels"],
        "band_reorder": json.dumps(band_reorder),
        "input_band_order": json.dumps(DEFAULT_PASTIS_BAND_ORDER),
        "output_band_order": json.dumps(output_band_order),
        "total_pixels": candidate["total_pixels"],
        "valid_pixels": candidate["valid_pixels"],
        "background_pixels": candidate["background_pixels"],
        "nonbackground_valid_pixels": candidate["nonbackground_valid_pixels"],
        "ignore_pixels": candidate["ignore_pixels"],
        "valid_ratio_total": candidate["valid_ratio_total"],
        "ignore_ratio_total": candidate["ignore_ratio_total"],
        "interest_ratio_total": candidate["interest_ratio_total"],
        "interest_ratio_valid": candidate["interest_ratio_valid"],
        "bright_ratio": candidate["bright_ratio"],
        "very_bright_ratio": candidate["very_bright_ratio"],
        "white_cloud_ratio": candidate["white_cloud_ratio"],
        "dark_ratio": candidate["dark_ratio"],
        "rgb_std": candidate["rgb_std"],
        "rgb_mean": candidate["rgb_mean"],
        "ndvi_mean": candidate["ndvi_mean"],
        "vegetation_ratio": candidate["vegetation_ratio"],
        "cloud_score": candidate["cloud_score"],
        "present_project_classes": json.dumps(candidate["present_project_classes"]),
        "project_value_counts": json.dumps(candidate["project_value_counts"], sort_keys=True),
        "present_pastis_semantic_classes": json.dumps(candidate["present_pastis_semantic_classes"]),
        "pastis_semantic_value_counts": json.dumps(
            candidate["pastis_semantic_value_counts"],
            sort_keys=True,
        ),
        "n_parcel": candidate["n_parcel"],
        "parcel_cover": candidate["parcel_cover"],
    }

    return row


def build_candidate(
    image: np.ndarray,
    remapped_mask: np.ndarray,
    semantic_mask: np.ndarray,
    item: dict[str, Any],
    acquisition_index: int,
    acquisition_date: int,
    year: int,
    month: int,
    day: int,
    s2_path: Path,
    target_path: Path,
    ignore_value: int,
    ndvi_threshold: float,
) -> dict[str, Any]:
    mask_stats = compute_mask_stats(remapped_mask, ignore_value=ignore_value)
    quality_metrics = compute_quality_metrics(image, ndvi_threshold=ndvi_threshold)

    height, width = remapped_mask.shape
    num_channels = int(image.shape[0])

    candidate = {
        "image": image,
        "mask": remapped_mask.astype(np.uint8, copy=False),
        "patch_id": int(item["patch_id"]),
        "fold": int(item["fold"]),
        "tile": item["tile"],
        "acquisition_index": acquisition_index,
        "acquisition_date": acquisition_date,
        "year": year,
        "month": month,
        "day": day,
        "source_s2": s2_path,
        "source_target": target_path,
        "height": height,
        "width": width,
        "num_channels": num_channels,
        "present_project_classes": get_present_classes(remapped_mask, ignore_value=ignore_value),
        "project_value_counts": get_value_counts(remapped_mask),
        "present_pastis_semantic_classes": [
            int(value) for value in np.unique(semantic_mask).tolist()
        ],
        "pastis_semantic_value_counts": get_value_counts(semantic_mask),
        "n_parcel": item["n_parcel"],
        "parcel_cover": item["parcel_cover"],
    }

    candidate.update(mask_stats)
    candidate.update(quality_metrics)

    return candidate


def main() -> None:
    args = parse_args()

    pastis_root = Path(args.pastis_root)
    mapping_yaml = Path(args.mapping_yaml)
    output_dir = Path(args.output_dir)

    if not pastis_root.exists():
        raise FileNotFoundError(f"PASTIS root not found: {pastis_root}")

    if not (1 <= args.month_start <= 12 and 1 <= args.month_end <= 12):
        raise ValueError("--month-start and --month-end must be in [1, 12].")

    data_s2_dir = pastis_root / "DATA_S2"
    annotations_dir = pastis_root / "ANNOTATIONS"
    metadata_path = pastis_root / "metadata.geojson"

    if not data_s2_dir.exists():
        raise FileNotFoundError(f"DATA_S2 directory not found: {data_s2_dir}")

    if not annotations_dir.exists():
        raise FileNotFoundError(f"ANNOTATIONS directory not found: {annotations_dir}")

    mapping, ignore_value = load_mapping(mapping_yaml)
    band_reorder = parse_band_reorder(args.band_reorder)
    output_band_order = get_output_band_order(band_reorder)

    metadata_rows = load_metadata(metadata_path)
    images_dir, masks_dir = prepare_output_dirs(output_dir, overwrite=args.overwrite)

    print("=" * 100)
    print("PASTIS SINGLE-DATE CONVERSION")
    print("=" * 100)
    print(f"PASTIS root:                {pastis_root}")
    print(f"Mapping YAML:               {mapping_yaml}")
    print(f"Output dir:                 {output_dir}")
    print(f"Month range:                {args.month_start} - {args.month_end}")
    print(f"Fold filter:                {args.fold}")
    print(f"Limit samples:              {args.limit_samples}")
    print(f"Limit patches:              {args.limit_patches}")
    print(f"Ignore value:               {ignore_value}")
    print(f"Min valid pixels:           {args.min_valid_pixels}")
    print(f"Min nonbackground pixels:   {args.min_nonbackground_pixels}")
    print(f"Min interest ratio total:   {args.min_interest_ratio_total}")
    print(f"Min interest ratio valid:   {args.min_interest_ratio_valid}")
    print(f"Max white cloud ratio:      {args.max_white_cloud_ratio}")
    print(f"Max very bright ratio:      {args.max_very_bright_ratio}")
    print(f"Max bright ratio:           {args.max_bright_ratio}")
    print(f"Max cloud score:            {args.max_cloud_score}")
    print(f"Min vegetation ratio:       {args.min_vegetation_ratio}")
    print(f"NDVI threshold:             {args.ndvi_threshold}")
    print(f"Clip reflectance:           {args.clip_reflectance}")
    print(f"Clip min:                   {args.clip_min}")
    print(f"Clip max:                   {args.clip_max}")
    print(f"Select best per patch:      {args.select_best_per_patch}")
    print(f"Input band order:           {DEFAULT_PASTIS_BAND_ORDER}")
    print(f"Band reorder:               {band_reorder}")
    print(f"Output band order:          {output_band_order}")
    print(f"Expected project order:     {PROJECT_BAND_ORDER}")

    if output_band_order != PROJECT_BAND_ORDER:
        raise ValueError(
            "Output band order does not match the expected project band order. "
            f"Got {output_band_order}, expected {PROJECT_BAND_ORDER}. "
            "For PASTIS, use: --band-reorder 0,1,2,6,3,4,5,7,8,9"
        )

    print()

    converted_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    counters = {
        "patches_seen": 0,
        "patches_with_saved_sample": 0,
        "acquisitions_seen": 0,
        "acquisitions_selected_by_month": 0,
        "candidates_passed_filters": 0,
        "skipped_missing_files": 0,
        "skipped_min_valid_pixels": 0,
        "skipped_min_nonbackground_pixels": 0,
        "skipped_min_interest_ratio_total": 0,
        "skipped_min_interest_ratio_valid": 0,
        "skipped_max_white_cloud_ratio": 0,
        "skipped_max_very_bright_ratio": 0,
        "skipped_max_bright_ratio": 0,
        "skipped_max_cloud_score": 0,
        "skipped_min_vegetation_ratio": 0,
        "skipped_patches_no_candidate": 0,
    }

    sample_index = 0

    for item in metadata_rows:
        if args.limit_samples > 0 and sample_index >= args.limit_samples:
            break

        if args.limit_patches > 0 and counters["patches_seen"] >= args.limit_patches:
            break

        patch_id = int(item["patch_id"])
        fold = int(item["fold"])

        if args.fold is not None and fold != int(args.fold):
            continue

        counters["patches_seen"] += 1

        s2_path = data_s2_dir / f"S2_{patch_id}.npy"
        target_path = annotations_dir / f"TARGET_{patch_id}.npy"

        if not s2_path.exists() or not target_path.exists():
            counters["skipped_missing_files"] += 1
            continue

        s2 = np.load(s2_path)
        target = np.load(target_path)

        if s2.ndim != 4:
            raise ValueError(f"Expected S2 shape (T, C, H, W), got {s2.shape} for {s2_path}")

        if target.ndim != 3 or target.shape[0] < 1:
            raise ValueError(
                f"Expected TARGET shape (3, H, W) or similar, got {target.shape} for {target_path}"
            )

        if s2.shape[1] != 10:
            raise ValueError(f"Expected 10 Sentinel-2 bands, got shape {s2.shape} for {s2_path}")

        semantic_mask = target[0]
        remapped_mask = remap_mask(
            semantic_mask=semantic_mask,
            mapping=mapping,
            ignore_value=ignore_value,
        )

        dates_s2 = item["dates_s2"]
        patch_candidates: list[dict[str, Any]] = []

        for acquisition_index_str, acquisition_date_raw in sorted(
            dates_s2.items(),
            key=lambda pair: int(pair[0]),
        ):
            counters["acquisitions_seen"] += 1

            acquisition_index = int(acquisition_index_str)
            acquisition_date = int(acquisition_date_raw)

            year, month, day = parse_yyyymmdd(acquisition_date)

            if not month_is_allowed(month, args.month_start, args.month_end):
                continue

            counters["acquisitions_selected_by_month"] += 1

            if acquisition_index < 0 or acquisition_index >= s2.shape[0]:
                raise IndexError(
                    f"Acquisition index {acquisition_index} out of range for patch {patch_id} "
                    f"with S2 shape {s2.shape}"
                )

            image = s2[acquisition_index].astype(np.float32, copy=False)

            if band_reorder is not None:
                image = image[band_reorder, :, :]

            image = maybe_clip_reflectance(
                image=image,
                enabled=args.clip_reflectance,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
            )

            image = np.ascontiguousarray(image, dtype=np.float32)

            candidate = build_candidate(
                image=image,
                remapped_mask=remapped_mask,
                semantic_mask=semantic_mask,
                item=item,
                acquisition_index=acquisition_index,
                acquisition_date=acquisition_date,
                year=year,
                month=month,
                day=day,
                s2_path=s2_path,
                target_path=target_path,
                ignore_value=ignore_value,
                ndvi_threshold=args.ndvi_threshold,
            )

            passed_filters, skip_reason = passes_filters(candidate, args)

            quality_row = {
                "patch_id": candidate["patch_id"],
                "fold": candidate["fold"],
                "tile": candidate["tile"],
                "acquisition_index": candidate["acquisition_index"],
                "acquisition_date": candidate["acquisition_date"],
                "year": candidate["year"],
                "month": candidate["month"],
                "day": candidate["day"],
                "passed_filters": passed_filters,
                "skip_reason": skip_reason,
                "interest_ratio_total": candidate["interest_ratio_total"],
                "interest_ratio_valid": candidate["interest_ratio_valid"],
                "valid_ratio_total": candidate["valid_ratio_total"],
                "ignore_ratio_total": candidate["ignore_ratio_total"],
                "bright_ratio": candidate["bright_ratio"],
                "very_bright_ratio": candidate["very_bright_ratio"],
                "white_cloud_ratio": candidate["white_cloud_ratio"],
                "dark_ratio": candidate["dark_ratio"],
                "rgb_std": candidate["rgb_std"],
                "rgb_mean": candidate["rgb_mean"],
                "ndvi_mean": candidate["ndvi_mean"],
                "vegetation_ratio": candidate["vegetation_ratio"],
                "cloud_score": candidate["cloud_score"],
                "present_project_classes": json.dumps(candidate["present_project_classes"]),
            }
            quality_rows.append(quality_row)

            if not passed_filters:
                counter_key = f"skipped_{skip_reason}"
                counters[counter_key] = counters.get(counter_key, 0) + 1
                continue

            counters["candidates_passed_filters"] += 1

            if args.select_best_per_patch:
                patch_candidates.append(candidate)
            else:
                row = save_candidate(
                    candidate=candidate,
                    sample_index=sample_index,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    band_reorder=band_reorder,
                    output_band_order=output_band_order,
                )
                converted_rows.append(row)

                sample_index += 1

                if sample_index % 100 == 0:
                    print(f"[CONVERT] saved samples: {sample_index}")

                if args.limit_samples > 0 and sample_index >= args.limit_samples:
                    break

        if args.select_best_per_patch:
            if patch_candidates:
                best_candidate = sorted(
                    patch_candidates,
                    key=lambda candidate: (
                        candidate["cloud_score"],
                        candidate["white_cloud_ratio"],
                        -candidate["vegetation_ratio"],
                    ),
                )[0]

                row = save_candidate(
                    candidate=best_candidate,
                    sample_index=sample_index,
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    band_reorder=band_reorder,
                    output_band_order=output_band_order,
                )
                converted_rows.append(row)

                sample_index += 1
                counters["patches_with_saved_sample"] += 1

                if sample_index % 100 == 0:
                    print(f"[CONVERT] saved samples: {sample_index}")
            else:
                counters["skipped_patches_no_candidate"] += 1
        else:
            counters["patches_with_saved_sample"] = sample_index

    metadata_csv = output_dir / "metadata.csv"
    quality_csv = output_dir / "quality_report.csv"

    save_metadata_csv(metadata_csv, converted_rows)
    save_quality_report(quality_csv, quality_rows)

    save_dataset_info(
        output_dir=output_dir,
        pastis_root=pastis_root,
        mapping_yaml=mapping_yaml,
        args=args,
        ignore_value=ignore_value,
        band_reorder=band_reorder,
        output_band_order=output_band_order,
        num_samples=sample_index,
        counters=counters,
    )

    print()
    print("=" * 100)
    print("PASTIS SINGLE-DATE CONVERSION SUMMARY")
    print("=" * 100)
    print(f"Patches seen:                         {counters['patches_seen']}")
    print(f"Patches with saved sample:            {counters['patches_with_saved_sample']}")
    print(f"Acquisitions seen:                    {counters['acquisitions_seen']}")
    print(f"Acquisitions selected by month:       {counters['acquisitions_selected_by_month']}")
    print(f"Candidates passed filters:            {counters['candidates_passed_filters']}")
    print(f"Skipped missing files:                {counters['skipped_missing_files']}")
    print(f"Skipped patches no candidate:         {counters['skipped_patches_no_candidate']}")
    print(f"Saved samples:                        {sample_index}")
    print(f"Images dir:                           {images_dir}")
    print(f"Masks dir:                            {masks_dir}")
    print(f"Metadata CSV:                         {metadata_csv}")
    print(f"Quality report:                       {quality_csv}")
    print(f"Dataset info:                         {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()