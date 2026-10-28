from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
import yaml
from scipy.io import loadmat

from sentinel_crop_seg.core.external.output_contract import prepare_conversion_output


DACIA5_12BAND_ORDER = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]

PROJECT_BANDS = [
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
        description="Generate a Dacia5 external test dataset as .npy patches."
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Path to Dataset_DACIA5/Images_Sentinel2_12bands_GeoTIFF",
    )
    parser.add_argument(
        "--masks-root",
        type=Path,
        required=True,
        help="Path to Dataset_DACIA5/Masks_and_legend",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        required=True,
        help="Path to config/dacia5/label_mapping.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing images/, masks/ and metadata.csv",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=["2020", "2021", "2022", "2023", "2024"],
        help="Years to include, e.g. 2020 2021 2024",
    )
    parser.add_argument(
        "--max-scenes-per-year",
        type=int,
        default=0,
        help="Limit the number of scenes per year. Use 0 for all.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in pixels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Patch stride in pixels.",
    )
    parser.add_argument(
        "--ignore-value",
        type=int,
        default=255,
        help="Ignore value used for unmapped labels.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting sample index for saved patch names.",
    )
    parser.add_argument(
        "--drop-all-ignore",
        action="store_true",
        help="Skip patches where all pixels are ignore_value.",
    )
    parser.add_argument(
        "--drop-all-background",
        action="store_true",
        help="Skip patches where all valid pixels are background.",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=0,
        help="Minimum number of valid pixels required to keep a patch.",
    )
    parser.add_argument(
        "--min-nonbackground-pixels",
        type=int,
        default=1000,
        help="Minimum number of non-background valid pixels required to keep a patch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an output directory that already contains results.",
    )
    return parser.parse_args()


def load_mapping(mapping_path: Path) -> Dict[int, int | None]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    raw_mapping = data["dacia5"]["remap_to_project_class_id"]
    mapping: Dict[int, int | None] = {}

    for key, value in raw_mapping.items():
        mapping[int(key)] = None if value is None else int(value)

    return mapping


def compute_project_band_indices() -> List[int]:
    index_by_band = {band_name: idx for idx, band_name in enumerate(DACIA5_12BAND_ORDER)}
    return [index_by_band[band_name] for band_name in PROJECT_BANDS]


def load_project_image(image_path: Path) -> np.ndarray:
    """Read the project's ten bands, keeping the source reflectance scale.

    This used to divide by 10,000 before saving. The project's own generator
    writes raw reflectance, and evaluation applies the checkpoint's
    ``normalize_mode``, so a pre-divided patch evaluated with a
    ``scale_10000`` checkpoint was scaled twice and came out 10,000x too
    small -- with no error, just wrong numbers. Scaling is now left entirely
    to the consumer, exactly as for a generated patch.
    """
    band_indices = compute_project_band_indices()

    with rasterio.open(image_path) as src:
        stack = src.read()

    return stack[band_indices].astype(np.float32)


def load_raw_mask(mask_path: Path) -> np.ndarray:
    data = loadmat(mask_path)
    if "mask" not in data:
        raise KeyError(f"Expected key 'mask' in MAT file: {mask_path}")
    return data["mask"].astype(np.uint8)


def remap_mask(mask: np.ndarray, mapping: Dict[int, int | None], ignore_value: int) -> np.ndarray:
    remapped = np.full(mask.shape, ignore_value, dtype=np.uint8)

    for source_label, target_label in mapping.items():
        if target_label is None:
            remapped[mask == source_label] = ignore_value
        else:
            remapped[mask == source_label] = np.uint8(target_label)

    return remapped


def build_axis_offsets(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        raise ValueError(f"Axis length {length} is smaller than patch size {patch_size}.")

    offsets = list(range(0, length - patch_size + 1, stride))
    last_offset = length - patch_size

    if not offsets:
        offsets = [0]

    if offsets[-1] != last_offset:
        offsets.append(last_offset)

    return offsets


def iter_patch_offsets(height: int, width: int, patch_size: int, stride: int):
    row_offsets = build_axis_offsets(height, patch_size, stride)
    col_offsets = build_axis_offsets(width, patch_size, stride)

    for row_off in row_offsets:
        for col_off in col_offsets:
            yield row_off, col_off


def extract_patch(array: np.ndarray, row_off: int, col_off: int, patch_size: int) -> np.ndarray:
    if array.ndim == 3:
        return array[:, row_off:row_off + patch_size, col_off:col_off + patch_size]
    if array.ndim == 2:
        return array[row_off:row_off + patch_size, col_off:col_off + patch_size]
    raise ValueError(f"Unsupported array ndim={array.ndim}")


def collect_patch_stats(mask_patch: np.ndarray, ignore_value: int) -> Dict[str, Any]:
    total_pixels = int(mask_patch.size)
    ignore_pixels = int(np.sum(mask_patch == ignore_value))
    valid_pixels = total_pixels - ignore_pixels
    background_pixels = int(np.sum(mask_patch == 0))
    nonbackground_valid_pixels = int(np.sum((mask_patch != ignore_value) & (mask_patch != 0)))

    unique_values, counts = np.unique(mask_patch, return_counts=True)
    value_counts = {int(v): int(c) for v, c in zip(unique_values.tolist(), counts.tolist())}

    present_valid_classes = sorted(
        int(v) for v in unique_values.tolist()
        if int(v) not in (0, ignore_value)
    )

    return {
        "total_pixels": total_pixels,
        "ignore_pixels": ignore_pixels,
        "valid_pixels": valid_pixels,
        "background_pixels": background_pixels,
        "nonbackground_valid_pixels": nonbackground_valid_pixels,
        "present_valid_classes": present_valid_classes,
        "value_counts": value_counts,
    }


def should_skip_patch(
    mask_patch: np.ndarray,
    ignore_value: int,
    drop_all_ignore: bool,
    drop_all_background: bool,
    min_valid_pixels: int,
    min_nonbackground_pixels: int,
) -> bool:
    if drop_all_ignore and np.all(mask_patch == ignore_value):
        return True

    valid = mask_patch != ignore_value
    valid_pixels = int(np.sum(valid))
    nonbackground_valid_pixels = int(np.sum(valid & (mask_patch != 0)))

    if min_valid_pixels > 0 and valid_pixels < min_valid_pixels:
        return True

    if min_nonbackground_pixels > 0 and nonbackground_valid_pixels < min_nonbackground_pixels:
        return True

    if drop_all_background and valid_pixels > 0:
        valid_values = mask_patch[valid]
        if np.all(valid_values == 0):
            return True

    return False


def ensure_output_dirs(output_dir: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    """Prepare the output directories under the shared conversion contract.

    ``exist_ok=True`` plus ``np.save`` silently replaced an earlier
    conversion, so repeating a run with different options destroyed the
    results it was meant to be compared against, and an authorized overwrite
    then left the earlier run's patches beside the new metadata. Both
    converters now delegate to one rule.
    """
    return prepare_conversion_output(output_dir, overwrite=overwrite)


def infer_year_from_path(image_path: Path) -> str:
    match = re.search(r"(\d{4})", image_path.parent.name)
    if match is None:
        raise ValueError(f"Could not infer year from parent folder: {image_path.parent}")
    return match.group(1)


def infer_date_from_filename(image_path: Path) -> str:
    match = re.search(r"(\d{8})", image_path.stem)
    if match is None:
        return image_path.stem
    raw = match.group(1)
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def collect_image_paths(images_root: Path, years: List[str], max_scenes_per_year: int) -> List[Path]:
    all_paths: List[Path] = []

    for year in years:
        year_dir = images_root / f"Sentinel2_{year}"
        if not year_dir.exists():
            print(f"[WARN] Missing year directory: {year_dir}")
            continue

        tif_paths = sorted(year_dir.glob("*.tif"))
        if max_scenes_per_year > 0:
            tif_paths = tif_paths[:max_scenes_per_year]

        all_paths.extend(tif_paths)

    return all_paths


def get_mask_path_for_year(masks_root: Path, year: str) -> Path:
    return masks_root / f"Masks_{year}" / f"Labels_mask_{year}.mat"


def write_metadata_csv(metadata_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_name",
        "year",
        "scene_date",
        "source_image",
        "source_mask",
        "row_off",
        "col_off",
        "patch_size",
        "total_pixels",
        "valid_pixels",
        "ignore_pixels",
        "background_pixels",
        "nonbackground_valid_pixels",
        "present_valid_classes",
        "value_counts",
    ]

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    images_root = args.images_root.resolve()
    masks_root = args.masks_root.resolve()
    mapping_path = args.mapping_path.resolve()
    output_dir = args.output_dir.resolve()

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks root not found: {masks_root}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mapping = load_mapping(mapping_path)
    images_dir, masks_dir = ensure_output_dirs(output_dir, overwrite=args.overwrite)

    image_paths = collect_image_paths(
        images_root=images_root,
        years=[str(y) for y in args.years],
        max_scenes_per_year=int(args.max_scenes_per_year),
    )

    if not image_paths:
        raise RuntimeError("No image scenes found for the requested years.")

    print("=" * 100)
    print("DACIA5 TEST DATASET GENERATION START")
    print("=" * 100)
    print(f"Images root:              {images_root}")
    print(f"Masks root:               {masks_root}")
    print(f"Output dir:               {output_dir}")
    print(f"Years:                    {args.years}")
    print(f"Scenes selected:          {len(image_paths)}")
    print(f"Patch size:               {args.patch_size}")
    print(f"Stride:                   {args.stride}")
    print(f"Ignore value:             {args.ignore_value}")
    print(f"Min valid pixels:         {args.min_valid_pixels}")
    print(f"Min nonbackground pixels: {args.min_nonbackground_pixels}")
    print()

    sample_index = int(args.start_index)
    metadata_rows: List[Dict[str, Any]] = []

    remapped_masks_by_year: Dict[str, np.ndarray] = {}

    total_candidate_windows = 0
    total_saved_patches = 0
    total_skipped_patches = 0

    for scene_idx, image_path in enumerate(image_paths, start=1):
        year = infer_year_from_path(image_path)
        scene_date = infer_date_from_filename(image_path)
        mask_path = get_mask_path_for_year(masks_root, year)

        if not mask_path.exists():
            print(f"[WARN] Missing mask for year {year}: {mask_path}")
            continue

        if year not in remapped_masks_by_year:
            raw_mask = load_raw_mask(mask_path)
            remapped_masks_by_year[year] = remap_mask(
                raw_mask,
                mapping=mapping,
                ignore_value=args.ignore_value,
            )

        remapped_mask = remapped_masks_by_year[year]
        image = load_project_image(image_path)

        if image.shape[1:] != remapped_mask.shape:
            raise RuntimeError(
                f"Image/mask shape mismatch for scene {image_path.name}: "
                f"image spatial shape={image.shape[1:]}, mask shape={remapped_mask.shape}"
            )

        _, height, width = image.shape
        row_offsets = build_axis_offsets(height, args.patch_size, args.stride)
        col_offsets = build_axis_offsets(width, args.patch_size, args.stride)

        scene_saved = 0
        scene_skipped = 0
        scene_candidates = 0

        print(
            f"[SCENE {scene_idx:03d}/{len(image_paths)}] "
            f"{image_path.name} | year={year} | date={scene_date} | "
            f"row_offsets={row_offsets} | col_offsets={col_offsets}"
        )

        for row_off, col_off in iter_patch_offsets(height, width, args.patch_size, args.stride):
            scene_candidates += 1
            total_candidate_windows += 1

            image_patch = extract_patch(image, row_off=row_off, col_off=col_off, patch_size=args.patch_size)
            mask_patch = extract_patch(remapped_mask, row_off=row_off, col_off=col_off, patch_size=args.patch_size)

            if should_skip_patch(
                mask_patch=mask_patch,
                ignore_value=args.ignore_value,
                drop_all_ignore=args.drop_all_ignore,
                drop_all_background=args.drop_all_background,
                min_valid_pixels=args.min_valid_pixels,
                min_nonbackground_pixels=args.min_nonbackground_pixels,
            ):
                scene_skipped += 1
                total_skipped_patches += 1
                continue

            sample_name = f"sample_{sample_index:06d}.npy"
            np.save(images_dir / sample_name, image_patch)
            np.save(masks_dir / sample_name, mask_patch)

            stats = collect_patch_stats(mask_patch, ignore_value=args.ignore_value)
            metadata_rows.append({
                "sample_name": sample_name,
                "year": year,
                "scene_date": scene_date,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "row_off": row_off,
                "col_off": col_off,
                "patch_size": args.patch_size,
                "total_pixels": stats["total_pixels"],
                "valid_pixels": stats["valid_pixels"],
                "ignore_pixels": stats["ignore_pixels"],
                "background_pixels": stats["background_pixels"],
                "nonbackground_valid_pixels": stats["nonbackground_valid_pixels"],
                "present_valid_classes": str(stats["present_valid_classes"]),
                "value_counts": str(stats["value_counts"]),
            })

            sample_index += 1
            scene_saved += 1
            total_saved_patches += 1

        print(
            f"  candidates={scene_candidates} | saved={scene_saved} | skipped={scene_skipped}"
        )

    metadata_path = output_dir / "metadata.csv"
    write_metadata_csv(metadata_path, metadata_rows)

    # State the contract once, machine-readably. The scale is the field that
    # matters: a consumer that normalizes again on top of an already-scaled
    # patch gets silently wrong values, so what was done here must not be
    # guessable only from the array contents.
    (output_dir / "conversion_contract.json").write_text(
        json.dumps({
            "source_dataset": "DACIA5",
            "output_band_order": list(PROJECT_BANDS),
            "source_band_order": list(DACIA5_12BAND_ORDER),
            "reflectance_scale": "source_native_unscaled",
            "reflectance_note": (
                "Values are the source reflectance, not divided by 10000. "
                "Apply the consuming checkpoint's normalize_mode exactly "
                "once; scale_10000 is correct for a checkpoint trained that "
                "way."
            ),
            "ignore_value": ignore_value,
            "ignore_note": (
                "Unmapped source labels are written as the ignore value, not "
                "as background, so those pixels are excluded from metrics "
                "rather than counted as errors."
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("DACIA5 TEST DATASET GENERATION SUMMARY")
    print("=" * 100)
    print(f"Total scenes processed:   {len(image_paths)}")
    print(f"Total candidate windows:  {total_candidate_windows}")
    print(f"Total saved patches:      {total_saved_patches}")
    print(f"Total skipped patches:    {total_skipped_patches}")
    print(f"Images dir:               {images_dir}")
    print(f"Masks dir:                {masks_dir}")
    print(f"Metadata CSV:             {metadata_path}")


if __name__ == "__main__":
    main()