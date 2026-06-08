from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.baselines.random_forest.features import (
    FeatureMode,
    build_feature_patch,
    count_output_channels,
)
from src.baselines.random_forest.sampling import (
    StratifiedPixelSampler,
    extract_pixel_samples,
)
from src.training.dataset_manifest import ManifestSegmentationDataset


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None

    value = value.strip()
    if not value:
        return None

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest baseline for Sentinel-2 / EuroCrops segmentation."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Path to the dataset manifest CSV.",
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output directory where the trained model and metadata will be saved.",
    )

    parser.add_argument(
        "--feature-mode",
        choices=["spectral", "spatial"],
        default="spectral",
        help="Feature mode used by Random Forest.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="Number of segmentation classes.",
    )

    parser.add_argument(
        "--band-indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated band indices to load from each patch.",
    )

    parser.add_argument(
        "--normalize-mode",
        choices=["none", "scale_10000", "standardize"],
        default="scale_10000",
        help="Normalization mode passed to ManifestSegmentationDataset.",
    )

    parser.add_argument(
        "--scale-divisor",
        type=float,
        default=10000.0,
        help="Divisor used when normalize_mode='scale_10000'.",
    )

    parser.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Limit number of train patches. Use 0 for all train patches.",
    )

    parser.add_argument(
        "--target-pixels-per-class",
        type=int,
        default=100_000,
        help="Target number of selected training pixels per class.",
    )

    parser.add_argument(
        "--max-pixels-per-class-per-patch",
        type=int,
        default=64,
        help="Maximum pixels selected from one class inside one patch.",
    )

    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Include class 0 in the training sample.",
    )

    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Ignore index in masks. Use -1 when there is no ignore index.",
    )

    parser.add_argument(
        "--spatial-window-sizes",
        type=str,
        default="3,5",
        help="Comma-separated local window sizes for spatial features.",
    )

    parser.add_argument(
        "--spatial-band-indices",
        type=str,
        default=None,
        help=(
            "Comma-separated band indices used for spatial context. "
            "If omitted, all selected input bands are used."
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=24,
        help="Maximum depth of each tree. Use 0 for unlimited depth.",
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples required at a leaf node.",
    )

    parser.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help=(
            "Fraction of the selected training table used to train each tree. "
            "Example: 0.8 means each tree uses 80 percent of the selected pixels. "
            "Requires bootstrap=True, which is enabled in this script."
        ),
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs used by scikit-learn.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def _safe_class_counts(y: np.ndarray, num_classes: int) -> Dict[int, int]:
    counts = np.bincount(y.astype(np.int64, copy=False), minlength=num_classes)
    return {int(class_id): int(counts[class_id]) for class_id in range(num_classes)}


def _remove_invalid_rows(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(X).all(axis=1)

    if valid.all():
        return X, y

    return X[valid], y[valid]


def _validate_max_samples(max_samples: float | None) -> float | None:
    if max_samples is None:
        return None

    max_samples = float(max_samples)

    if not (0.0 < max_samples <= 1.0):
        raise ValueError(
            f"max_samples must be in the interval (0, 1], got {max_samples}."
        )

    return max_samples


def build_training_table(
    dataset: ManifestSegmentationDataset,
    *,
    feature_mode: FeatureMode,
    num_classes: int,
    target_pixels_per_class: int,
    max_pixels_per_class_per_patch: int,
    include_background: bool,
    ignore_index: int | None,
    spatial_window_sizes: Sequence[int],
    spatial_band_indices: Sequence[int] | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    sampler = StratifiedPixelSampler(
        num_classes=num_classes,
        target_pixels_per_class=target_pixels_per_class,
        max_pixels_per_class_per_patch=max_pixels_per_class_per_patch,
        seed=seed,
        ignore_index=ignore_index,
        include_background=include_background,
    )

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    patches_used = 0
    patches_with_selected_pixels = 0

    print("[TRAIN] Target pixels per class:")
    print(json.dumps(sampler.target_summary(), indent=2))

    for idx in range(len(dataset)):
        x, y = dataset[idx]

        feature_patch = build_feature_patch(
            x,
            feature_mode=feature_mode,
            spatial_window_sizes=spatial_window_sizes,
            spatial_band_indices=spatial_band_indices,
        )

        selected_indices, _ = sampler.sample_patch_indices(y)

        patches_used += 1

        if selected_indices.size == 0:
            if sampler.is_complete():
                break
            continue

        X_patch, y_patch = extract_pixel_samples(
            feature_patch,
            y,
            selected_indices,
        )

        X_patch, y_patch = _remove_invalid_rows(X_patch, y_patch)

        if len(y_patch) == 0:
            if sampler.is_complete():
                break
            continue

        X_parts.append(X_patch.astype(np.float32, copy=False))
        y_parts.append(y_patch.astype(np.int64, copy=False))

        patches_with_selected_pixels += 1

        if patches_used % 500 == 0:
            print(
                "[TRAIN] "
                f"patches_used={patches_used}/{len(dataset)} | "
                f"patches_with_pixels={patches_with_selected_pixels} | "
                f"selected={sampler.selected_summary()}"
            )

        if sampler.is_complete():
            print("[TRAIN] Sampling target reached.")
            break

    if not X_parts or not y_parts:
        raise RuntimeError(
            "No training pixels were selected. Check manifest paths, masks, "
            "num_classes, include_background, and ignore_index."
        )

    X_train = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y_train = np.concatenate(y_parts, axis=0).astype(np.int64, copy=False)

    metadata = {
        "patches_used": patches_used,
        "patches_with_selected_pixels": patches_with_selected_pixels,
        "selected_per_class": sampler.selected_summary(),
        "target_per_class": sampler.target_summary(),
        "class_counts_in_training_table": _safe_class_counts(y_train, num_classes),
        "num_training_pixels": int(len(y_train)),
        "num_features": int(X_train.shape[1]),
    }

    return X_train, y_train, metadata


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_samples: float | None,
    n_jobs: int,
    seed: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=max_samples,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=1,
    )

    model.fit(X_train, y_train)
    return model


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    band_indices = parse_int_list(args.band_indices)
    spatial_window_sizes = parse_int_list(args.spatial_window_sizes)
    spatial_band_indices = parse_int_list(args.spatial_band_indices)

    if spatial_window_sizes is None:
        spatial_window_sizes = [3, 5]

    ignore_index = None if args.ignore_index < 0 else int(args.ignore_index)
    max_depth = None if args.max_depth <= 0 else int(args.max_depth)
    max_samples = _validate_max_samples(args.max_samples)

    print("[INFO] Loading train dataset...")
    train_dataset = ManifestSegmentationDataset(
        manifest_csv=manifest_path,
        split="train",
        limit=args.train_limit,
        shuffle=True,
        seed=args.seed,
        augment=False,
        band_indices=band_indices,
        normalize_mode=args.normalize_mode,
        scale_divisor=args.scale_divisor,
    )

    dataset_description = train_dataset.describe()

    input_channels = int(dataset_description["num_channels"])
    output_channels = count_output_channels(
        input_channels=input_channels,
        feature_mode=args.feature_mode,
        spatial_window_sizes=spatial_window_sizes,
        spatial_band_indices=spatial_band_indices,
    )

    print("[INFO] Dataset description:")
    print(json.dumps(dataset_description, indent=2))

    print("[INFO] Random Forest feature configuration:")
    print(
        json.dumps(
            {
                "feature_mode": args.feature_mode,
                "input_channels": input_channels,
                "output_feature_channels": output_channels,
                "band_indices": band_indices,
                "spatial_window_sizes": spatial_window_sizes,
                "spatial_band_indices": spatial_band_indices,
            },
            indent=2,
        )
    )

    print("[INFO] Random Forest model configuration:")
    print(
        json.dumps(
            {
                "n_estimators": args.n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "bootstrap": True,
                "max_samples": max_samples,
                "class_weight": "balanced_subsample",
                "n_jobs": args.n_jobs,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    print("[INFO] Building stratified training table...")
    X_train, y_train, sampling_metadata = build_training_table(
        train_dataset,
        feature_mode=args.feature_mode,
        num_classes=args.num_classes,
        target_pixels_per_class=args.target_pixels_per_class,
        max_pixels_per_class_per_patch=args.max_pixels_per_class_per_patch,
        include_background=args.include_background,
        ignore_index=ignore_index,
        spatial_window_sizes=spatial_window_sizes,
        spatial_band_indices=spatial_band_indices,
        seed=args.seed,
    )

    print("[INFO] Training table:")
    print(
        json.dumps(
            {
                "X_train_shape": list(X_train.shape),
                "y_train_shape": list(y_train.shape),
                "class_counts": sampling_metadata["class_counts_in_training_table"],
            },
            indent=2,
        )
    )

    print("[INFO] Training Random Forest...")
    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_samples=max_samples,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    model_path = out_dir / f"random_forest_{args.feature_mode}.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "feature_mode": args.feature_mode,
        "num_classes": args.num_classes,
        "band_indices": band_indices,
        "normalize_mode": args.normalize_mode,
        "scale_divisor": args.scale_divisor,
        "train_limit": args.train_limit,
        "target_pixels_per_class": args.target_pixels_per_class,
        "max_pixels_per_class_per_patch": args.max_pixels_per_class_per_patch,
        "include_background": args.include_background,
        "ignore_index": ignore_index,
        "spatial_window_sizes": spatial_window_sizes,
        "spatial_band_indices": spatial_band_indices,
        "n_estimators": args.n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "bootstrap": True,
        "max_samples": max_samples,
        "class_weight": "balanced_subsample",
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "dataset_description": dataset_description,
        "sampling": sampling_metadata,
    }

    metadata_path = out_dir / f"training_metadata_{args.feature_mode}.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()