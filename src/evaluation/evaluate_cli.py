from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    DEFAULT_CLASS_NAMES,
    RandomForestSegmentationPredictor,
    evaluate_segmentation_predictor,
    load_unet_checkpoint,
    read_allowed_sample_names_by_month,
    validate_month_filter_args,
)
from src.training.dataset import SegmentationPatchDataset
from src.training.dataset_manifest import ManifestSegmentationDataset


def parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation entry point for crop segmentation experiments. "
            "Supports U-Net and Random Forest evaluation on manifest-based or root-folder datasets."
        )
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=["unet", "random_forest"],
        help="Model family to evaluate.",
    )

    parser.add_argument(
        "--data-source",
        required=True,
        choices=["manifest", "root"],
        help="Evaluation data source. Supported values: manifest, root.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help=(
            "Path to the trained model checkpoint. "
            "For U-Net this is a .pt file. For Random Forest this is a .joblib file."
        ),
    )

    parser.add_argument(
        "--manifest-csv",
        default=None,
        type=str,
        help="Path to the manifest CSV. Required when --data-source manifest.",
    )

    parser.add_argument(
        "--dataset-root",
        default=None,
        type=str,
        help=(
            "Path to a dataset root containing images/masks or images_tif/masks_tif. "
            "Required when --data-source root."
        ),
    )

    parser.add_argument(
        "--sample-format",
        default="npy",
        choices=["npy", "geotiff"],
        help="Patch format used when --data-source root.",
    )

    parser.add_argument(
        "--metadata-csv",
        default=None,
        type=str,
        help=(
            "Optional metadata CSV used for filtering root datasets. "
            "Expected columns: scene_date and either sample_name or sample_id."
        ),
    )

    parser.add_argument(
        "--month-start",
        default=None,
        type=int,
        help="Optional first acquisition month to keep, for example 4 for April.",
    )

    parser.add_argument(
        "--month-end",
        default=None,
        type=int,
        help="Optional last acquisition month to keep, for example 9 for September.",
    )

    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help=(
            "Manifest split to evaluate, for example train, val, or test. "
            "Used only with --data-source manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where evaluation reports will be saved.",
    )

    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Optional maximum number of samples to evaluate. Use 0 for all samples.",
    )

    parser.add_argument(
        "--num-classes",
        default=11,
        type=int,
        help="Number of classes. Used for Random Forest; U-Net uses the value stored in the checkpoint.",
    )

    parser.add_argument(
        "--ignore-index",
        default=255,
        type=int,
        help="Ground-truth label value ignored during metric computation.",
    )

    parser.add_argument(
        "--ignore-background",
        action="store_true",
        help="Exclude target class 0 from metric computation.",
    )

    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device.",
    )

    parser.add_argument(
        "--normalize-mode-override",
        default=None,
        type=str,
        help=(
            "Override normalization mode. For U-Net, defaults to the checkpoint config. "
            "For Random Forest, defaults to scale_10000."
        ),
    )

    parser.add_argument(
        "--scale-divisor-override",
        default=None,
        type=float,
        help=(
            "Override scale divisor. For U-Net, defaults to the checkpoint config. "
            "For Random Forest, defaults to 10000.0."
        ),
    )

    parser.add_argument(
        "--band-indices",
        default=None,
        type=str,
        help=(
            "Optional comma-separated band indices. "
            "For U-Net, defaults to checkpoint band_indices. "
            "For Random Forest, defaults to all bands."
        ),
    )

    parser.add_argument(
        "--feature-mode",
        default=None,
        choices=["spectral", "spatial"],
        help="Random Forest feature mode. Required when --model-type random_forest.",
    )

    parser.add_argument(
        "--predict-batch-size",
        default=32768,
        type=int,
        help="Number of pixels predicted at once for Random Forest.",
    )

    parser.add_argument(
        "--spatial-window-sizes",
        default="3,5",
        type=str,
        help="Comma-separated spatial window sizes for Random Forest spatial features.",
    )

    parser.add_argument(
        "--spatial-band-indices",
        default=None,
        type=str,
        help="Optional comma-separated band indices used for Random Forest spatial features.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_unet_manifest_dataset(
    *,
    manifest_csv: Path,
    split: str,
    limit: int,
    checkpoint_config,
    band_indices_override: tuple[int, ...] | None,
    normalize_mode_override: str | None,
    scale_divisor_override: float | None,
) -> ManifestSegmentationDataset:
    band_indices = checkpoint_config.band_indices
    normalize_mode = checkpoint_config.normalize_mode
    scale_divisor = checkpoint_config.scale_divisor

    if band_indices_override is not None:
        band_indices = band_indices_override

    if normalize_mode_override is not None:
        normalize_mode = normalize_mode_override

    if scale_divisor_override is not None:
        scale_divisor = float(scale_divisor_override)

    return ManifestSegmentationDataset(
        manifest_csv=manifest_csv,
        split=split,
        limit=limit,
        shuffle=False,
        augment=False,
        band_indices=band_indices,
        normalize_mode=normalize_mode,
        scale_divisor=scale_divisor,
    )


def build_random_forest_manifest_dataset(
    *,
    manifest_csv: Path,
    split: str,
    limit: int,
    band_indices: tuple[int, ...] | None,
    normalize_mode_override: str | None,
    scale_divisor_override: float | None,
) -> ManifestSegmentationDataset:
    normalize_mode = normalize_mode_override if normalize_mode_override is not None else "scale_10000"
    scale_divisor = scale_divisor_override if scale_divisor_override is not None else 10000.0

    return ManifestSegmentationDataset(
        manifest_csv=manifest_csv,
        split=split,
        limit=limit,
        shuffle=False,
        augment=False,
        band_indices=band_indices,
        normalize_mode=normalize_mode,
        scale_divisor=float(scale_divisor),
    )


def build_root_dataset(
    *,
    dataset_root: Path,
    limit: int,
    band_indices: tuple[int, ...] | None,
    normalize_mode: str,
    scale_divisor: float,
    sample_format: str,
    allowed_sample_names: set[str] | None = None,
) -> SegmentationPatchDataset:
    return SegmentationPatchDataset(
        dataset_root,
        limit=limit,
        shuffle=False,
        augment=False,
        band_indices=band_indices,
        normalize_mode=normalize_mode,
        scale_divisor=scale_divisor,
        sample_format=sample_format,
        allowed_sample_names=allowed_sample_names,
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv is not None else None
    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else None
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv is not None else None
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint/model file not found: {checkpoint_path}")

    if args.data_source == "manifest":
        if manifest_csv is None:
            raise ValueError("--manifest-csv is required when --data-source manifest.")

        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    elif args.data_source == "root":
        if dataset_root is None:
            raise ValueError("--dataset-root is required when --data-source root.")

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")

    month_filter_enabled = validate_month_filter_args(
        metadata_csv=metadata_csv,
        month_start=args.month_start,
        month_end=args.month_end,
    )

    if metadata_csv is not None and not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    if month_filter_enabled and args.data_source != "root":
        raise ValueError("Month filtering is currently supported only with --data-source root.")

    allowed_sample_names = None

    if month_filter_enabled:
        allowed_sample_names = read_allowed_sample_names_by_month(
            metadata_csv=metadata_csv,
            month_start=int(args.month_start),
            month_end=int(args.month_end),
        )

        if not allowed_sample_names:
            raise ValueError(
                "Month filtering produced zero allowed samples. "
                "Check --metadata-csv, --month-start, and --month-end."
            )

    device = resolve_device(args.device)
    band_indices_override = parse_int_list(args.band_indices)

    summary_metadata = {
        "model_type": args.model_type,
        "data_source": args.data_source,
        "checkpoint": str(checkpoint_path),
        "manifest_csv": str(manifest_csv) if manifest_csv is not None else None,
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "sample_format": args.sample_format if args.data_source == "root" else None,
        "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
        "month_filter_enabled": month_filter_enabled,
        "month_start": args.month_start,
        "month_end": args.month_end,
        "allowed_samples_after_month_filter": (
            len(allowed_sample_names) if allowed_sample_names is not None else None
        ),
        "split": args.split if args.data_source == "manifest" else None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
    }

    if args.model_type == "unet":
        loaded_checkpoint = load_unet_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        normalize_mode = (
            args.normalize_mode_override
            if args.normalize_mode_override is not None
            else loaded_checkpoint.config.normalize_mode
        )
        scale_divisor = (
            args.scale_divisor_override
            if args.scale_divisor_override is not None
            else loaded_checkpoint.config.scale_divisor
        )
        band_indices = (
            band_indices_override
            if band_indices_override is not None
            else loaded_checkpoint.config.band_indices
        )

        if args.data_source == "manifest":
            dataset = build_unet_manifest_dataset(
                manifest_csv=manifest_csv,
                split=args.split,
                limit=args.limit,
                checkpoint_config=loaded_checkpoint.config,
                band_indices_override=band_indices_override,
                normalize_mode_override=args.normalize_mode_override,
                scale_divisor_override=args.scale_divisor_override,
            )
        else:
            dataset = build_root_dataset(
                dataset_root=dataset_root,
                limit=args.limit,
                band_indices=band_indices,
                normalize_mode=normalize_mode,
                scale_divisor=float(scale_divisor),
                sample_format=args.sample_format,
                allowed_sample_names=allowed_sample_names,
            )

        predictor = loaded_checkpoint.model
        num_classes = loaded_checkpoint.config.num_classes

        summary_metadata.update(
            {
                "checkpoint_config": asdict(loaded_checkpoint.config),
                "normalize_mode": normalize_mode,
                "scale_divisor": scale_divisor,
                "band_indices": band_indices,
            }
        )

    elif args.model_type == "random_forest":
        if args.feature_mode is None:
            raise ValueError("--feature-mode is required when --model-type random_forest.")

        spatial_window_sizes = parse_int_list(args.spatial_window_sizes)
        spatial_band_indices = parse_int_list(args.spatial_band_indices)

        rf_normalize_mode = (
            args.normalize_mode_override
            if args.normalize_mode_override is not None
            else "scale_10000"
        )
        rf_scale_divisor = (
            args.scale_divisor_override
            if args.scale_divisor_override is not None
            else 10000.0
        )

        if args.data_source == "manifest":
            dataset = build_random_forest_manifest_dataset(
                manifest_csv=manifest_csv,
                split=args.split,
                limit=args.limit,
                band_indices=band_indices_override,
                normalize_mode_override=args.normalize_mode_override,
                scale_divisor_override=args.scale_divisor_override,
            )
        else:
            dataset = build_root_dataset(
                dataset_root=dataset_root,
                limit=args.limit,
                band_indices=band_indices_override,
                normalize_mode=rf_normalize_mode,
                scale_divisor=float(rf_scale_divisor),
                sample_format=args.sample_format,
                allowed_sample_names=allowed_sample_names,
            )

        predictor = RandomForestSegmentationPredictor.from_joblib(
            model_path=checkpoint_path,
            feature_mode=args.feature_mode,
            predict_batch_size=args.predict_batch_size,
            spatial_window_sizes=spatial_window_sizes,
            spatial_band_indices=spatial_band_indices,
        )

        num_classes = args.num_classes

        summary_metadata.update(
            {
                "feature_mode": args.feature_mode,
                "predict_batch_size": args.predict_batch_size,
                "normalize_mode": rf_normalize_mode,
                "scale_divisor": rf_scale_divisor,
                "band_indices": band_indices_override,
                "spatial_window_sizes": spatial_window_sizes,
                "spatial_band_indices": spatial_band_indices,
            }
        )

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("[EVALUATE] model_type:", args.model_type)
    print("[EVALUATE] data_source:", args.data_source)
    print("[EVALUATE] device:", device)
    print("[EVALUATE] checkpoint:", checkpoint_path)
    print("[EVALUATE] manifest_csv:", manifest_csv)
    print("[EVALUATE] dataset_root:", dataset_root)
    print("[EVALUATE] metadata_csv:", metadata_csv)
    print("[EVALUATE] month_filter_enabled:", month_filter_enabled)
    print("[EVALUATE] month_start:", args.month_start)
    print("[EVALUATE] month_end:", args.month_end)

    if allowed_sample_names is not None:
        print("[EVALUATE] allowed_samples_after_month_filter:", len(allowed_sample_names))

    print("[EVALUATE] sample_format:", args.sample_format if args.data_source == "root" else None)
    print("[EVALUATE] split:", args.split if args.data_source == "manifest" else None)
    print("[EVALUATE] output_dir:", output_dir)
    print("[EVALUATE] batch_size:", args.batch_size)
    print("[EVALUATE] num_workers:", args.num_workers)
    print("[EVALUATE] limit:", args.limit)
    print("[EVALUATE] ignore_index:", args.ignore_index)
    print("[EVALUATE] ignore_background:", args.ignore_background)
    print("[EVALUATE] dataset:", dataset.describe())

    if hasattr(dataset, "describe_storage"):
        print("[EVALUATE] dataset storage:", dataset.describe_storage())

    result = evaluate_segmentation_predictor(
        predictor=predictor,
        dataloader=dataloader,
        num_classes=num_classes,
        device=device,
        output_dir=output_dir,
        ignore_index=args.ignore_index,
        ignore_background=args.ignore_background,
        class_names=DEFAULT_CLASS_NAMES,
        summary_metadata=summary_metadata,
        progress_every=50,
        log_prefix="[EVALUATE]",
    )

    print()
    print("[EVALUATE] RESULTS")
    print(f"[EVALUATE] mIoU_all_classes={result.summary['mIoU_all_classes']:.4f}")
    print(f"[EVALUATE] mIoU_present_classes={result.summary['mIoU_present_classes']:.4f}")
    print(f"[EVALUATE] pixel_accuracy={result.summary['pixel_accuracy']:.4f}")
    print(
        "[EVALUATE] mean_class_accuracy_all_classes="
        f"{result.summary['mean_class_accuracy_all_classes']:.4f}"
    )
    print(
        "[EVALUATE] mean_class_accuracy_present_classes="
        f"{result.summary['mean_class_accuracy_present_classes']:.4f}"
    )
    print("[EVALUATE] saved outputs:")

    for name, path in result.saved_paths.items():
        print(f"[EVALUATE]   {name}: {path}")


if __name__ == "__main__":
    main()