from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    RandomForestSegmentationPredictor,
    load_unet_checkpoint,
    read_allowed_sample_names_by_month,
    validate_month_filter_args,
)
from src.training.dataset import SegmentationPatchDataset
from src.training.dataset_manifest import ManifestSegmentationDataset
from src.utils.class_palette import (
    CLASS_ID_TO_COLOR,
    IGNORE_COLOR,
    NUM_CLASSES,
)
from src.utils.visualization import (
    make_cmap_and_norm,
    mask_for_display,
    save_visualization,
    stretch_rgb,
)


def parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def safe_prediction_filename(output_index: int, sample_id: str) -> str:
    safe = str(sample_id).replace("\\", "/").strip("/")
    safe = safe.replace("/", "__")
    safe = Path(safe).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)

    if not safe:
        safe = "sample"

    return f"{output_index:06d}_{safe}.npy"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")

    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def build_color_table() -> np.ndarray:
    colors = [CLASS_ID_TO_COLOR[class_id] for class_id in range(NUM_CLASSES)]
    colors.append(IGNORE_COLOR)

    return np.array(
        [hex_to_rgb(color) for color in colors],
        dtype=np.uint8,
    )


def colorize_mask(mask: np.ndarray, ignore_index: int) -> np.ndarray:
    display_mask = mask_for_display(mask, ignore_index=ignore_index)
    color_table = build_color_table()

    return color_table[display_mask]


def save_rgb_image(output_path: Path, rgb: np.ndarray) -> None:
    plt.imsave(output_path, np.clip(rgb, 0.0, 1.0))


def save_color_mask_image(
    output_path: Path,
    mask: np.ndarray,
    ignore_index: int,
) -> None:
    color_mask = colorize_mask(mask, ignore_index=ignore_index)
    plt.imsave(output_path, color_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified prediction entry point for crop segmentation experiments. "
            "Supports U-Net and Random Forest prediction on manifest-based or root-folder datasets."
        )
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=["unet", "random_forest"],
        help="Model family to use for prediction.",
    )

    parser.add_argument(
        "--data-source",
        required=True,
        choices=["manifest", "root"],
        help="Prediction data source. Supported values: manifest, root.",
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
            "Manifest split to predict, for example train, val, or test. "
            "Used only with --data-source manifest."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where prediction outputs will be saved.",
    )

    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Prediction batch size.",
    )

    parser.add_argument(
        "--limit",
        default=0,
        type=int,
        help="Optional maximum number of samples to predict. Use 0 for all samples.",
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
        help="Prediction device.",
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

    parser.add_argument(
        "--no-save-npy",
        action="store_true",
        help="Do not save predicted masks as .npy files. Mainly useful for future extensions.",
    )

    parser.add_argument(
        "--save-color",
        action="store_true",
        help="Save RGB images, ground-truth color masks, and prediction color masks.",
    )

    parser.add_argument(
        "--save-panels",
        action="store_true",
        help="Save RGB / ground truth / prediction comparison panels.",
    )

    parser.add_argument(
        "--mask-pred-ignore",
        action="store_true",
        help=(
            "For visualization only, display ignored ground-truth pixels as ignore "
            "in the prediction panel/color mask."
        ),
    )

    parser.add_argument(
        "--ignore-index",
        default=255,
        type=int,
        help="Ground-truth ignore label used only for visualization.",
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


def sample_to_record(sample: Any) -> dict[str, str]:
    if hasattr(sample, "sample_id"):
        image_path = Path(sample.image_path)
        mask_path = Path(sample.mask_path)

        return {
            "dataset": str(sample.dataset),
            "sample_id": str(sample.sample_id),
            "split": str(sample.split),
            "image_path": image_path.as_posix(),
            "mask_path": mask_path.as_posix(),
        }

    image_path = Path(sample.image_path)
    mask_path = Path(sample.mask_path)
    dataset_name = image_path.parent.parent.name
    sample_id = f"{dataset_name}/{image_path.name}"

    return {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "split": "",
        "image_path": image_path.as_posix(),
        "mask_path": mask_path.as_posix(),
    }


@torch.no_grad()
def predict_batch(
    *,
    model_type: str,
    predictor: Any,
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    images = images.to(device=device, dtype=torch.float32)

    if model_type == "unet":
        predictor.eval()
        logits = predictor(images)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        predictions = torch.argmax(logits, dim=1)

    elif model_type == "random_forest":
        predictions = predictor.predict(images)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return predictions.detach().cpu().numpy().astype(np.int64, copy=False)


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

    prediction_metadata: dict[str, Any] = {
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
        "save_npy": not args.no_save_npy,
        "save_color": args.save_color,
        "save_panels": args.save_panels,
        "mask_pred_ignore": args.mask_pred_ignore,
        "ignore_index": args.ignore_index,
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

        predictor = loaded_checkpoint.model.to(device)
        predictor.eval()

        prediction_metadata.update(
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

        prediction_metadata.update(
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

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_npy_dir = output_dir / "pred_masks_npy"
    if not args.no_save_npy:
        pred_npy_dir.mkdir(parents=True, exist_ok=True)

    pred_color_dir = output_dir / "pred_masks_color"
    gt_color_dir = output_dir / "gt_masks_color"
    rgb_dir = output_dir / "rgb"
    panels_dir = output_dir / "panels"

    if args.save_color:
        pred_color_dir.mkdir(parents=True, exist_ok=True)
        gt_color_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)

    if args.save_panels:
        panels_dir.mkdir(parents=True, exist_ok=True)

    cmap, norm = make_cmap_and_norm(include_ignore=True)

    print("[PREDICT] model_type:", args.model_type)
    print("[PREDICT] data_source:", args.data_source)
    print("[PREDICT] device:", device)
    print("[PREDICT] checkpoint:", checkpoint_path)
    print("[PREDICT] manifest_csv:", manifest_csv)
    print("[PREDICT] dataset_root:", dataset_root)
    print("[PREDICT] metadata_csv:", metadata_csv)
    print("[PREDICT] month_filter_enabled:", month_filter_enabled)
    print("[PREDICT] month_start:", args.month_start)
    print("[PREDICT] month_end:", args.month_end)

    if allowed_sample_names is not None:
        print("[PREDICT] allowed_samples_after_month_filter:", len(allowed_sample_names))

    print("[PREDICT] sample_format:", args.sample_format if args.data_source == "root" else None)
    print("[PREDICT] split:", args.split if args.data_source == "manifest" else None)
    print("[PREDICT] output_dir:", output_dir)
    print("[PREDICT] batch_size:", args.batch_size)
    print("[PREDICT] num_workers:", args.num_workers)
    print("[PREDICT] limit:", args.limit)
    print("[PREDICT] save_color:", args.save_color)
    print("[PREDICT] save_panels:", args.save_panels)
    print("[PREDICT] mask_pred_ignore:", args.mask_pred_ignore)
    print("[PREDICT] ignore_index:", args.ignore_index)
    print("[PREDICT] dataset:", dataset.describe())

    if hasattr(dataset, "describe_storage"):
        print("[PREDICT] dataset storage:", dataset.describe_storage())

    rows: list[dict[str, Any]] = []
    processed = 0

    for batch_idx, batch in enumerate(dataloader):
        images, masks = batch

        predictions = predict_batch(
            model_type=args.model_type,
            predictor=predictor,
            images=images,
            device=device,
        )

        batch_size = int(predictions.shape[0])

        for item_idx in range(batch_size):
            output_index = processed + item_idx
            sample = dataset.samples[output_index]
            sample_record = sample_to_record(sample)

            prediction_path = ""
            rgb_path = ""
            gt_color_path = ""
            pred_color_path = ""
            panel_path = ""

            prediction_filename = safe_prediction_filename(
                output_index=output_index,
                sample_id=sample_record["sample_id"],
            )

            if not args.no_save_npy:
                prediction_file = pred_npy_dir / prediction_filename
                np.save(prediction_file, predictions[item_idx])
                prediction_path = prediction_file.as_posix()

            if args.save_color or args.save_panels:
                x_np = images[item_idx].detach().cpu().numpy()
                y_np = masks[item_idx].detach().cpu().numpy().astype(np.int64, copy=False)
                pred_np = predictions[item_idx].astype(np.int64, copy=False)

                rgb = stretch_rgb(x_np)

                pred_to_show = pred_np.copy()
                if args.mask_pred_ignore:
                    pred_to_show[y_np == args.ignore_index] = args.ignore_index

                visual_stem = Path(prediction_filename).stem

                if args.save_color:
                    rgb_path_obj = rgb_dir / f"{visual_stem}_rgb.png"
                    gt_color_path_obj = gt_color_dir / f"{visual_stem}_gt.png"
                    pred_color_path_obj = pred_color_dir / f"{visual_stem}_pred.png"

                    save_rgb_image(rgb_path_obj, rgb)
                    save_color_mask_image(
                        output_path=gt_color_path_obj,
                        mask=y_np,
                        ignore_index=args.ignore_index,
                    )
                    save_color_mask_image(
                        output_path=pred_color_path_obj,
                        mask=pred_to_show,
                        ignore_index=args.ignore_index,
                    )

                    rgb_path = rgb_path_obj.as_posix()
                    gt_color_path = gt_color_path_obj.as_posix()
                    pred_color_path = pred_color_path_obj.as_posix()

                if args.save_panels:
                    panel_path_obj = panels_dir / f"{visual_stem}_panel.png"

                    save_visualization(
                        output_path=panel_path_obj,
                        rgb=rgb,
                        gt_mask=y_np,
                        pred_mask=pred_to_show,
                        ignore_index=args.ignore_index,
                        title=sample_record["sample_id"],
                        cmap=cmap,
                        norm=norm,
                        include_legend=False,
                    )

                    panel_path = panel_path_obj.as_posix()

            rows.append(
                {
                    "output_index": output_index,
                    "dataset": sample_record["dataset"],
                    "sample_id": sample_record["sample_id"],
                    "split": sample_record["split"],
                    "image_path": sample_record["image_path"],
                    "mask_path": sample_record["mask_path"],
                    "prediction_path": prediction_path,
                    "rgb_path": rgb_path,
                    "gt_color_path": gt_color_path,
                    "pred_color_path": pred_color_path,
                    "panel_path": panel_path,
                    "height": int(predictions[item_idx].shape[0]),
                    "width": int(predictions[item_idx].shape[1]),
                }
            )

        processed += batch_size

        if processed == batch_size or processed % 50 == 0:
            print(f"[PREDICT] processed {processed}/{len(dataset)} samples")

    samples_csv = output_dir / "prediction_samples.csv"
    with samples_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "output_index",
            "dataset",
            "sample_id",
            "split",
            "image_path",
            "mask_path",
            "prediction_path",
            "rgb_path",
            "gt_color_path",
            "pred_color_path",
            "panel_path",
            "height",
            "width",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    summary = {
        **prediction_metadata,
        "num_predictions": processed,
        "prediction_samples_csv": samples_csv,
        "pred_masks_npy_dir": pred_npy_dir if not args.no_save_npy else None,
        "rgb_dir": rgb_dir if args.save_color else None,
        "gt_color_dir": gt_color_dir if args.save_color else None,
        "pred_color_dir": pred_color_dir if args.save_color else None,
        "panels_dir": panels_dir if args.save_panels else None,
    }

    summary_path = output_dir / "prediction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    print()
    print("[PREDICT] done")
    print("[PREDICT] predictions:", processed)
    print("[PREDICT] saved CSV:", samples_csv)
    print("[PREDICT] saved summary:", summary_path)

    if not args.no_save_npy:
        print("[PREDICT] saved NPY predictions:", pred_npy_dir)

    if args.save_color:
        print("[PREDICT] saved RGB images:", rgb_dir)
        print("[PREDICT] saved GT color masks:", gt_color_dir)
        print("[PREDICT] saved prediction color masks:", pred_color_dir)

    if args.save_panels:
        print("[PREDICT] saved panels:", panels_dir)


if __name__ == "__main__":
    main()