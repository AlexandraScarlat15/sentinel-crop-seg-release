"""Shared, Random-Forest-free U-Net CLI setup for curated research releases."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from sentinel_crop_seg.core.evaluation.checkpoint import load_unet_checkpoint
from sentinel_crop_seg.core.training.dataset import SegmentationPatchDataset
from sentinel_crop_seg.core.training.dataset_manifest import ManifestSegmentationDataset
from sentinel_crop_seg.core.training.mask_resolver import (
    build_mask_resolver,
    load_mask_resolver_settings,
    require_manifest_data_source,
)


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted U-Net .pt checkpoint.")
    parser.add_argument("--data-source", choices=("manifest", "root"), default="manifest")
    parser.add_argument("--manifest-csv", type=Path, help="Manifest with img_path/mask_path and split.")
    parser.add_argument("--dataset-root", type=Path, help="Root images/masks or images_tif/masks_tif.")
    parser.add_argument("--sample-format", choices=("npy", "geotiff"), default="npy")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--label-config", type=Path, help="Optional versioned-label YAML.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-unsafe-checkpoint", action="store_true",
        help="Permit arbitrary pickle deserialization when loading the "
             "checkpoint. Needed only for a checkpoint that stores custom "
             "objects; it executes code from whoever wrote the file.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all available samples.")
    parser.add_argument("--band-indices", help="Optional comma-separated zero-based indices.")
    parser.add_argument("--normalize-mode-override", choices=("none", "scale_10000", "standardize"))
    parser.add_argument("--scale-divisor-override", type=float)


def _device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()) else "cpu")


def _bands(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        indices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--band-indices requires comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise ValueError("--band-indices requires nonnegative zero-based indices")
    return indices


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_runtime(args: argparse.Namespace) -> SimpleNamespace:
    """Return checkpoint/model/dataloader and auditable input provenance.

    Uses exactly the stored checkpoint preprocessing and band order by default.
    Does not select, train or modify source patches.
    """
    if args.batch_size <= 0 or args.num_workers < 0 or args.limit < 0:
        raise ValueError("batch-size must be positive; num-workers and limit must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    settings = load_mask_resolver_settings(args.label_config)
    require_manifest_data_source(settings, args.data_source)
    resolver = build_mask_resolver(settings)
    device = _device(args.device)
    loaded = load_unet_checkpoint(
        args.checkpoint, device=device,
        allow_unsafe_pickle=args.allow_unsafe_checkpoint,
    )
    cfg = loaded.config
    selected_bands = _bands(args.band_indices)
    if selected_bands is None:
        selected_bands = cfg.band_indices
    normalize = args.normalize_mode_override or cfg.normalize_mode
    divisor = args.scale_divisor_override if args.scale_divisor_override is not None else cfg.scale_divisor
    if divisor <= 0:
        raise ValueError("Scale divisor must be positive")

    manifest = args.manifest_csv if args.data_source == "manifest" else None
    if manifest is not None:
        dataset = ManifestSegmentationDataset(
            manifest_csv=manifest, split=args.split, limit=args.limit,
            shuffle=False, augment=False, band_indices=selected_bands,
            normalize_mode=normalize, scale_divisor=divisor,
            mask_resolver=resolver, calendar_features=cfg.calendar_features,
        )
    else:
        if args.data_source == "manifest":
            raise ValueError("--manifest-csv is required for --data-source manifest")
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required for --data-source root")
        if cfg.calendar_features != "none":
            raise ValueError("Calendar-feature checkpoint requires manifest acquisition dates")
        dataset = SegmentationPatchDataset(
            args.dataset_root, limit=args.limit, shuffle=False, augment=False,
            band_indices=selected_bands, normalize_mode=normalize,
            scale_divisor=divisor, sample_format=args.sample_format,
        )
    if dataset.describe()["num_channels"] != cfg.in_channels:
        raise ValueError(
            "Input channel mismatch: dataset has "
            f"{dataset.describe()['num_channels']}, checkpoint requires {cfg.in_channels}"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    provenance = {
        "model_type": "unet",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "data_source": args.data_source,
        "manifest_csv": str(manifest) if manifest is not None else None,
        "manifest_sha256": file_sha256(manifest) if manifest is not None else None,
        "dataset_root": str(args.dataset_root) if args.data_source == "root" else None,
        "split": args.split if args.data_source == "manifest" else None,
        "num_classes": cfg.num_classes,
        "in_channels": cfg.in_channels,
        "band_indices": list(selected_bands) if selected_bands is not None else None,
        "normalize_mode": normalize,
        "scale_divisor": float(divisor),
        "calendar_features": cfg.calendar_features,
        "label_provenance": {
            **settings.provenance(),
            "corrected_record_count": resolver.corrected_record_count,
        },
        "device": str(device),
    }
    return SimpleNamespace(dataset=dataset, dataloader=loader, model=loaded.model,
                           checkpoint=loaded, device=device, provenance=provenance)
