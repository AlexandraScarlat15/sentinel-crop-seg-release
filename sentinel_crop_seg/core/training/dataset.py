from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from sentinel_crop_seg.core.data.patch_io import (
    load_image_patch,
    load_mask_patch,
    validate_image_mask_pair,
)
from sentinel_crop_seg.core.training.boundary_ignore import apply_boundary_ignore, validate_boundary_ignore_px
from sentinel_crop_seg.core.training.parcel_boundary import ParcelBoundaryCache


@dataclass(frozen=True)
class SamplePaths:
    image_path: Path
    mask_path: Path


def _normalize_sample_format(sample_format: str) -> str:
    sample_format = str(sample_format).lower().strip()

    if sample_format in {"npy", "numpy"}:
        return "npy"

    if sample_format in {"tif", "tiff", "geotiff", "geo_tiff"}:
        return "geotiff"

    raise ValueError(
        f"Unsupported sample_format={sample_format!r}. "
        "Expected one of: 'npy', 'geotiff'."
    )


def _layout_for_format(dataset_folder: Path, sample_format: str) -> tuple[Path, Path, str]:
    sample_format = _normalize_sample_format(sample_format)

    if sample_format == "npy":
        return dataset_folder / "images", dataset_folder / "masks", ".npy"

    if sample_format == "geotiff":
        return dataset_folder / "images_tif", dataset_folder / "masks_tif", ".tif"

    raise RuntimeError(f"Unhandled sample_format: {sample_format}")


def _has_layout(dataset_folder: Path, sample_format: str) -> bool:
    images_dir, masks_dir, suffix = _layout_for_format(dataset_folder, sample_format)

    return (
        images_dir.exists()
        and masks_dir.exists()
        and any(images_dir.glob(f"sample_*{suffix}"))
    )


def _find_dataset_folders(root: Path, sample_format: str) -> List[Path]:
    if _has_layout(root, sample_format):
        return [root]

    folders: List[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and _has_layout(path, sample_format):
            folders.append(path)

    return folders


def _index_samples(dataset_folder: Path, sample_format: str) -> List[SamplePaths]:
    images_dir, masks_dir, suffix = _layout_for_format(dataset_folder, sample_format)

    image_files = sorted(images_dir.glob(f"sample_*{suffix}"))
    if not image_files:
        return []

    samples: List[SamplePaths] = []
    for image_path in image_files:
        mask_path = masks_dir / image_path.name

        if mask_path.exists():
            samples.append(
                SamplePaths(
                    image_path=image_path,
                    mask_path=mask_path,
                )
            )

    return samples


def _sample_matches_allowed_names(
    sample: SamplePaths,
    allowed_sample_names: set[str] | None,
) -> bool:
    if allowed_sample_names is None:
        return True

    image_name = sample.image_path.name
    image_stem = sample.image_path.stem

    # For nested dataset layouts such as:
    # root/AT_2021_beet_v2/images/sample_000001.npy
    dataset_name = sample.image_path.parent.parent.name
    dataset_relative_name = f"{dataset_name}/{image_name}"
    dataset_relative_stem = f"{dataset_name}/{image_stem}"

    return (
        image_name in allowed_sample_names
        or image_stem in allowed_sample_names
        or dataset_relative_name in allowed_sample_names
        or dataset_relative_stem in allowed_sample_names
    )


def _apply_geo_augment(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    k = int(np.random.randint(0, 4))
    if k:
        x = np.rot90(x, k=k, axes=(1, 2))
        y = np.rot90(y, k=k, axes=(0, 1))

    if bool(np.random.randint(0, 2)):
        x = x[:, :, ::-1]
        y = y[:, ::-1]

    if bool(np.random.randint(0, 2)):
        x = x[:, ::-1, :]
        y = y[::-1, :]

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    return x, y


class SegmentationPatchDataset:
    """
    Dataset for segmentation patches stored either as NumPy arrays or GeoTIFF files.

    Supported layouts:

    NumPy:
        root/images/sample_XXXXXX.npy
        root/masks/sample_XXXXXX.npy

    GeoTIFF:
        root/images_tif/sample_XXXXXX.tif
        root/masks_tif/sample_XXXXXX.tif
    """

    def __init__(
        self,
        root: str | Path,
        *,
        limit: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        image_dtype: np.dtype = np.float32,
        mask_dtype: np.dtype = np.int64,
        augment: bool = False,
        augment_prob: float = 1.0,
        band_indices: Sequence[int] | None = None,
        normalize_mode: str = "none",
        scale_divisor: float = 10000.0,
        channel_means: Sequence[float] | None = None,
        channel_stds: Sequence[float] | None = None,
        sample_format: str = "npy",
        allowed_sample_names: Sequence[str] | None = None,
        boundary_ignore_px: int = 0,
        ignore_index: int | None = None,
        parcel_boundary_cache: ParcelBoundaryCache | None = None,
    ) -> None:
        self.root = Path(root)
        self.sample_format = _normalize_sample_format(sample_format)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        folders = _find_dataset_folders(self.root, self.sample_format)
        if not folders:
            raise FileNotFoundError(
                f"No dataset folders found under: {self.root}\n"
                "Expected one of these layouts:\n"
                "  NumPy:   root/images + root/masks\n"
                "  GeoTIFF: root/images_tif + root/masks_tif"
            )

        samples: List[SamplePaths] = []
        for folder in folders:
            samples.extend(_index_samples(folder, self.sample_format))

        if not samples:
            raise FileNotFoundError(f"No samples found under: {self.root}")

        allowed_names_set = (
            None
            if allowed_sample_names is None
            else {str(name) for name in allowed_sample_names}
        )

        if allowed_names_set is not None:
            samples = [
                sample
                for sample in samples
                if _sample_matches_allowed_names(sample, allowed_names_set)
            ]

            if not samples:
                raise FileNotFoundError(
                    "No samples matched allowed_sample_names under: "
                    f"{self.root}"
                )

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(samples)

        if limit and limit > 0:
            samples = samples[: int(limit)]

        self.samples = samples
        self.image_dtype = image_dtype
        self.mask_dtype = mask_dtype

        self.augment = bool(augment)
        self.augment_prob = float(augment_prob)
        if not (0.0 <= self.augment_prob <= 1.0):
            raise ValueError(f"augment_prob must be in [0, 1], got {self.augment_prob}")

        self.seed = int(seed)
        self.boundary_ignore_px = validate_boundary_ignore_px(boundary_ignore_px)
        if self.boundary_ignore_px > 0 and ignore_index is None:
            raise ValueError("boundary_ignore_px requires a configured ignore_index")
        if self.boundary_ignore_px > 0 and parcel_boundary_cache is None:
            raise ValueError("boundary_ignore_px requires a parcel_boundary_cache")
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.parcel_boundary_cache = parcel_boundary_cache

        self.band_indices = (
            None if band_indices is None else tuple(int(index) for index in band_indices)
        )
        self.normalize_mode = str(normalize_mode).lower().strip()
        self.scale_divisor = float(scale_divisor)

        if self.band_indices is not None and len(self.band_indices) == 0:
            raise ValueError("band_indices must be None or a non-empty sequence")

        valid_modes = {"none", "scale_10000", "standardize"}
        if self.normalize_mode not in valid_modes:
            raise ValueError(
                f"Invalid normalize_mode={self.normalize_mode}. "
                f"Expected one of {sorted(valid_modes)}"
            )

        self.channel_means = (
            None if channel_means is None else np.asarray(channel_means, dtype=np.float32)
        )
        self.channel_stds = (
            None if channel_stds is None else np.asarray(channel_stds, dtype=np.float32)
        )

        if self.normalize_mode == "standardize":
            if self.channel_means is None or self.channel_stds is None:
                raise ValueError("standardize mode requires channel_means and channel_stds")

            if self.band_indices is not None:
                expected_channels = len(self.band_indices)

                if len(self.channel_means) != expected_channels:
                    raise ValueError(
                        "channel_means length must match number of selected bands"
                    )

                if len(self.channel_stds) != expected_channels:
                    raise ValueError(
                        "channel_stds length must match number of selected bands"
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def _select_bands(self, x: np.ndarray) -> np.ndarray:
        if self.band_indices is None:
            return x

        num_channels = x.shape[0]
        for index in self.band_indices:
            if index < 0 or index >= num_channels:
                raise ValueError(
                    f"Band index {index} out of range for image with "
                    f"{num_channels} channels"
                )

        return x[list(self.band_indices), :, :]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize_mode == "none":
            return x

        if self.normalize_mode == "scale_10000":
            return x / self.scale_divisor

        if self.normalize_mode == "standardize":
            means = self.channel_means[:, None, None]
            stds = self.channel_stds[:, None, None]
            stds = np.where(stds > 0.0, stds, 1.0)
            return (x - means) / stds

        raise RuntimeError(f"Unhandled normalize_mode: {self.normalize_mode}")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self.samples[idx]

        x = load_image_patch(sample.image_path)
        y = load_mask_patch(sample.mask_path)

        validate_image_mask_pair(
            image=x,
            mask=y,
            image_path=sample.image_path,
            mask_path=sample.mask_path,
        )

        x = self._select_bands(x)
        x = x.astype(self.image_dtype, copy=False)
        x = self._normalize(x)

        y = y.astype(self.mask_dtype, copy=False)

        if self.boundary_ignore_px > 0:
            assert self.ignore_index is not None
            assert self.parcel_boundary_cache is not None
            y = apply_boundary_ignore(
                y,
                boundary_ignore_px=self.boundary_ignore_px,
                ignore_index=self.ignore_index,
                # Root datasets lack manifest identities and therefore cannot
                # safely reconstruct parcel geometry for an ablation.
                parcel_survival_depth=self.parcel_boundary_cache.load_survival_depth(
                    dataset=sample.image_path.parent.parent.name,
                    sample_id=sample.image_path.stem,
                    shape=y.shape,
                ),
            )

        if self.augment and np.random.random() < self.augment_prob:
            x, y = _apply_geo_augment(x, y)

        return x, y

    def describe(self) -> Dict[str, int]:
        num_channels = 0

        if len(self.samples) > 0:
            x0 = load_image_patch(self.samples[0].image_path)
            x0 = self._select_bands(x0)
            num_channels = int(x0.shape[0])

        return {
            "num_samples": len(self.samples),
            "num_channels": num_channels,
        }

    def describe_storage(self) -> Dict[str, str]:
        return {
            "sample_format": self.sample_format,
            "image_suffix": self.samples[0].image_path.suffix.lower() if self.samples else "",
            "mask_suffix": self.samples[0].mask_path.suffix.lower() if self.samples else "",
        }


class NpySegmentationDataset(SegmentationPatchDataset):
    """
    Backward-compatible alias for older training scripts.

    New code should use SegmentationPatchDataset.
    """

    pass
