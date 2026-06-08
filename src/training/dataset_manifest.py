from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.patch_io import (
    load_image_patch,
    load_mask_patch,
    validate_image_mask_pair,
)


@dataclass(frozen=True)
class ManifestSample:
    dataset: str
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str


def _resolve_manifest_path(path_value: str | Path, base_dir: Path) -> Path:
    """
    Resolve a path from a manifest row.

    Absolute paths are used as-is.
    Relative paths are interpreted relative to base_dir.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return base_dir / path


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


class ManifestSegmentationDataset:
    """
    Dataset for segmentation patches listed in a manifest CSV.

    The manifest must contain at least:

        dataset,sample_id,img_path,mask_path,split

    The image and mask paths may point either to NumPy files or GeoTIFF files:

        .npy
        .tif
        .tiff

    Relative paths are resolved relative to the manifest CSV parent directory,
    unless base_dir is provided explicitly.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        *,
        base_dir: str | Path | None = None,
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
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.split = str(split).strip().lower()

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {self.split}")

        self.base_dir = (
            Path(base_dir)
            if base_dir is not None
            else self.manifest_csv.parent
        )

        df = pd.read_csv(self.manifest_csv)

        required_cols = {"dataset", "sample_id", "img_path", "mask_path", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

        df["split"] = df["split"].astype(str).str.strip().str.lower()
        df = df[df["split"] == self.split].copy()

        if df.empty:
            raise ValueError(f"No rows found for split='{self.split}' in {self.manifest_csv}")

        samples: List[ManifestSample] = []
        for row in df.itertuples(index=False):
            img_path = _resolve_manifest_path(row.img_path, self.base_dir)
            mask_path = _resolve_manifest_path(row.mask_path, self.base_dir)

            if not img_path.exists():
                raise FileNotFoundError(f"Image path not found: {img_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask path not found: {mask_path}")

            samples.append(
                ManifestSample(
                    dataset=str(row.dataset),
                    sample_id=str(row.sample_id),
                    image_path=img_path,
                    mask_path=mask_path,
                    split=str(row.split),
                )
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

        if self.augment and np.random.random() < self.augment_prob:
            x, y = _apply_geo_augment(x, y)

        return x, y

    def describe(self) -> Dict[str, int]:
        num_channels = 0

        if len(self.samples) > 0:
            first_sample = self.samples[0]
            x0 = load_image_patch(first_sample.image_path)
            x0 = self._select_bands(x0)
            num_channels = int(x0.shape[0])

        return {
            "num_samples": len(self.samples),
            "num_channels": num_channels,
        }


    def describe_storage(self) -> Dict[str, str]:
        if not self.samples:
            return {
                "image_format": "",
                "mask_format": "",
            }

        first_sample = self.samples[0]

        return {
            "image_format": first_sample.image_path.suffix.lower().lstrip("."),
            "mask_format": first_sample.mask_path.suffix.lower().lstrip("."),
        }