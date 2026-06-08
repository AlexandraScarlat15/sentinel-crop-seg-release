from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


SUPPORTED_IMAGE_SUFFIXES = {".npy", ".tif", ".tiff"}
SUPPORTED_MASK_SUFFIXES = {".npy", ".tif", ".tiff"}


def _normalized_suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()


def load_image_patch(path: str | Path) -> np.ndarray:
    """
    Load a multispectral image patch from NumPy or GeoTIFF.

    Returned shape is always (C, H, W).
    No normalization is applied here.
    """
    path = Path(path)
    suffix = _normalized_suffix(path)

    if suffix == ".npy":
        image = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            image = src.read()
    else:
        raise ValueError(
            f"Unsupported image patch format: {path}. "
            f"Expected one of {sorted(SUPPORTED_IMAGE_SUFFIXES)}."
        )

    if image.ndim != 3:
        raise ValueError(
            f"Expected image patch with shape (C, H, W), got {image.shape} at {path}"
        )

    return image


def load_mask_patch(path: str | Path) -> np.ndarray:
    """
    Load a segmentation mask patch from NumPy or GeoTIFF.

    Returned shape is always (H, W).
    No dtype conversion is applied here.
    """
    path = Path(path)
    suffix = _normalized_suffix(path)

    if suffix == ".npy":
        mask = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(
                    f"Expected single-band mask GeoTIFF, got {src.count} bands at {path}"
                )
            mask = src.read(1)
    else:
        raise ValueError(
            f"Unsupported mask patch format: {path}. "
            f"Expected one of {sorted(SUPPORTED_MASK_SUFFIXES)}."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected mask patch with shape (H, W), got {mask.shape} at {path}"
        )

    return mask


def validate_image_mask_pair(
    image: np.ndarray,
    mask: np.ndarray,
    image_path: str | Path,
    mask_path: str | Path,
) -> None:
    """
    Validate image-mask compatibility after loading.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    if image.ndim != 3:
        raise ValueError(
            f"Expected image (C, H, W), got shape={image.shape} at {image_path}"
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected mask (H, W), got shape={mask.shape} at {mask_path}"
        )

    if image.shape[1:] != mask.shape:
        raise ValueError(
            f"Image/mask spatial mismatch: image={image.shape}, mask={mask.shape} "
            f"for image={image_path.name}, mask={mask_path.name}"
        )