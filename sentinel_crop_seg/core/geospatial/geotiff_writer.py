from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio


def save_multiband_geotiff(
    output_path: str | Path,
    image: np.ndarray,
    transform,
    crs: str,
    band_names: Sequence[str] | None = None,
) -> None:
    """
    Save a multi-band image patch as a georeferenced GeoTIFF.

    Expected image shape: (bands, height, width).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (bands, H, W), got {image.shape}")

    count, height, width = image.shape

    if band_names is not None and len(band_names) != count:
        raise ValueError(
            f"Expected {count} band names, got {len(band_names)}"
        )

    data = np.asarray(image)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data)

        if band_names is not None:
            for index, band_name in enumerate(band_names, start=1):
                dst.set_band_description(index, str(band_name))


def save_mask_geotiff(
    output_path: str | Path,
    mask: np.ndarray,
    transform,
    crs: str,
    nodata: int | None = None,
) -> None:
    """
    Save a single-band segmentation mask as a georeferenced GeoTIFF.

    Expected mask shape: (height, width).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mask.ndim != 2:
        raise ValueError(f"Expected mask with shape (H, W), got {mask.shape}")

    data = np.asarray(mask).astype(np.uint8, copy=False)
    height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }

    if nodata is not None:
        profile["nodata"] = int(nodata)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "segmentation_mask")