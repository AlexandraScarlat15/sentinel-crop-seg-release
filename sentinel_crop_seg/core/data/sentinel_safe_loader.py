"""Read Sentinel-2 SAFE spectral bands and SCL on a shared B02 grid.

Spectral bands use bilinear resampling and categorical SCL uses nearest
resampling. Requested windows are expressed in B02 reference pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from sentinel_crop_seg.core.data.sentinel_loader import SentinelDataLoader


class SentinelSafeDataLoader(SentinelDataLoader):
    """Load Sentinel-2 L2A SAFE assets aligned to the 10 m B02 reference grid.

    Args:
        config: Dataset bands/resolutions and Sentinel cache-root settings.
        safe_id: Existing SAFE directory identifier under the cache root.

    Raises:
        FileNotFoundError: If the SAFE folder or required bands are absent.
        ValueError: If B02 is missing or has an incompatible reference grid.
    """

    def __init__(self, config: dict, safe_id: str):
        self.config = config

        paths_cfg = config.get("paths", {})
        root = Path(paths_cfg["sentinel_data_root"])
        self.safe_root = root / "cache" / safe_id

        if not self.safe_root.exists():
            raise FileNotFoundError(f"SAFE folder not found: {self.safe_root}")

        self.band_resolutions: Dict[str, int] = {
            band: int(cfg["resolution"])
            for band, cfg in config["dataset"]["bands"].items()
        }

        self.band_paths = self._discover_band_paths()
        self.scl_path = self._discover_scl_path()

        self._ref = self._load_reference_grid()

    def _find_img_data_dirs(self) -> List[Path]:
        """Find IMG_DATA directories across every SAFE granule."""
        granule_root = self.safe_root / "GRANULE"
        if not granule_root.exists():
            raise FileNotFoundError(f"SAFE missing GRANULE folder: {granule_root}")

        img_data_dirs: List[Path] = []
        for g in granule_root.iterdir():
            if not g.is_dir():
                continue
            img_data = g / "IMG_DATA"
            if img_data.exists() and img_data.is_dir():
                img_data_dirs.append(img_data)

        if not img_data_dirs:
            raise FileNotFoundError(f"No IMG_DATA folder found in SAFE: {self.safe_root}")

        return img_data_dirs

    def _discover_band_paths(self) -> Dict[str, Path]:
        """Find each configured JP2 band, preferring its resolution folder."""
        band_paths: Dict[str, Path] = {}
        img_data_dirs = self._find_img_data_dirs()

        for band_name, res in self.band_resolutions.items():
            wanted_res_folder = f"R{res}m"

            matches: List[Path] = []
            for img_data in img_data_dirs:
                matches += list(img_data.rglob(f"*{band_name}*.jp2"))

            if not matches:
                existing: List[str] = []
                for img_data in img_data_dirs:
                    existing += [p.name for p in img_data.rglob("*.jp2")]
                preview = "\n".join(existing[:80])

                raise FileNotFoundError(
                    f"Band {band_name} not found in SAFE.\n"
                    f"SAFE: {self.safe_root}\n"
                    f"IMG_DATA dirs searched:\n" + "\n".join(str(d) for d in img_data_dirs) + "\n\n"
                    f"Preview of existing JP2 files (first 80):\n{preview}"
                )

            preferred = [p for p in matches if wanted_res_folder in str(p)]
            band_paths[band_name] = preferred[0] if preferred else matches[0]

        return band_paths

    def _discover_scl_path(self) -> Optional[Path]:
        """Find the optional SCL JP2 across granules, preferring R20m."""
        img_data_dirs = self._find_img_data_dirs()

        matches: List[Path] = []
        for img_data in img_data_dirs:
            matches += list(img_data.rglob("*SCL*.jp2"))

        if not matches:
            return None

        preferred = [p for p in matches if "R20m" in str(p)]
        return preferred[0] if preferred else matches[0]

    def _load_reference_grid(self) -> Dict[str, Any]:
        """Read and validate the 10 m B02 spatial reference grid."""
        if "B02" not in self.band_paths:
            raise ValueError("B02 band not found. Cannot build a 10m reference grid.")

        with rasterio.open(self.band_paths["B02"]) as src:
            transform = src.transform
            crs = src.crs.to_string()
            width = src.width
            height = src.height

            px_x = abs(transform.a)
            px_y = abs(transform.e)
            if not (9.0 <= px_x <= 11.0 and 9.0 <= px_y <= 11.0):
                raise ValueError(
                    f"B02 reference is not ~10m. Got pixel size {px_x} x {px_y}. "
                    f"Band path: {self.band_paths['B02']}"
                )

            return {
                "crs": crs,
                "transform": transform,
                "width": width,
                "height": height,
            }

    def list_bands(self) -> List[str]:
        return list(self.band_paths.keys())

    def get_spatial_metadata(self) -> Dict[str, Any]:
        """Return B02 CRS, affine transform and image dimensions."""
        return dict(self._ref)

    def _clamp_window_to_ref(self, window: Tuple[int, int, int, int]) -> Window:
        """Constrain a requested window to the available B02 pixel extent."""
        col, row, width, height = window
        ref_w = int(self._ref["width"])
        ref_h = int(self._ref["height"])

        col = max(0, min(int(col), ref_w - 1))
        row = max(0, min(int(row), ref_h - 1))

        width = max(1, min(int(width), ref_w - col))
        height = max(1, min(int(height), ref_h - row))

        return Window(col, row, width, height)

    def read_window(
        self,
        window: Tuple[int, int, int, int],
        bands: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Read spectral bands as a channel-first array aligned to B02.

        Args:
            window: Requested `(column, row, width, height)` in 10 m pixels.
            bands: Optional band subset in output-channel order; defaults to
                all configured bands.

        Returns:
            Spectral array with shape `(channels, height, width)`, cropped
            to the valid B02 image extent.

        Raises:
            KeyError: If an unconfigured band is requested.
            RuntimeError: If a band cannot be aligned to the reference window.
        """
        if bands is None:
            bands = self.list_bands()

        win = self._clamp_window_to_ref(window)
        height = int(win.height)
        width = int(win.width)

        ref_crs = self._ref["crs"]
        ref_transform = self._ref["transform"]
        ref_width = int(self._ref["width"])
        ref_height = int(self._ref["height"])

        data: List[np.ndarray] = []

        for band in bands:
            if band not in self.band_paths:
                raise KeyError(f"Requested band '{band}' not discovered. Available: {list(self.band_paths.keys())}")

            band_path = self.band_paths[band]

            with rasterio.open(band_path) as src:
                with WarpedVRT(
                    src,
                    crs=ref_crs,
                    transform=ref_transform,
                    width=ref_width,
                    height=ref_height,
                    resampling=Resampling.bilinear,
                ) as vrt:
                    patch = vrt.read(1, window=win)

            if patch.shape != (height, width):
                raise RuntimeError(
                    f"Band {band} returned shape {patch.shape}, expected {(height, width)}. "
                    f"Band path: {band_path}"
                )

            data.append(patch)

        return np.stack(data, axis=0)

    def read_scl(self, window: Tuple[int, int, int, int]) -> np.ndarray:
        """Read categorical SCL on the B02 grid with nearest resampling.

        Args:
            window: Requested `(column, row, width, height)` in 10 m pixels.

        Returns:
            Two-dimensional SCL array for the clamped reference window.

        Raises:
            RuntimeError: If the SAFE has no SCL file or alignment fails.
        """
        if self.scl_path is None:
            raise RuntimeError("SCL band not available in this SAFE")

        win = self._clamp_window_to_ref(window)
        height = int(win.height)
        width = int(win.width)

        ref_crs = self._ref["crs"]
        ref_transform = self._ref["transform"]
        ref_width = int(self._ref["width"])
        ref_height = int(self._ref["height"])

        with rasterio.open(self.scl_path) as src:
            with WarpedVRT(
                src,
                crs=ref_crs,
                transform=ref_transform,
                width=ref_width,
                height=ref_height,
                resampling=Resampling.nearest,
            ) as vrt:
                scl = vrt.read(1, window=win)

        if scl.shape != (height, width):
            raise RuntimeError(f"SCL returned shape {scl.shape}, expected {(height, width)}")

        return scl
