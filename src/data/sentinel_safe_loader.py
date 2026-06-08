"""
Sentinel SAFE Data Loader (robust)

Goals:
- Use a single 10m reference grid (B02).
- Discover band JP2 files robustly across ALL granules / IMG_DATA layouts.
- Read all bands aligned on the same CRS/transform/width/height via WarpedVRT.
- Windows are in 10m pixel coordinates (reference grid).

Notes:
- Some SAFE packages contain multiple granules.
- Folder structure can differ slightly (IMG_DATA, IMG_DATA/R10m, IMG_DATA/R20m etc.).
- Filename patterns also vary. We search by substring (e.g. "*B08*.jp2") and then prefer
  the requested resolution folder (R10m/R20m/R60m) if present.

Your dataset config expects:
config["dataset"]["bands"] = { "B02": {"resolution": 10}, ... }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from src.data.sentinel_loader import SentinelDataLoader


class SentinelSafeDataLoader(SentinelDataLoader):
    """
    Robust Sentinel-2 L2A SAFE loader.

    Guarantees:
      - reference grid = B02 at 10m
      - all bands read on exact same grid via WarpedVRT
      - windows are defined in reference 10m pixel coordinates
    """

    def __init__(self, config: dict, safe_id: str):
        self.config = config

        paths_cfg = config.get("paths", {})
        root = Path(paths_cfg["sentinel_data_root"])
        self.safe_root = root / "cache" / safe_id

        if not self.safe_root.exists():
            raise FileNotFoundError(f"SAFE folder not found: {self.safe_root}")

        # Band resolutions from config
        self.band_resolutions: Dict[str, int] = {
            band: int(cfg["resolution"])
            for band, cfg in config["dataset"]["bands"].items()
        }

        # Discover paths
        self.band_paths = self._discover_band_paths()
        self.scl_path = self._discover_scl_path()

        # Reference 10m grid (B02)
        self._ref = self._load_reference_grid()

    def _find_img_data_dirs(self) -> List[Path]:
        """
        Return all IMG_DATA dirs from all granules inside SAFE.
        """
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
        """
        Find JP2 path for each required band.
        Strategy:
        - search across ALL IMG_DATA dirs
        - match by substring "*{band_name}*.jp2" (robust to naming differences)
        - prefer the requested resolution folder (R10m/R20m/R60m) when available
        """
        band_paths: Dict[str, Path] = {}
        img_data_dirs = self._find_img_data_dirs()

        for band_name, res in self.band_resolutions.items():
            wanted_res_folder = f"R{res}m"

            matches: List[Path] = []
            for img_data in img_data_dirs:
                matches += list(img_data.rglob(f"*{band_name}*.jp2"))

            if not matches:
                # Helpful debug output
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

            # Prefer correct resolution folder if present
            preferred = [p for p in matches if wanted_res_folder in str(p)]
            band_paths[band_name] = preferred[0] if preferred else matches[0]

        return band_paths

    def _discover_scl_path(self) -> Optional[Path]:
        """
        Find SCL JP2 path (usually 20m). Not all SAFE exports include it.
        We search across all granules and prefer R20m if present.
        """
        img_data_dirs = self._find_img_data_dirs()

        matches: List[Path] = []
        for img_data in img_data_dirs:
            matches += list(img_data.rglob("*SCL*.jp2"))

        if not matches:
            return None

        preferred = [p for p in matches if "R20m" in str(p)]
        return preferred[0] if preferred else matches[0]

    def _load_reference_grid(self) -> Dict[str, Any]:
        """
        Use B02 (10m) as the reference grid for everything.
        """
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
        """
        Always return the reference 10m grid metadata (B02).
        """
        return dict(self._ref)

    def _clamp_window_to_ref(self, window: Tuple[int, int, int, int]) -> Window:
        """
        Clamp requested 10m window to reference grid bounds.
        """
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
        """
        Read all bands on the same 10m reference grid using WarpedVRT.
        Input window is in 10m pixel coordinates (reference grid).
        Output shape: (num_bands, height, width).
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
        """
        Read SCL and warp it to the 10m reference grid.
        Uses nearest resampling (categorical).
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
