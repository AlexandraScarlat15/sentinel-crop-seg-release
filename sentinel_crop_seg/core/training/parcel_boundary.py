"""Read-only lookup of parcel-wise geometric erosion supervision caches.

The cache is produced by ``tools/build_parcel_boundary_erosion_cache.py`` from
the EuroCrops polygons and each patch's original Sentinel grid.  One uint8
``survival_depth`` raster is stored per patch: a labelled pixel has value ``k``
when the *same physical parcel* remains at that pixel after 10, 20, ... ``k*10``
metres of independent inward polygon buffering.  It is deliberately not a
semantic-class raster erosion and is shared by the 1/2/3-pixel experiments.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentinel_crop_seg.core.training.boundary_ignore import validate_boundary_ignore_px


PARCEL_BOUNDARY_MODE = "per_parcel_inward_buffer_then_rasterize_all_touched_false"
PARCEL_BOUNDARY_CACHE_RELATIVE_PATH = Path("local_audit/parcel_boundary_erosion_v1/cache_manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParcelBoundaryCache:
    """Index the immutable per-patch parcel-survival rasters."""

    manifest_path: Path
    records: dict[tuple[str, str], Path]
    #: Only the patches that carry labelled pixels with no owning source parcel.
    ownership_records: dict[tuple[str, str], Path] = field(default_factory=dict)
    max_depth_px: int = 3  # Legacy 0,1,2,3 caches remain valid for old runs.

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        expected_training_manifest: Path,
        expected_label_version: str,
        required_depth_px: int = 0,
    ) -> "ParcelBoundaryCache":
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(
                "Parcel-boundary cache manifest is required for boundary_ignore_px > 0: "
                f"{path}. Run tools/build_parcel_boundary_erosion_cache.py first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != PARCEL_BOUNDARY_MODE:
            raise ValueError("Parcel-boundary cache uses an unexpected processing mode")
        if payload.get("limited"):
            raise ValueError("A limited parcel-boundary cache cannot be used for training")
        training = payload.get("training_manifest", {})
        if training.get("sha256") != sha256_file(expected_training_manifest):
            raise ValueError("Parcel-boundary cache was built for a different training manifest")
        if payload.get("label_version") != expected_label_version:
            raise ValueError("Parcel-boundary cache label version does not match training config")
        # Old metadata without depths represents the original [0,1,2,3]
        # cache. Never silently promote it to cover deeper, uncomputed buffers.
        needed = validate_boundary_ignore_px(required_depth_px)
        depths = payload.get("erosion_depths_px", [0, 1, 2, 3])
        if not isinstance(depths, list) or not depths:
            raise ValueError("Parcel-boundary cache has no valid erosion depth coverage")
        if any(type(value) is not int for value in depths):
            raise ValueError("Parcel-boundary cache erosion depths must be integers")
        maximum = validate_boundary_ignore_px(max(depths))
        if depths != list(range(maximum + 1)) or maximum < 3:
            raise ValueError(
                f"Parcel-boundary cache has incomplete/noncontiguous erosion depths: {depths}"
            )
        if needed > maximum:
            raise ValueError(
                f"Requested erosion depth {needed}px exceeds computed parcel-boundary "
                f"cache coverage {maximum}px; rebuild the cache to at least {needed}px"
            )

        records: dict[tuple[str, str], Path] = {}
        ownership: dict[tuple[str, str], Path] = {}
        for item in payload.get("records", []):
            key = (str(item["dataset"]), str(item["sample_id"]))
            if key in records:
                raise ValueError(f"Duplicate parcel-boundary cache key: {key}")

            def resolve(value: str) -> Path:
                candidate = Path(value)
                return candidate if candidate.is_absolute() else path.parent / candidate

            records[key] = resolve(item["survival_depth_path"])
            # Present only for the few patches that have such pixels at all, so
            # the exclusion stays sparse rather than a second full raster set.
            if item.get("ownership_unrecoverable_path"):
                ownership[key] = resolve(item["ownership_unrecoverable_path"])
        if not records:
            raise ValueError("Parcel-boundary cache has no records")
        return cls(
            manifest_path=path, records=records, ownership_records=ownership,
            max_depth_px=maximum,
        )

    def load_survival_depth(self, *, dataset: str, sample_id: str, shape: tuple[int, int]) -> np.ndarray:
        key = (str(dataset), str(sample_id))
        path = self.records.get(key)
        if path is None:
            raise KeyError(f"Parcel-boundary cache has no record for {dataset}/{sample_id}")
        depth = np.load(path, allow_pickle=False)
        if depth.shape != shape:
            raise ValueError(
                f"Parcel-boundary survival raster shape {depth.shape} does not match mask {shape} "
                f"for {dataset}/{sample_id}"
            )
        if (depth.dtype.kind not in "iu" or np.any(depth < 0)
                or np.any(depth > self.max_depth_px)):
            raise ValueError(
                f"Parcel-boundary survival raster has invalid values for declared "
                f"0..{self.max_depth_px}px cache: {dataset}/{sample_id}"
            )
        return depth

    def load_ownership_unrecoverable(
        self, *, dataset: str, sample_id: str, shape: tuple[int, int]
    ) -> np.ndarray:
        """Boolean mask of labelled pixels that no source parcel covers.

        Stored sparsely as ``(n, 2)`` row/column indices, because only a small
        minority of patches carry any such pixel.  An absent record means none,
        which is the common case and returns an all-False mask.
        """
        path = self.ownership_records.get((str(dataset), str(sample_id)))
        result = np.zeros(shape, dtype=bool)
        if path is None:
            return result
        indices = np.load(path, allow_pickle=False)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError(
                f"Ownership-unrecoverable record for {dataset}/{sample_id} must be (n, 2) "
                f"row/column indices, got shape {indices.shape}"
            )
        if indices.size:
            rows, cols = indices[:, 0], indices[:, 1]
            if rows.max() >= shape[0] or cols.max() >= shape[1] or rows.min() < 0 or cols.min() < 0:
                raise ValueError(
                    f"Ownership-unrecoverable indices for {dataset}/{sample_id} fall outside {shape}"
                )
            result[rows, cols] = True
        return result
