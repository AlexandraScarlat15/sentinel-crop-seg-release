"""Apply a precomputed parcel-geometry survival mask to semantic labels."""
from __future__ import annotations

import numpy as np


# A bounded, explicit performance/safety cap for the 10 m parcel buffers.
# A greater requested depth must also be computed by its source-geometry cache.
MAX_BOUNDARY_IGNORE_PX = 32


def validate_boundary_ignore_px(boundary_ignore_px: int) -> int:
    """Accept ONLY integer depths (0..32); no float truncation or bools."""
    if isinstance(boundary_ignore_px, (bool, np.bool_)) or not isinstance(
        boundary_ignore_px, (int, np.integer)
    ):
        raise ValueError(
            "boundary_ignore_px must be an integer in [0, 32], "
            f"got {boundary_ignore_px!r}"
        )
    value = int(boundary_ignore_px)
    if not (0 <= value <= MAX_BOUNDARY_IGNORE_PX):
        raise ValueError(
            f"boundary_ignore_px must be in [0, {MAX_BOUNDARY_IGNORE_PX}], "
            f"got {boundary_ignore_px!r}"
        )
    return value


def apply_boundary_ignore(
    mask: np.ndarray,
    *,
    boundary_ignore_px: int,
    ignore_index: int,
    parcel_survival_depth: np.ndarray | None,
    background_index: int = 0,
    ownership_unrecoverable: np.ndarray | None = None,
) -> np.ndarray:
    """Ignore labelled pixels not retained by independent parcel erosion.

    ``parcel_survival_depth`` is rasterized from each physical source parcel
    after 10/20/30 m inward geometric buffers in the patch's projected CRS.
    A positive depth therefore excludes only labelled pixels whose own parcel
    does not survive that buffer.  It does not infer boundaries from semantic
    labels, and a patch edge has no special treatment.  Existing ignores and
    background remain unchanged; removed supervision becomes ``ignore_index``.

    ``ownership_unrecoverable`` is a separate, depth-independent exclusion: a
    labelled pixel that NO source parcel covers has no physical parcel, so it
    has no core-versus-boundary status at any depth.  It is applied at every
    depth INCLUDING zero, which is the only way the eroded arms and a baseline
    arm can be compared on one supervision population.  It is optional and
    defaults to None, so the Model-A baseline is bit-for-bit unchanged unless
    it is passed deliberately.

    Neither exclusion ever changes a semantic class, invents a parcel owner, or
    turns a labelled pixel into background.
    """
    depth = validate_boundary_ignore_px(boundary_ignore_px)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")

    result = np.asarray(mask).astype(np.int64, copy=True)
    labelled = (result != int(background_index)) & (result != int(ignore_index))

    if depth > 0:
        survival = np.asarray(parcel_survival_depth)
        if survival.shape != result.shape:
            raise ValueError(
                f"parcel_survival_depth shape {survival.shape} does not match mask {result.shape}"
            )
        if survival.dtype.kind not in "iu" or np.any(survival < 0):
            raise ValueError("parcel_survival_depth must contain nonnegative integer values")
        result[labelled & (survival < depth)] = int(ignore_index)

    if ownership_unrecoverable is not None:
        unrecoverable = np.asarray(ownership_unrecoverable)
        if unrecoverable.shape != result.shape:
            raise ValueError(
                f"ownership_unrecoverable shape {unrecoverable.shape} does not match mask {result.shape}"
            )
        result[labelled & unrecoverable.astype(bool)] = int(ignore_index)

    return result
