from __future__ import annotations

import re


_SENTINEL_SCENE_DATE_PATTERN = re.compile(r"_(\d{8})T\d{6}_")
_SENTINEL_TILE_PATTERN = re.compile(r"(?:^|_)T(\d{2}[A-Z]{3})(?:_|\.|$)")


def infer_sentinel_scene_date(product_id: str | None) -> str:
    """Extract the acquisition date encoded in a Sentinel-2 product ID.

    Args:
        product_id: Full product or SAFE identifier; may be missing.

    Returns:
        ISO `YYYY-MM-DD` acquisition date, or an empty string when absent.
    """
    if not product_id:
        return ""

    match = _SENTINEL_SCENE_DATE_PATTERN.search(str(product_id))

    if match is None:
        return ""

    raw_date = match.group(1)

    return f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"


def infer_sentinel_tile_id(product_id: str | None) -> str:
    """Extract the MGRS tile identifier encoded in a Sentinel-2 product ID.

    Args:
        product_id: Full product or SAFE identifier; may be missing.

    Returns:
        Tile identifier such as `T33TWN`, or an empty string when absent.
    """
    if not product_id:
        return ""

    match = _SENTINEL_TILE_PATTERN.search(str(product_id))

    if match is None:
        return ""

    return f"T{match.group(1)}"