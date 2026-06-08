from __future__ import annotations

import re


_SENTINEL_SCENE_DATE_PATTERN = re.compile(r"_(\d{8})T\d{6}_")
_SENTINEL_TILE_PATTERN = re.compile(r"(?:^|_)T(\d{2}[A-Z]{3})(?:_|\.|$)")


def infer_sentinel_scene_date(product_id: str | None) -> str:
    """
    Infer Sentinel-2 acquisition date from a product id or SAFE id.

    Example:
    S2B_MSIL2A_20210810T095029_N0500_R079_T33TWN_20230211T233939
    -> 2021-08-10
    """
    if not product_id:
        return ""

    match = _SENTINEL_SCENE_DATE_PATTERN.search(str(product_id))

    if match is None:
        return ""

    raw_date = match.group(1)

    return f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"


def infer_sentinel_tile_id(product_id: str | None) -> str:
    """
    Infer Sentinel-2 tile id from a product id or SAFE id.

    Example:
    S2B_MSIL2A_20210810T095029_N0500_R079_T33TWN_20230211T233939
    -> T33TWN
    """
    if not product_id:
        return ""

    match = _SENTINEL_TILE_PATTERN.search(str(product_id))

    if match is None:
        return ""

    return f"T{match.group(1)}"