from dataclasses import dataclass
from typing import Tuple

@dataclass
class SentinelProduct:
    """
    Metadata-only representation of a Sentinel-2 product.
    """
    product_id: str
    tile_id: str
    acquisition_date: str
    cloud_cover: float
    footprint: Tuple[float, float, float, float]
    stac_item: object | None = None
