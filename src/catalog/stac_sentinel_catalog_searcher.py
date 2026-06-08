"""
STAC Sentinel Catalog Searcher
==============================

Real Sentinel-2 catalog discovery using the Copernicus STAC API.
Metadata only (no downloads).
Includes robust retry logic for both connection and searching.
"""

import time
from typing import List, Tuple, Optional

from pystac_client import Client
from pystac_client.exceptions import APIError
from src.models.sentinel_product import SentinelProduct



class STACSentinelCatalogSearcher:
    """
    Sentinel-2 catalog discovery via Copernicus STAC.
    """

    STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
    COLLECTION = "sentinel-2-l2a"

    def __init__(self) -> None:
        """
        Initializeaza clientul STAC cu mecanism de retry la conectare.
        """
        self.client = None
        max_retries = 5
        
        print(f"[STAC] Connecting to {self.STAC_URL}...")
        
        for attempt in range(max_retries):
            try:
                
                self.client = Client.open(self.STAC_URL)
                
                break
            except Exception as e:
                wait_time = (attempt + 1) * 5 # 5s, 10s, 15s...
                print(f"[STAC] Connection failed (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"[STAC] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[STAC] Fatal Error: Could not connect to Copernicus STAC after {max_retries} attempts.")
                   
                    raise e

    def search(
        self,
        roi: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        max_cloud_cover: Optional[float] = None,
        limit: int = 20,
    ) -> List[SentinelProduct]:
        """
        Search Sentinel-2 products via STAC with Server-Side Filtering and Retry Logic.
        """
        
        query = {}
        if max_cloud_cover is not None:
            query["eo:cloud_cover"] = {"lt": max_cloud_cover}

        print(f"[STAC] Searching bbox={roi} clouds<{max_cloud_cover}...")
        
        search = self.client.search(
            collections=[self.COLLECTION],
            bbox=roi,
            datetime=f"{start_date}/{end_date}",
            limit=limit,
            query=query,
        )

        products: List[SentinelProduct] = []
        items = []

        # RETRY LOGIC 
        max_retries = 3
        for attempt in range(max_retries):
            try:
                
                items = list(search.items())
                print(f"[STAC] API returned {len(items)} items.")
                break
            
            except APIError as e:
                error_msg = str(e).lower()
                
                is_server_error = "conflict with recovery" in error_msg or "500" in error_msg or "502" in error_msg or "503" in error_msg or "timeout" in error_msg
                
                if is_server_error and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"[STAC] Server error during search (Attempt {attempt+1}/{max_retries}): {e}")
                    print(f"[STAC] Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[STAC] Search failed after {max_retries} attempts.")
                    raise e
            except Exception as e:
                print(f"[STAC] Unexpected search error: {e}")
                raise e

        for item in items:
            cloud = item.properties.get("eo:cloud_cover", 0.0)
            
            tile_id = (
                item.properties.get("s2:mgrs_tile") 
                or item.properties.get("mgrs:tile") 
                or self._extract_tile_from_id(item.id)
            )

            products.append(
                SentinelProduct(
                    product_id=item.id, 
                    tile_id=tile_id,
                    acquisition_date=item.datetime.date().isoformat(),
                    cloud_cover=float(cloud),
                    footprint=item.geometry,
                    stac_item=item,
                )
            )

        return products

    def _extract_tile_from_id(self, item_id: str) -> Optional[str]:
        parts = item_id.split("_")
        for part in parts:
            if (
                len(part) == 6
                and part.startswith("T")
                and part[1:3].isdigit()
                and part[3:].isalpha()
            ):
                return part
        return None