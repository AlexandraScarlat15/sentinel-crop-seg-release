from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
import time
import shutil

from sentinel_crop_seg.core.catalog.copernicus_safe_downloader import CopernicusSafeDownloader
from sentinel_crop_seg.core.models.sentinel_product import SentinelProduct
from sentinel_crop_seg.core.data.odata_resolver import resolve_uuid_from_odata


class SentinelCacheManager:
    """
    SAFE cache is global per product_id (download once).
    Processed state is per ROI+product key.

    index.json format:
    {
      "products": {
         "<product_id>": {
            "safe_id": "...",
            "path": "...",
            "registered_at": <unix>,
            "last_used": <unix>
         }
      },
      "processed": {
         "<roi_product_key>": true
      }
    }

    NEW:
    - You can store processed state in a separate file by passing processed_state_path.
      Then index.json still stores products globally, but processed keys are read/written
      from that separate file.
    """

    def __init__(self, config: dict, processed_state_path: Optional[str] = None) -> None:
        paths_cfg = config.get("paths", {})
        sentinel_cfg = config.get("sentinel", {}).get("cache", {})

        root = Path(paths_cfg["sentinel_data_root"])
        self.cache_dir = root / "cache"
        self.index_path = root / "index.json"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(sentinel_cfg.get("max_size_gb", 100)) * 1024**3
        self.keep_safe = bool(sentinel_cfg.get("keep_safe_after_processing", False))

        self.processed_path: Optional[Path] = None
        if processed_state_path:
            self.processed_path = Path(processed_state_path)
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)

        self._index: Dict[str, Any] = {"products": {}, "processed": {}}
        self._load_index()

        if self.processed_path is not None:
            self._index["processed"] = self._load_processed_external()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "products" in data and "processed" in data:
                if not isinstance(data["products"], dict):
                    data["products"] = {}
                if not isinstance(data["processed"], dict):
                    data["processed"] = {}
                self._index = data
                return

            if isinstance(data, dict):
                products: Dict[str, Any] = {}
                for product_id, meta in data.items():
                    if isinstance(meta, dict) and "safe_id" in meta and "path" in meta:
                        products[product_id] = {
                            "safe_id": meta["safe_id"],
                            "path": meta["path"],
                            "registered_at": float(meta.get("registered_at", time.time())),
                            "last_used": float(meta.get("last_used", time.time())),
                        }
                self._index = {"products": products, "processed": {}}
                self._save_index()
                return

            self._index = {"products": {}, "processed": {}}

        except Exception:
            self._index = {"products": {}, "processed": {}}

    def _save_index(self) -> None:
        """
        Save index.json. If processed_path is used, we DO NOT persist processed keys
        inside index.json (to avoid mixing strategies).
        """
        data = dict(self._index)
        if self.processed_path is not None:
            data["processed"] = {}  

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_processed_external(self) -> Dict[str, bool]:
        if self.processed_path is None:
            return {}

        if not self.processed_path.exists():
            return {}

        try:
            with open(self.processed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                
                return {str(k): bool(v) for k, v in data.items()}
            return {}
        except Exception:
            return {}

    def _save_processed_external(self) -> None:
        if self.processed_path is None:
            return
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(self._index["processed"], f, indent=2)

    def is_processed(self, roi_product_key: str) -> bool:
        return bool(self._index["processed"].get(roi_product_key, False))

    def mark_processed(self, roi_product_key: str) -> None:
        self._index["processed"][roi_product_key] = True

        if self.processed_path is not None:
            self._save_processed_external()
            self._save_index()  
        else:
            self._save_index()

    def unmark_processed(self, roi_product_key: str) -> None:
        if roi_product_key in self._index["processed"]:
            del self._index["processed"][roi_product_key]

            if self.processed_path is not None:
                self._save_processed_external()
                self._save_index()
            else:
                self._save_index()

    def ensure_safe_available(self, product: SentinelProduct) -> str:
        product_id = product.product_id
        safe_id = f"{product_id}.SAFE"
        safe_path = self.cache_dir / safe_id

        if safe_path.exists():
            meta = self._index["products"].get(product_id)
            if isinstance(meta, dict):
                meta["last_used"] = time.time()
            else:
                self._index["products"][product_id] = {
                    "safe_id": safe_id,
                    "path": str(safe_path),
                    "registered_at": time.time(),
                    "last_used": time.time(),
                }
            self._save_index()
            return safe_id

        uuid = resolve_uuid_from_odata(product_id)

        downloader = CopernicusSafeDownloader()
        downloader.download_safe(
            product_id=uuid,
            out_dir=self.cache_dir,
        )

        self._index["products"][product_id] = {
            "safe_id": safe_id,
            "path": str(safe_path),
            "registered_at": time.time(),
            "last_used": time.time(),
        }
        self._save_index()

        return safe_id

    def cleanup_if_needed(self) -> None:
        """
        If cache exceeds max_size_gb, evict oldest SAFEs (by last_used).
        Only runs if keep_safe_after_processing = False.
        """
        if self.keep_safe:
            return

        current_size = self._current_cache_size()
        if current_size <= self.max_size_bytes:
            return

        products_by_age = sorted(
            self._index["products"].items(),
            key=lambda item: float(item[1].get("last_used", 0.0)),
        )

        for product_id, meta in products_by_age:
            safe_id = meta.get("safe_id")
            if not safe_id:
                continue

            path = self.cache_dir / safe_id
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

            self._index["products"].pop(product_id, None)

            current_size = self._current_cache_size()
            if current_size <= self.max_size_bytes:
                break

        self._save_index()

    def _current_cache_size(self) -> int:
        total = 0
        if not self.cache_dir.exists():
            return 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total
